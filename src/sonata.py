import numpy as np

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp 
from scipy.spatial.distance import cdist

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, PPoly

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans

import itertools
import ot


class sonata(object):
    def __init__(self, kmin=10, sigma=0.1, t=0.1, kmax=200, kmode="distance", kmetric="euclidean", percnt_thres=95, eval_knn=False) -> None:

        self.kmin = kmin
        self.kmax = kmax
        self.kmode = kmode
        self.kmetric = kmetric

        self.sigma = sigma

        self.percnt_thres = percnt_thres
        self.t = t
        self.eval_knn = eval_knn

        self.geo_mat = None
        self.knn = None
        self.l1_mat = None
        self.cell_amat = None
        self.group_amats = None
        # for plt
        self.ambiguous_links = None
        self.ambiguous_nodes = None
        self.cluster_labels = None

        # for elbow methods
        self.K = None
        self.K_yerror = None
        self.K_xstep = None

    def alt_mapping(self, data):
        # # data preprocess
        # if self.norm:
        #     data = self.data_normalize(data)
        #     print('data normalized')
        # if self.pca_k != 0:
        #     data = PCA(n_components=self.pca_k).fit(data).fit_transform(data)
        #     print('data preprocessed by pca')

        # cell-wise ambiguity
        self.construct_graph(data)
        self.l1_mat = self.geo_similarity(geo_dist=self.geo_mat)
        self.cell_ambiguity()

        # group-wise ambiguity
        self.group_amats = self.group_ambiguity(data)

        return self.group_amats
    
    def data_normalize(self, data):
        assert (self.norm in ["l1","l2","max", "zscore"]), "Norm argument has to be either one of 'max', 'l1', 'l2' or 'zscore'."

        if self.norm=="zscore":
            scaler = StandardScaler()
            norm_data = scaler.fit_transform(data)

        else:
            norm_data = normalize(data, norm=self.norm, axis=1)

        return norm_data
    
    def construct_graph(self, data):
        """
        constructing knn graph and calculating geodestic distance

        kmax/kmin: k should be sufficiently large to connect the corresponding k-NN graph
        kmode: 'connectivity'/'distance'
        metric: 'euclidean'/'correlation'
        """
        print('constructing knn graph ...')
        nbrs = NearestNeighbors(n_neighbors=self.kmin, metric=self.kmetric, n_jobs=-1).fit(data)
        knn = nbrs.kneighbors_graph(data, mode = self.kmode)

        connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        while connected_components != 1:
            if self.kmin > np.max((self.kmax, 0.01*len(data))):
                break
            self.kmin += 2
            nbrs = NearestNeighbors(n_neighbors=self.kmin, metric=self.kmetric, n_jobs=-1).fit(data)
            knn = nbrs.kneighbors_graph(data, mode = self.kmode)
            connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        print('final k ={}'.format(self.kmin))

        # calculate the shortest distance between nodes
        dist = sp.csgraph.floyd_warshall(knn, directed=False)
        
        dist_max = np.nanmax(dist[dist != np.inf])
        dist[dist > dist_max] = 2*dist_max
        
        # global max-min normalization
        norm_geo_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

        self.geo_mat = norm_geo_dist
        self.knn = knn
    
    def geo_similarity(self, geo_dist):
        sorted_geo_dist = np.sort(geo_dist, axis=1)
        l1_dist = cdist(sorted_geo_dist, sorted_geo_dist, 'cityblock') / sorted_geo_dist.shape[1]
        return l1_dist
    
    def cell_similarity(self, mat):
        d_matrix = mat/np.power(self.sigma, 2)
        d_matrix_e = np.exp(-d_matrix)
        d_matrix_sum = np.sum(d_matrix_e, axis = 1).reshape(d_matrix_e.shape[0],1)
        cell_amat = d_matrix_e/d_matrix_sum
        # normalize cell-wise ambiguity to 0-1 range
        cell_amat = (cell_amat - np.min(cell_amat, axis=1, keepdims=True)) / (np.max(cell_amat, axis=1, keepdims=True) -
                                                                    np.min(cell_amat, axis=1, keepdims=True))
        return cell_amat

    def cell_ambiguity(self):
        print('calculating cell-wise ambiguity ...')
        # ambiguity
        init_cell_amat = self.cell_similarity(self.l1_mat)

        # ambiguity safeguard
        cell_amat_safe = self.safe_ambiguity(self.l1_mat)

        # ambiguity calibration
        cell_amat_clean = self.fit_spline(cell_amat_safe)

        self.cell_amat = cell_amat_clean

    def safe_ambiguity(self, mat):
        n = self.geo_mat.shape[0]
        geo_mat_shuffled = self.geo_mat.copy().flatten()
        np.random.shuffle(geo_mat_shuffled)
        geo_mat_shuffled = geo_mat_shuffled.reshape(self.geo_mat.shape)
        l1_mat_shuffled = self.geo_similarity(geo_mat_shuffled)

        percent_l1 = np.percentile(mat, q=self.percnt_thres, axis=1)
        percent_l1_shuffled = np.percentile(l1_mat_shuffled, q=self.percnt_thres, axis=1)
        control_vec = percent_l1 * np.sign(percent_l1 - percent_l1_shuffled)
        l1_mat_aug = np.zeros((n+1, n+1))
        l1_mat_aug[:n, :n] = mat
        l1_mat_aug[n, :n] = control_vec
        l1_mat_aug[:n, n] = control_vec
        cell_amat_safe = self.cell_similarity(l1_mat_aug)

        cell_vec_aug = cell_amat_safe[:n, n]
        cell_amat_safe = cell_amat_safe[:n, :n]
        # Ambiguous value should not be higher than baseline(cell_vec_aug)
        for node_idx in range(n):
            cell_amat_safe[node_idx, :] = np.maximum(cell_amat_safe[node_idx, :], cell_vec_aug[node_idx])
        return cell_amat_safe

    def fit_spline(self, cell_amat, r=10):
        cell_amat_clean = np.copy(cell_amat)
        for node_id in range(cell_amat.shape[0]):
            tuple_arr = list(zip(self.geo_mat[node_id, :], cell_amat[node_id, :], ))
            tuple_arr.sort(key=lambda x: x[0])

            geo_arr = np.asarray([x[0] for x in tuple_arr], dtype=float)
            Y_arr = np.asarray([x[1] for x in tuple_arr], dtype=float)
            X_arr = np.asarray(list(range(len(tuple_arr))), dtype=float)
 
            # smoothen cell-wise ambiguity by averaging nearest neighbors with radius = r
            Y_arr_smooth = np.copy(Y_arr)
            for mid_idx in range(len(Y_arr)):
                Y_arr_smooth[mid_idx] = np.mean(Y_arr[max(0, mid_idx-r):min(len(Y_arr), mid_idx+r)])

            # remove noise for all-ambiguous datasets
            Y_arr_smooth = np.where(Y_arr_smooth < 0.1, np.round(Y_arr_smooth, 1), Y_arr_smooth)
            Y_arr = Y_arr_smooth

            # fitting spline 
            spline = UnivariateSpline(X_arr, Y_arr, k = 4) 

            # find minimal of curve
            dv1 = spline.derivative(n = 1) # 1st derivative
            y_dv1 = dv1(X_arr)
            tck = splrep(X_arr, y_dv1)
            ppoly = PPoly.from_spline(tck)
            dv1_roots = ppoly.roots(extrapolate=False) # 1st derivative = 0
            dv2 = spline.derivative(n = 2) # 2nd derivative

            # remove ambiguous neighbors by 1st curve min for ambiguity
            curve_m = np.where(dv2(dv1_roots) > 0)[0]

            if len(curve_m) > 0:
                idx = int(dv1_roots[curve_m[0]])
                cell_amat_clean[node_id, self.geo_mat[node_id, :] <= geo_arr[idx]] = np.min(cell_amat_clean[node_id, :])
        
        return cell_amat_clean


    def group_ambiguity(self, data):
        print('calculating group-wise ambiguity ...')
        ambiguous_nodes = self.select_ambiguous_nodes()
        unambiguous_nodes = np.setdiff1d(list(range(data.shape[0])), ambiguous_nodes)
        self.ambiguous_nodes = ambiguous_nodes

        if len(ambiguous_nodes) == 0:
            print("There is no ambiguous")
            map_mat = None
        else:
            ambiguous_node_groups = self.find_ambiguous_groups(data, ambiguous_nodes)
            map_mat = self.map_uncertain_groups(data, ambiguous_node_groups, unambiguous_nodes)

        return map_mat
        

    def select_ambiguous_nodes(self):
        cell_amat_copy = self.cell_amat.copy()
        n = cell_amat_copy.shape[0]
        cell_amat_copy[cell_amat_copy <= self.t] = 0
        cell_amat_sum_arr = cell_amat_copy.sum(axis=1)
        ambiguous_nodes = np.where(cell_amat_sum_arr > 0)[0]
        return ambiguous_nodes

    def find_ambiguous_groups(self, data, ambiguous_nodes):
        '''
        using semi-supervised-clustering with cannot link
        code: https://github.com/datamole-ai/active-semi-supervised-clustering
        '''
        
        data_mat_ambiguous = data[ambiguous_nodes, :]
        cell_amat_ambiguous = self.cell_amat[ambiguous_nodes, :][:, ambiguous_nodes]

        # uncertainty threshold
        ambiguous_indices = np.where(cell_amat_ambiguous > self.t)
        
        # all cell-wise ambiguity
        cannot_links = list(zip(ambiguous_indices[0], ambiguous_indices[1]))
        
        # group ambiguous nodes  -- elbow method to choose k
        print('deciding best k for clustering ...')
        if self.K == None:
            self.elbow_k(data_mat_ambiguous, cannot_links, k_range=10)
            print('K = {} groups choosen by elbow method'.format(self.K))
        clusterer = PCKMeans(n_clusters=self.K)
        clusterer.fit(data_mat_ambiguous, cl=cannot_links)
        labels = np.asarray(clusterer.labels_, dtype=int)

        ambiguous_node_groups = []
        for class_label in np.unique(labels):
            class_indices = np.where(labels == class_label)[0]
            ambiguous_node_groups.append(ambiguous_nodes[class_indices])
        
        self.ambiguous_links = cannot_links
        self.cluster_labels = labels
        return ambiguous_node_groups

    # elbow k for find_ambiguous_groups    
    def elbow_k(self, data, cannot_link, k_range = 10):
        from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
        cl_arr = np.transpose(np.array(cannot_link))

        # calculate # of ambiguity pairs for each k
        y_error = []
        for k in range(1, k_range):
            clusterer = PCKMeans(n_clusters=k)
            clusterer.fit(data, cl=cannot_link)

            labels = clusterer.labels_
            n_pairs = 0
            for label in np.unique(labels):
                this_cluster = np.where(labels == label)[0]
                n_pairs += np.sum(np.logical_and(np.isin(cl_arr[0], this_cluster), 
                                                            np.isin(cl_arr[1], this_cluster)))
            y_error.append(n_pairs)

        # normalize error
        y_error = np.array(y_error)/np.square(data.shape[0])
        # normalize x axis
        x_step = 1/data.shape[0]

        # choose the best k by 2nd derivative
        best_k = np.argmax(second_grad(y_error, step = x_step)) + 2

        self.K = best_k
        self.K_yerror = y_error
        self.K_xstep = x_step

    def map_uncertain_groups(self, data, ambiguous_node_groups, unambiguous_nodes):
        np.seterr(under='ignore')
        ambiguous_nodes = np.asarray([], dtype=int)
        for group in ambiguous_node_groups:
            ambiguous_nodes = np.concatenate((ambiguous_nodes, group))
        assert data.shape[0] == len(ambiguous_nodes)+len(unambiguous_nodes)

        ### evaluate valid group
        if self.eval_knn:
            valid_perm = eval_valid_group_knn(ambiguous_node_groups, unambiguous_nodes, self.knn)
        else:
            valid_perm= list(itertools.permutations(range(len(ambiguous_node_groups))))[1:]
        print("all vaild perms are: ", valid_perm)

        assert len(ambiguous_node_groups) > 1
        for perms in valid_perm:
            # keep the diagonal for original nodes
            map_mat = np.zeros_like(self.geo_mat)
            print("perms: ", perms)

            # keep the diagonal for certain nodes
            for node in unambiguous_nodes: map_mat[node, node] = 1.0        

            for i in range(len(ambiguous_node_groups)):
                group_idx1 = i
                group_idx2 = perms[i]

                node_group1 = ambiguous_node_groups[group_idx1]
                node_group2 = ambiguous_node_groups[group_idx2]

                if group_idx1 == group_idx2:
                    # keep the diagonal for unchanged nodes
                    for node in np.concatenate((node_group1, node_group2)): map_mat[node, node] = 1.0 
                else:
                    print('changed group id: ', group_idx1, group_idx2)
                    # aligning ambiguous group pairs
                    cost = 1-self.cell_amat
                    cost = cost[node_group1, :][:, node_group2]
                    p1 = ot.unif(len(node_group1))
                    p2 = ot.unif(len(node_group2))
                    T = ot.emd(p1, p2, cost)
                    T = normalize(T, norm='l1', axis=1, copy=False)

                    # fill in the T matrix for mapped nodes 
                    for group1_idx in range(len(node_group1)):
                        group1_node = node_group1[group1_idx]
                        for group2_idx in range(len(node_group2)):
                            group2_node = node_group2[group2_idx]
                            map_mat[group1_node, group2_node] = T[group1_idx, group2_idx]

            yield map_mat




# second graduate for k elbow
def second_grad(k_arr, step):
    # first grad
    first_grad = (k_arr[1:] - k_arr[:-1])/step 
    # print('first_grad = {}\nangles = {}'.format(first_grad, np.degrees(np.arctan(first_grad))))

    # 2nd grad
    second_grad = (first_grad[1:] - first_grad[:-1])/(1+first_grad[1:]*first_grad[:-1])
    second_grad = np.arctan(np.abs(second_grad))
    # print('2nd_grad = {}\ndelta angle = {}'.format(second_grad, np.degrees(second_grad)))

    return second_grad

def eval_valid_group_knn(ambiguous_node_groups, unambiguous_nodes, knn):
    knn = knn.tocoo()
    knn_arr = knn.toarray()

    connected_groups = []
    for tup in list(itertools.combinations(range(len(ambiguous_node_groups)), 2)):
        group_idx1, group_idx2 = tup[0], tup[1]
        node_group1 = ambiguous_node_groups[group_idx1]
        node_group2 = ambiguous_node_groups[group_idx2]
        sub_knn_arr = knn_arr[node_group1, :][:, node_group2]
        if np.sum(sub_knn_arr) > 0.0:
            connected_groups.append(tup)
    print("connected_groups_uncertain: ", connected_groups)

    for group_idx1 in range(len(ambiguous_node_groups)):
        group_idx2 = -1 # certain nodes
        node_group1 = ambiguous_node_groups[group_idx1]
        sub_knn_arr = knn_arr[node_group1, :][:, unambiguous_nodes]
        if np.sum(sub_knn_arr) > 0:
            connected_groups.append((group_idx1, group_idx2))
    connected_groups = np.array(connected_groups)
    print('connected_groups_all: ', connected_groups)   
    
    valid_perm = []
    sorted_connected_groups = np.sort(connected_groups, axis = 1)
    for perms in list(itertools.permutations(range(len(ambiguous_node_groups))))[1:]:
        new_connected_groups = np.copy(connected_groups)
        for i in range(len(ambiguous_node_groups)):
            group_idx1 = i
            group_idx2 = perms[i]
            if group_idx1 != group_idx2:
                new_connected_groups[connected_groups == group_idx1] = group_idx2  
        sorted_new_connected_groups = np.sort(new_connected_groups, axis = 1)  

        if sorted(sorted_connected_groups.tolist()) == sorted(sorted_new_connected_groups.tolist()):
            valid_perm.append(perms)

    return valid_perm
