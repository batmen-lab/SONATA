import os, argparse, ot
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from utils.utils import *
import itertools

class sonata(object):
    """
    SONATA algorithm for disambiguating manifold alignment of single-cell data
    https://www.biorxiv.org/content/10.1101/2023.10.05.561049v3
    
    Input for SONATA: data in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
    Basic Usage: 
        sn = sonata.sonata(params)
        ambiguous_labels, ambiguous_idx = sn.diagnose(data)
    
    For more examples, refer to the examples folder.
    """
    def __init__(self, params) -> None:
        self.scot_k = params["scot_k"]
        self.scot_mode = params["scot_mode"]
        self.scot_metric = params["scot_metric"]
        
        self.repeat = params["repeat"]
        self.n_cluster = params["n_cluster"]
        self.noise_scale = params["noise_scale"]
        self.n_dila = params["n_dila"]
        self.n_neighbor = params["n_neighbor"]
        self.n_bin = params["n_bin"]
        self.elbow_k_range = params["elbow_k_range"]
        self.spline_iter = params["spline_iter"]
        self.pval_thres = params["pval_thres"]
        
        self.save_dir = params["save_dir"]

    def diagnose(self, data):
        cx_mat, coupling_mat, clust_cx_mat, clust_coupling_mat = self.noise_alignment(data)

        exclude_pairs = self.get_outlier(clust_coupling_mat, clust_cx_mat, coupling_mat, cx_mat)
        
        ambiguous_labels, ambiguous_idx = self.find_ambiguous_groups(data, exclude_pairs)                

        return ambiguous_labels, ambiguous_idx

    
    def noise_alignment(self, data, refresh=False):    
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        coupling_iters_path = os.path.join(self.save_dir, 'coupling_iters')  
        os.makedirs(coupling_iters_path, exist_ok=True)
        
        if (refresh==True) or not (os.path.exists(os.path.join(coupling_iters_path, f"coupling_iter{self.repeat-1}.txt"))):
            np.seterr(under='ignore')
            self.noise_alignment_scot(data, self.repeat, save_dir = coupling_iters_path)

        self.clust_labels = h_clustering(data, self.n_cluster)
        cx_mat, coupling_mat, clust_cx_mat, clust_coupling_mat = self.denoise_coupling(data, load_path=self.save_dir, repeat=range(self.repeat))
        return cx_mat, coupling_mat, clust_cx_mat, clust_coupling_mat
    
    
    def noise_alignment_scot(self, data, repeat, save_dir):
        graph, two_hop = get2hop(data, mode=self.scot_mode, metric=self.scot_metric, k=self.n_neighbor)
        os.makedirs(save_dir, exist_ok=True)

        for iter in range(repeat):
            save_url = os.path.join(save_dir, f"coupling_iter{iter}.txt")
            print("---------------SCOT Alignment Iter={}--------------".format(iter))

            # + noise1: neighborhood noise
            # + noise2: add n 2hop neighbors
            include_self=True if self.scot_mode=="connectivity" else False
            
            graph1 = kneighbors_graph(data, n_neighbors=self.n_neighbor, mode='distance', metric='euclidean')
            data1 = data + np.random.normal(loc=0.0, scale=np.mean(graph1.data)*self.noise_scale, size=data.shape)

            Xgraph = kneighbors_graph(data1, n_neighbors=self.n_neighbor, mode=self.scot_mode, metric=self.scot_metric, include_self=include_self)
            Ygraph, _ = add_neighbors(graph.copy(), two_hop, n_dila = self.n_dila)
            Cx = init_distances(Xgraph)
            Cy = init_distances(Ygraph)

            coupling, _= ot.gromov.entropic_gromov_wasserstein(Cx, Cy, p=ot.unif(data.shape[0]), q=ot.unif(data.shape[0]), 
                                                                    loss_fun='square_loss', epsilon=1e-3, log=True, verbose=True)
            
            np.savetxt(save_url, coupling)


    def denoise_coupling(self, data, load_path, repeat=[]):
        N, P = data.shape
        cx_mat = np.zeros((N, N))
        coupling_mat = np.zeros((N, N)) 
        clust_label = self.clust_labels

        unq_clust_label, unq_clust_label_idx = np.unique(clust_label, return_index=True)
        ordered_clust_label = unq_clust_label[np.argsort(unq_clust_label_idx)]
        freq_mat = np.zeros((len(unq_clust_label), len(unq_clust_label)))

        coupling_iters_path = os.path.join(load_path, 'coupling_iters')

        for iter in repeat:
            print("---------------Coupling Denoising Iter={}--------------".format(iter))
            print("Load_path = {}".format(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt")))
            coupling = np.loadtxt(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"))

            ## denoise
            # 1. set coupling < average = 0
            coupling[np.where(coupling < np.mean(coupling))] = 0
            clust_coupling = coupling2clustcoupling(coupling, clust_label, ordered_clust_label)
            # 2. calculate freqency matrix
            clust_coupling_copy = clust_coupling.copy()
            clust_coupling_copy[np.where(clust_coupling_copy > 0)] = 1
            freq_mat += clust_coupling_copy
            coupling_mat += coupling 

        # calculate cx_mat
        graph, _ = get2hop(data, self.scot_mode, self.scot_metric, k=self.n_neighbor)
        cx_mat = init_distances(graph) 

        # avg cluster-wise coupling & Cx matrix
        clust_coupling_mat = coupling2clustcoupling(coupling_mat, clust_label, ordered_clust_label)
        clust_cx_mat = coupling2clustcoupling(cx_mat, clust_label, ordered_clust_label)

        freq_mat[np.where(freq_mat == 0)] = 1
        clust_coupling_mat = clust_coupling_mat / freq_mat
        coupling_mat = clustcoupling2coupling(clust_coupling_mat, clust_label, ordered_clust_label)

        return cx_mat, coupling_mat, clust_cx_mat, clust_coupling_mat
        

    def get_outlier(self, clust_coupling_mat, clust_cx_mat, coupling_mat, cx_mat, cx_tol=2e-2, coupling_tol=1e-5):
        assert coupling_mat.shape == cx_mat.shape
        N = coupling_mat.shape[0]

        cx_mat /= cx_mat.max()
        coupling_mat /= coupling_mat.max()
        coupling_mat[coupling_mat < coupling_tol] = 0
        cx_mat[cx_mat < cx_tol] = 0

        avg_coupling_list = []
        avg_cx_list = []
        clust_pair_list = []

        uniq_clusters = np.unique(self.clust_labels)
        for idx1 in range(len(uniq_clusters)):
            clust_indices1 = np.where(self.clust_labels==idx1)[0]

            for idx2 in range(idx1, len(uniq_clusters)):
                clust_indices2 = np.where(self.clust_labels==idx2)[0]

                coupling_submat = coupling_mat[clust_indices1, :][:, clust_indices2]
                coupling_avg = np.mean(coupling_submat) 

                cx_submat = cx_mat[clust_indices1, :][:, clust_indices2]
                cx_avg = np.mean(cx_submat)

                # nonzero_indices = np.where(coupling_submat > 0)
                # if len(nonzero_indices[0]) <= 0: continue

                avg_coupling_list.append(coupling_avg)
                avg_cx_list.append(cx_avg)
                clust_pair_list.append((idx1, idx2))

        # find missing zeros to help with denoising
        x_arr, y_arr, pred_Y = fit_missingzeros(clust_cx_mat, clust_coupling_mat, n_bin=self.n_bin, fit_type="max_nondec_spline")
        diff_y = pred_Y - y_arr

        # add missingness
        for idx in range(len(x_arr)): 
            nums_zero = max(0, int(diff_y[idx]))
            avg_cx_list += [x_arr[idx]] * nums_zero
            # keep the same length as the avg_cx_list
            avg_coupling_list += [0] * nums_zero
            clust_pair_list += [(-1, -2)] * nums_zero

        avg_coupling_list = np.asarray(avg_coupling_list)
        avg_coupling_list /= np.max(avg_coupling_list)
        avg_cx_list = np.asarray(avg_cx_list)
        clust_pair_list = np.asarray(clust_pair_list)

        sort_indices = np.argsort(avg_cx_list)
        avg_cx_list = avg_cx_list[sort_indices]
        avg_coupling_list = avg_coupling_list[sort_indices]
        clust_pair_list = clust_pair_list[sort_indices]

        include_indices = np.asarray(list(range(len(avg_coupling_list))), dtype=int)
        exclude_indices = np.asarray([], dtype=int)
        geo_thres = np.max(avg_cx_list[clust_pair_list[:,0] == clust_pair_list[:, 1]])
        masked_indices = np.where(avg_cx_list <= geo_thres)[0]
        
        ## fit until no outliers or max iter
        is_outliers = True
        iter = 0
        while iter < self.spline_iter and is_outliers == True:
            currX = avg_cx_list[include_indices]
            currY = avg_coupling_list[include_indices]
    
            ### fit spline: the P% lowest scatters to fit a spline (LinearGAM)
            newY, res, lowest_idx = fit_spline(currX, currY, grid=0.1)
    
            ## P-value Calculation: left lowest_idx nodes as neighbors for node i, one-tail test
            p_values = p_value(res, currX, lowest_idx, geo_thres)

            indices = np.where(p_values <= self.pval_thres)[0]
            print("Outlier cluster indices={}".format(indices))

            if len(indices) > 0:
                outlier_indices = include_indices[indices]
                outlier_indices = np.setdiff1d(outlier_indices, masked_indices)
                is_outliers = False if len(outlier_indices) == 0 else True

                include_indices = np.setdiff1d(include_indices, outlier_indices)
                exclude_indices = np.union1d(exclude_indices, outlier_indices)
                print("iter={}\tinclude_indices={}\texclude_indices={}".format(iter, include_indices, exclude_indices))
            else:
                is_outliers = False

            iter +=1
        print("length of include_indices={}\texclude_indices={}".format(len(include_indices), len(exclude_indices)))
        return clust_pair_list[exclude_indices]


    def find_ambiguous_groups(self, data, exclude_clust_pairs, K=None,):
        '''
        using semi-supervised-clustering with cannot link
        code: https://github.com/datamole-ai/active-semi-supervised-clustering
        '''
        from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
        if len(exclude_clust_pairs) == 0:
            return [], np.array([], dtype=int), np.array([], dtype=int), []
        
        # calculate ambiguous nodes
        ambiguous_idx = []
        for clust_pair in exclude_clust_pairs:
            ambiguous_idx += np.where((self.clust_labels == clust_pair[0]) | (self.clust_labels == clust_pair[1]))[0].tolist()
        ambiguous_idx = np.sort(list(set(ambiguous_idx)))

        ambiguous_data = data[ambiguous_idx, :]

        # all cell-wise ambiguity
        cannot_links = exclude_clust_pairs # [[], []]
        cannot_links = []
        for clust_pair in exclude_clust_pairs:
            pf = np.where(self.clust_labels[ambiguous_idx] == clust_pair[0])[0]
            ps = np.where(self.clust_labels[ambiguous_idx] == clust_pair[1])[0]
            cannot_links.append(np.array(np.meshgrid(pf, ps)).T.reshape(-1, 2))
        cannot_links = np.vstack(cannot_links).tolist()

        # group ambiguous nodes  -- elbow method to choose k
        print('deciding best k for clustering ...')
        if K == None:
            K, y_error, x_step = elbow_k(ambiguous_data, cannot_links, k_range=self.elbow_k_range)
            print('K = {} groups choosen by elbow method'.format(K))
            
        clusterer = PCKMeans(n_clusters=K)
        clusterer.fit(ambiguous_data, cl=cannot_links)
        labels = np.asarray(clusterer.labels_, dtype=int)
    
        ambiguous_labels = np.empty_like(labels)
        for clust_label in np.unique(self.clust_labels[ambiguous_idx]):
            clust_idx = np.where(self.clust_labels[ambiguous_idx] == clust_label)[0]
            unique, counts = np.unique(labels[clust_idx], return_counts=True)
            most_freq_cls = unique[np.argmax(counts)]
            ambiguous_labels[clust_idx] = most_freq_cls

        ambiguous_idx_groups = []
        for class_label in np.unique(ambiguous_labels):
            class_indices = np.where(ambiguous_labels == class_label)[0]
            ambiguous_idx_groups.append(class_indices)
            print("Ambiguous group {} = {}".format(class_label, class_indices))
        
        self.cannot_links = cannot_links

        return ambiguous_labels, ambiguous_idx
    

def map_ambiguous_groups(data, geo_mat, ambiguous_labels, ambiguous_idx):
    np.seterr(under='ignore')
    N = data.shape[0]
    unambiguous_idx = np.setdiff1d(list(range(N)), ambiguous_idx)
    assert N == len(ambiguous_idx)+len(unambiguous_idx)
    
    uniq_groups = np.unique(ambiguous_labels)

    ### evaluate valid group
    valid_perm= list(itertools.permutations(range(len(uniq_groups))))[1:]
    # print("all perms = {} ".format(valid_perm))

    ### calculate the cost matrix
    cost_mat = cdist(np.sort(geo_mat, axis=1), np.sort(geo_mat, axis=1), 'cityblock') / geo_mat.shape[1] 
           
    assert len(uniq_groups) > 1
    for perms in valid_perm:
        # keep the diagonal for original nodes
        map_mat = np.zeros((N, N))
        # print("perms: ", perms)

        # keep the diagonal for unambiguous nodes
        for node in unambiguous_idx: map_mat[node, node] = 1.0        

        for i in range(len(uniq_groups)):
            group_idx1 = i
            group_idx2 = perms[i]

            node_group1 = ambiguous_idx[ambiguous_labels==group_idx1]
            node_group2 = ambiguous_idx[ambiguous_labels==group_idx2]

            if group_idx1 == group_idx2:
                # keep the diagonal for unchanged nodes
                for node in np.concatenate((node_group1, node_group2)): map_mat[node, node] = 1.0 
            else:
                # print('changed group id: ', group_idx1, group_idx2)
                # print('{} = {}'.format(group_idx1, node_group1) )
                # print('{} = {}'.format(group_idx2, node_group2) )
                
                # aligning ambiguous group pairs
                cost = cost_mat[node_group1, :][:, node_group2]
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
