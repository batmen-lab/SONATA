import os, argparse, ot
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from utils.utils import *
from utils.vis import *
import itertools
from types import SimpleNamespace

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
    def __init__(self, params: dict) -> None:
        self.scot_k = params.get("scot_k", 10)
        self.scot_e = params.get("scot_e", 1e-3)
        self.scot_mode = params.get("scot_mode", "distance")
        self.scot_metric = params.get("scot_metric", "euclidean")
        
        self.repeat = params.get("repeat", 10)
        self.n_cluster = params.get("n_cluster", 20)
        self.noise_scale = params.get("noise_scale", 0.2)
        self.n_neighbor = params.get("n_neighbor", 10)
        self.elbow_k_range = params.get("elbow_k_range", 11)
        self.pval_thres = params.get("pval_thres", 1e-2)
        
        self.verbose = params.get("verbose", True)
        
        self.clust_labels = None
        self.spline_data = None

    def diagnose(self, data, save_dir=None):
        self.clust_labels = h_clustering(data, self.n_cluster)
        
        # sonata pipeline
        mat_dict = self.noise_alignment(data, save_dir=save_dir)
        outlier_pairs = self.get_outlier(mat_dict)
        diagnose_result = self.find_ambiguous_groups(data, outlier_pairs)             

        return diagnose_result

    
    def noise_alignment(self, data, refresh=False, save_dir=None):    
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        coupling_iters_path = os.path.join(save_dir, 'coupling_iters')  
        os.makedirs(coupling_iters_path, exist_ok=True)
        
        if (refresh==True) or not (os.path.exists(os.path.join(coupling_iters_path, f"coupling_iter{self.repeat-1}.txt"))):
            np.seterr(under='ignore')
            for iter in range(self.repeat):
                print("---------------SCOT Alignment Iter={}--------------".format(iter))
                self.noise_alignment_scot(data, save_url = os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"))
        
        mat_dict = self.denoise_coupling(data, load_path=save_dir, repeat=range(self.repeat))
        return mat_dict
     
    def noise_alignment_scot(self, data, save_url):
        include_self=True if self.scot_mode=="connectivity" else False
        
        # + noise1: neighborhood noise        
        graph1 = kneighbors_graph(data, n_neighbors=self.n_neighbor, mode='distance', metric='euclidean')
        data1 = data + np.random.normal(loc=0.0, scale=np.mean(graph1.data)*self.noise_scale, size=data.shape)
        Xgraph = kneighbors_graph(data1, n_neighbors=self.n_neighbor, mode=self.scot_mode, metric=self.scot_metric, include_self=include_self)

        # + noise2: add n 2hop neighbors        
        graph, two_hop = get2hop(data, mode=self.scot_mode, metric=self.scot_metric, k=self.n_neighbor)
        Ygraph, _ = add_neighbors(graph.copy(), two_hop, n_neighbor = self.n_neighbor)
        
        # SCOT alignment
        Cx = init_distances(Xgraph)
        Cy = init_distances(Ygraph)
        coupling, _= ot.gromov.entropic_gromov_wasserstein(Cx, Cy, p=ot.unif(data.shape[0]), q=ot.unif(data.shape[0]), 
                                                                loss_fun='square_loss', epsilon=self.scot_e, log=True, verbose=True)
        
        np.savetxt(save_url, coupling)

    def denoise_coupling(self, data, load_path, repeat=[]):
        N, P = data.shape
        dist_mat = np.zeros((N, N))
        coupling_mat = np.zeros((N, N)) 

        unq_clust_label, unq_clust_label_idx = np.unique(self.clust_labels, return_index=True)
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
            clust_coupling = coupling2clustcoupling(coupling, self.clust_labels, ordered_clust_label)
            # 2. calculate freqency matrix
            clust_coupling_copy = clust_coupling.copy()
            clust_coupling_copy[np.where(clust_coupling_copy > 0)] = 1
            freq_mat += clust_coupling_copy
            coupling_mat += coupling 

        # calculate distance mat
        graph, _ = get2hop(data, self.scot_mode, self.scot_metric, k=self.n_neighbor)
        dist_mat = init_distances(graph) 

        # avg cluster-wise coupling & Cx matrix
        clust_coupling_mat = coupling2clustcoupling(coupling_mat, self.clust_labels, ordered_clust_label)
        clust_dist_mat = coupling2clustcoupling(dist_mat, self.clust_labels, ordered_clust_label)

        freq_mat[np.where(freq_mat == 0)] = 1
        clust_coupling_mat = clust_coupling_mat / freq_mat
        
        coupling_mat = clustcoupling2coupling(clust_coupling_mat, self.clust_labels, ordered_clust_label)

        return {"clust_dist_mat": clust_dist_mat, 
                "clust_coupling_mat": clust_coupling_mat,
                "dist_mat": dist_mat, 
                "coupling_mat": coupling_mat}
        
    def get_outlier(self, mat_dict, dist_tol=2e-2, coupling_tol=1e-5, spline_total_iter=1):
        dist_mat = mat_dict["dist_mat"]
        coupling_mat = mat_dict["coupling_mat"]
        assert coupling_mat.shape == dist_mat.shape
        
        dist_mat /= dist_mat.max()
        coupling_mat /= coupling_mat.max()
        coupling_mat[coupling_mat < coupling_tol] = 0
        dist_mat[dist_mat < dist_tol] = 0

        avg_coupling_list = []
        avg_dist_list = []
        clust_pair_list = []

        uniq_clusters = np.unique(self.clust_labels)
        for idx1 in range(len(uniq_clusters)):
            clust_indices1 = np.where(self.clust_labels==idx1)[0]

            for idx2 in range(idx1, len(uniq_clusters)):
                clust_indices2 = np.where(self.clust_labels==idx2)[0]

                coupling_submat = coupling_mat[clust_indices1, :][:, clust_indices2]
                coupling_avg = np.mean(coupling_submat) 

                dist_submat = dist_mat[clust_indices1, :][:, clust_indices2]
                dist_avg = np.mean(dist_submat)

                # nonzero_indices = np.where(coupling_submat > 0)
                # if len(nonzero_indices[0]) <= 0: continue

                avg_coupling_list.append(coupling_avg)
                avg_dist_list.append(dist_avg)
                clust_pair_list.append((idx1, idx2))

        ## Trick: If all farthest neighbors are ambiguous nodes, add "missing zeros" to facilitate spline fitting.
        new_lists = add_missingness(clust_mats=(mat_dict["clust_dist_mat"], mat_dict["clust_coupling_mat"], ),
                                    add_lists = {"avg_dist_list": avg_dist_list, 
                                                "avg_coupling_list": avg_coupling_list, 
                                                "clust_pair_list": clust_pair_list}, 
                                    fit_type="max_nondec_spline")
        avg_dist_list, avg_coupling_list, clust_pair_list = new_lists
           

        avg_coupling_list = np.asarray(avg_coupling_list)
        avg_coupling_list /= np.max(avg_coupling_list)
        avg_dist_list = np.asarray(avg_dist_list)
        clust_pair_list = np.asarray(clust_pair_list)

        sort_indices = np.argsort(avg_dist_list)
        avg_dist_list = avg_dist_list[sort_indices]
        avg_coupling_list = avg_coupling_list[sort_indices]
        clust_pair_list = clust_pair_list[sort_indices]

        include_indices = np.asarray(list(range(len(avg_coupling_list))), dtype=int)
        exclude_indices = np.asarray([], dtype=int)
        geo_thres = np.max(avg_dist_list[clust_pair_list[:,0] == clust_pair_list[:, 1]])
        masked_indices = np.where(avg_dist_list <= geo_thres)[0]
        
        ## fit until no outliers or max iter, in our manuscirpt, we set spline_total_iter=1 for all datasets
        is_outliers = True
        iter = 0
        while iter < spline_total_iter and is_outliers == True:
            currX = avg_dist_list[include_indices]
            currY = avg_coupling_list[include_indices]
    
            ### fit spline: the P% lowest scatters to fit a spline (LinearGAM)
            newY, res, lowest_idx = fit_spline(currX, currY)
    
            ## P-value Calculation: left lowest_idx nodes as neighbors for node i, one-tail test
            p_values = p_value(res, currX, lowest_idx, geo_thres)

            indices = np.where(p_values <= self.pval_thres)[0]
            if self.verbose:
                print("Outlier cluster indices={}".format(indices))

            if len(indices) > 0:
                outlier_indices = include_indices[indices]
                outlier_indices = np.setdiff1d(outlier_indices, masked_indices)
                is_outliers = False if len(outlier_indices) == 0 else True

                include_indices = np.setdiff1d(include_indices, outlier_indices)
                exclude_indices = np.union1d(exclude_indices, outlier_indices)
                if self.verbose:
                    print("iter={}\tinclude_indices={}\texclude_indices={}".format(iter, include_indices, exclude_indices))
            else:
                is_outliers = False

            iter +=1
        if self.verbose:
            print("length of include_indices={}\texclude_indices={}".format(len(include_indices), len(exclude_indices)))
        
        self.spline_data = SimpleNamespace(
            spline_dist = avg_dist_list,
            spline_coupling = avg_coupling_list,
            spline_x = currX,
            spline_y = newY,
            exclude_indices = exclude_indices,
            include_indices = include_indices,            
        )

        return clust_pair_list[exclude_indices]


    def find_ambiguous_groups(self, data, exclude_clust_pairs, K=None,):
        '''
        using semi-supervised-clustering with cannot link
        code: https://github.com/datamole-ai/active-semi-supervised-clustering
        '''
        from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
        # TODO: makesure the return type
        if len(exclude_clust_pairs) == 0:
            result = SimpleNamespace(
                ambiguous_labels=np.array([], dtype=int), 
                ambiguous_idx=np.array([], dtype=int), 
                cannot_links=[]
            )
            return result
        
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
            if self.verbose:
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
            if self.verbose:
                print("Ambiguous group {} = {}".format(class_label, class_indices))

        result = SimpleNamespace(
            ambiguous_labels=ambiguous_labels, 
            ambiguous_idx=ambiguous_idx, 
            cannot_links=cannot_links
        )

        return result
    

def map_ambiguous_groups(data, ambiguous_labels, ambiguous_idx):
    np.seterr(under='ignore')
    N = data.shape[0]
    unambiguous_idx = np.setdiff1d(list(range(N)), ambiguous_idx)
    assert N == len(ambiguous_idx)+len(unambiguous_idx)
    
    geo_mat = geodistance(data)
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
