import os, argparse, ot
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from utils.utils import *
from utils.vis import *
import itertools
from types import SimpleNamespace
from tqdm import tqdm
from quantizedGW import *

class sonata(object):
    """
    SONATA algorithm for disambiguating manifold alignment of single-cell data
    https://www.biorxiv.org/content/10.1101/2023.10.05.561049v3
    
    Input for SONATA: data in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
    Basic Usage: 
        sn = sonata.sonata(noise_scale=0.2, k=10)
        DiagnoseResult = sn.diagnose(data)
    
        If you want to save the intermediate OT results, you can specify a save_dir:
        DiagnoseResult = sn.diagnose(data, save_dir="path/to/save_dir")
    
    For more examples, refer to the examples folder.
    
    params: 
        - **noise_scale**: 
            The scale of gaussian noise added to generate variational versions of the manifold. Default: 0.2.
        - **n_neighbor**: 
            Number of neighbors when constructing noise manifold. Default: 10.  
        - **mode**:
            Mode for constructing the graph. Options: "connectivity" or "distance". Default: "connectivity".
        - **metric**:
            Metric to use for distance computation. Default: "correlation".
        - **e**:
            Coefficient of the entropic regularization term in the objective function of OT formulation. Default: 1e-3.
        - **repeat**:
            Number of iterations for alignment. Default: 10.
        - **n_cluster**: 
            Number of cell groups used in hierarchical clustering to achieve a smooth and efficient spline fit. Recommended: n_cluster <= $\sqrt{n\_samples}$. Default: 20.
        - **pval_thres**: 
            Threshold value for p-value thresholding. Default: 1e-2.
        - **elbow_k_range**: 
            The range of constrained cluster numbers used by the elbow method to determine the optimal cluster count. Default: 11.            
        - **scalableOT**:
            If True, uses the scalable version of OT. Default: False.
        - **scale_sample_rate**:
            The sample rate for the scalable version of OT. Default: 0.1.
        - **verbose**:
            If True, prints the progress of the algorithm. Default: True.

    data: 
        A NumPy array or matrix where rows correspond to samples and columns correspond to features.    
    
    return:
        An object of SimpleNamespace containing the following attributes:
        - ambiguous_labels: A numpy array of ambiguous group labels for ambiguous samples.
        - ambiguous_idx: A numpy array of indices of ambiguous samples.
        - cannot_links: A list of ambiguous sample pairs.
        
    """
    def __init__(self, noise_scale=0.2, n_neighbor=10, e=1e-3, mode="connectivity", metric="correlation", repeat=10, n_cluster=20,
                 elbow_k_range=11, pval_thres=1e-2, scalableOT=False, scale_sample_rate=0.1, seed=42, verbose=True) -> None:
        self.noise_scale = noise_scale
        self.n_neighbor = n_neighbor
        self.e = e
        self.mode = mode
        self.metric = metric
        
        self.repeat = repeat
        self.n_cluster = n_cluster
        self.elbow_k_range = elbow_k_range
        self.pval_thres = pval_thres
        
        self.verbose = verbose
        self.scalableOT = scalableOT
        self.scale_sample_rate = scale_sample_rate
        self.seed = seed

        self.clust_labels = None
        self.spline_data = None
        
        if self.verbose:
            print("===========================")
            print("n_neighbor={}, noise_scale={}, scaleOT = {}".format(
                self.n_neighbor, self.noise_scale, self.scalableOT))

    def diagnose(self, data, save_dir=None):
        set_seed(self.seed)
        self.clust_labels = h_clustering(data, self.n_cluster)
        
        # sonata pipeline
        mat_dict = self.noise_alignment(data, save_dir=save_dir)
        outlier_pairs = self.get_outlier(mat_dict)
        diagnose_result = self.find_ambiguous_groups(data, outlier_pairs)          
                          
        return diagnose_result
   
    def noise_alignment(self, data, refresh=True, save_dir=None):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            coupling_iters_path = os.path.join(save_dir, 'coupling_iters')  
            os.makedirs(coupling_iters_path, exist_ok=True)
        
        if (refresh==False) and (os.path.exists(os.path.join(coupling_iters_path, f"coupling_iter{self.repeat-1}.txt"))):
            assert save_dir is not None, "Please provide a save_dir to load the coupling matrices."
            coupling_list = []
            for iter in range(self.repeat):
                coupling = np.loadtxt(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"))
                coupling_list.append(coupling)
        else:
            np.seterr(under='ignore')
            coupling_list = []
            for iter in tqdm(range(self.repeat)):
                if self.verbose: print("---------------OT Alignment Iter={}--------------".format(iter))
                
                coupling = self.noise_alignment_ot(data)
                coupling_list.append(coupling)  
                
                if save_dir:
                    np.savetxt(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"), coupling)
                
        mat_dict = self.denoise_coupling(data, coupling_list, repeat=range(self.repeat))
        return mat_dict
    
    def noise_alignment_ot(self, data):
        include_self=True if self.mode=="connectivity" else False

        # + noise1: neighborhood noise               
        stds = data.std(axis=0)
        noise = np.random.normal(loc=0.0, scale=stds * self.noise_scale, size=data.shape)
        data1 = data + noise
        Xgraph = kneighbors_graph(data1, n_neighbors=self.n_neighbor, mode=self.mode, metric=self.metric, include_self=include_self)
        # + noise2: add n 2hop neighbors        
        graph, two_hop = get2hop_fast(data, mode=self.mode, metric=self.metric, k=self.n_neighbor)
        Ygraph, _ = add_neighbors_fast(graph.copy(), two_hop, n_neighbor = self.n_neighbor)
        # graph, two_hop = get2hop(data, mode=self.mode, metric=self.metric, k=self.n_neighbor)
        # Ygraph, _ = add_neighbors(graph.copy(), two_hop, n_neighbor = self.n_neighbor)
    
        # build graph
        Cx = init_distances(Xgraph)
        Cy = init_distances(Ygraph)
           
        if self.scalableOT:
            ## a scalable version: quantizedGW
            node_subset1 = list(set(sample(list(range(data.shape[0])), int(self.scale_sample_rate*len(data)))))    
            node_subset2 = list(set(sample(list(range(data.shape[0])), int(self.scale_sample_rate*len(data))))) 
            coupling = compressed_gw(Cx,Cy,p1=ot.unif(data.shape[0]),p2=ot.unif(data.shape[0]),
                                                node_subset1=node_subset1,node_subset2=node_subset2,
                                                verbose = True,return_dense = True)
        else:
            coupling, _= ot.gromov.entropic_gromov_wasserstein(Cx, Cy, p=ot.unif(data.shape[0]), q=ot.unif(data.shape[0]), 
                                                                    loss_fun='square_loss', epsilon=self.e, log=True, verbose=self.verbose)
    

        return coupling

    def denoise_coupling(self, data, coupling_list, repeat=[]):
        N, P = data.shape
        dist_mat = np.zeros((N, N))
        coupling_mat = np.zeros((N, N)) 

        unq_clust_label, unq_clust_label_idx = np.unique(self.clust_labels, return_index=True)
        ordered_clust_label = unq_clust_label[np.argsort(unq_clust_label_idx)]
        freq_mat = np.zeros((len(unq_clust_label), len(unq_clust_label)))

        for iter in repeat:
            if self.verbose: print("---------------Coupling Denoising Iter={}--------------".format(iter))
            
            coupling = coupling_list[iter]

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
        graph, _ = get2hop(data, self.mode, self.metric, k=self.n_neighbor)
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
            
            if self.verbose: print("Outlier cluster indices={}".format(indices))

            if len(indices) > 0:
                outlier_indices = include_indices[indices]
                outlier_indices = np.setdiff1d(outlier_indices, masked_indices)
                is_outliers = False if len(outlier_indices) == 0 else True

                include_indices = np.setdiff1d(include_indices, outlier_indices)
                exclude_indices = np.union1d(exclude_indices, outlier_indices)
                
                if self.verbose: print("iter={}\tinclude_indices={}\texclude_indices={}".format(iter, include_indices, exclude_indices))
            else:
                is_outliers = False

            iter +=1
            
        if self.verbose: print("length of include_indices={}\texclude_indices={}".format(len(include_indices), len(exclude_indices)))
        
        self.spline_data = SimpleNamespace(
            spline_dist = avg_dist_list,
            spline_coupling = avg_coupling_list,
            spline_x = currX,
            spline_y = newY,
            exclude_indices = exclude_indices,
            include_indices = include_indices,   
            clust_pair_list = clust_pair_list,         
        )

        return clust_pair_list[exclude_indices]


    def find_ambiguous_groups(self, data, exclude_clust_pairs, K=None,):
        '''
        using semi-supervised-clustering with cannot link
        code: https://github.com/datamole-ai/active-semi-supervised-clustering
        '''
        from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
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
        if self.verbose: print('deciding best k for clustering ...')
        
        if K == None:
            K, y_error, x_step = elbow_k(ambiguous_data, cannot_links, k_range=self.elbow_k_range)
            
            if self.verbose: print('K = {} groups choosen by elbow method'.format(K))
            
        cluster = PCKMeans(n_clusters=K)
        cluster.fit(ambiguous_data, cl=cannot_links)
        labels = np.asarray(cluster.labels_, dtype=int)
    
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
            
            if self.verbose: print("Ambiguous group {} = {}".format(class_label, class_indices))

        result = SimpleNamespace(
            ambiguous_labels=ambiguous_labels, 
            ambiguous_idx=ambiguous_idx, 
            cannot_links=cannot_links
        )

        return result


    def diagnose_by_groups(self, data, save_dir=None, refresh=False):
        """This function is similar to the `diagnose` function but first groups the coupling matrices 
            before diagnosing ambiguity using the SONATA method.
        """
        set_seed(self.seed)
        self.clust_labels = h_clustering(data, self.n_cluster)
        
        # step 1: generate mapping data if not exists
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            coupling_iters_path = os.path.join(save_dir, 'coupling_iters')  
            os.makedirs(coupling_iters_path, exist_ok=True)

        if (refresh==False) and (os.path.exists(os.path.join(coupling_iters_path, f"coupling_iter{self.repeat-1}.txt"))):
            assert save_dir is not None, "Please provide a save_dir to load the coupling matrices."
            coupling_list = []
            for iter in range(self.repeat):
                coupling = np.loadtxt(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"))
                coupling_list.append(coupling)
                plt_heatmap(np.log(coupling), title=f"coupling_iter{iter}", save_url=os.path.join(coupling_iters_path, f"logcoupling_iter{iter}.png"))
        else:
            np.seterr(under='ignore')
            coupling_list = []
            for iter in tqdm(range(self.repeat)):
                if self.verbose: print("---------------OT Alignment Iter={}--------------".format(iter))
                
                coupling = self.noise_alignment_ot(data)
                coupling_list.append(coupling)  
                
                if save_dir:
                    np.savetxt(os.path.join(coupling_iters_path, f"coupling_iter{iter}.txt"), coupling)
                                       
        # step2: group coupling matrices       
        coupling_clusters = self.group_couplings(coupling_list)
        # find the diagonal cluster
        best_cluster = self.search_best_diagonal(coupling_clusters, data, coupling_list)
        
        group_diagnose_result = []
        for cluster, repeats in coupling_clusters.items():
            if best_cluster == cluster: 
                continue
            else:
                repeats_total = list(repeats) + list(coupling_clusters[best_cluster])
                
            # denoising coupling matrices for each group
            mat_dict = self.denoise_coupling(data, coupling_list, repeat=repeats_total)
            outlier_pairs = self.get_outlier(mat_dict)
            diagnose_result = self.find_ambiguous_groups(data, outlier_pairs)      
            
            group_diagnose_result.append(diagnose_result)
                    
        return group_diagnose_result
 
    def group_couplings(self, coupling_list, k_clusters=0):
        coupling_iters = np.array([gamma.reshape(-1) for gamma in coupling_list])
            
        ## grouping couplings by clustering
        if k_clusters == 0:
            ncluster_range=range(1, 10)
            yscore = inertias_elbow(coupling_iters, ncluster_range)
            # choose the best k by 2nd derivative
            max_k = np.argmax(second_grad(np.array(yscore), 1)) + 2
            clustering = KMeans(n_clusters=max_k).fit(coupling_iters)
            if self.verbose: print("N clusters found by the algorithm = {}".format(clustering.n_clusters))
            
        else: 
            clustering = KMeans(n_clusters=k_clusters).fit(coupling_iters)

        if self.verbose: print("Cluster labels = {}".format(clustering.labels_))

        clust_label_dic = {}
        for label in clustering.labels_:
            clust_label_dic[label] = np.where(clustering.labels_==label)[0]   
        
        return clust_label_dic   

    def search_best_diagonal(self, coupling_clusters, data, coupling_list):
        best_s = 0
        best_cluster = None
        for cluster, repeats in coupling_clusters.items():
            mat_dict = self.denoise_coupling(data, coupling_list, repeat=repeats)
            coupling_mat = mat_dict['coupling_mat']
            
            s = check_diagonal_score(coupling_mat)
            if s > best_s:
                best_s = s
                best_cluster = cluster
        return best_cluster
    

def map_ambiguous_groups(data, ambiguous_labels, ambiguous_idx):
    np.seterr(under='ignore')
    N = data.shape[0]
    unambiguous_idx = np.setdiff1d(list(range(N)), ambiguous_idx)
    assert N == len(ambiguous_idx)+len(unambiguous_idx)
    
    geo_mat = geodistance(data)
    uniq_groups = np.unique(ambiguous_labels)

    ### evaluate valid group
    valid_perm= list(itertools.permutations(range(len(uniq_groups))))[1:]

    ### calculate the cost matrix
    cost_mat = cdist(np.sort(geo_mat, axis=1), np.sort(geo_mat, axis=1), 'cityblock') / geo_mat.shape[1] 
           
    assert len(uniq_groups) > 1
    for perms in valid_perm:
        # keep the diagonal for original nodes
        map_mat = np.zeros((N, N))

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
