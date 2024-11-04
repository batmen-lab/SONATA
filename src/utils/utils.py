import numpy as np
import scipy.sparse as sp 
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from pygam import LinearGAM, s
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
import scipy

def load_data(matrix_file):
    file_type = matrix_file.split('.')[-1]
    if file_type == 'txt':
        data = np.loadtxt(matrix_file)
    elif file_type == 'csv':
        data = np.loadtxt(matrix_file, delimiter=',')
    else:
        data = np.load(matrix_file) 

    return data

def sorted_by_label(data, label):
    sorted_label = []
    sorted_data = []
    sorted_indices = []

    print("Labels in order{}".format(np.sort(np.unique(label))))
    print("Label distribution: {}".format(Counter(label)))

    for l in np.sort(np.unique(label)):
        idx = np.where(label == l)[0]
        sorted_indices.append(idx)
        sorted_label.append(label[idx])
        sorted_data.append(data[idx])

    sorted_label = np.concatenate(sorted_label)
    sorted_data = np.concatenate(sorted_data)
    sorted_indices = np.concatenate(sorted_indices)

    return sorted_data, sorted_label, sorted_indices


def h_clustering(data_mat, n_cluster):
    clustering = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean').fit(data_mat)
    return clustering.labels_

def get2hop(data, mode, metric, k):
    include_self=True if mode=="connectivity" else False

    two_hop = {}
    graph = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=include_self)
    for i in range(graph.shape[0]):
        ineighs = np.where((graph[i].toarray() > 0) & (np.arange(graph.shape[0]) != i))[1]
        i2hop = {}
        for j in ineighs:
            jneighs = np.where((graph[j].toarray() > 0) & (np.arange(graph.shape[0]) != j) & (np.arange(graph.shape[0]) != i))[1]
            for v in jneighs:
                if v not in i2hop:
                    i2hop[v] = (graph[i, j] + graph[j, v]) / 2

        two_hop[i] = i2hop

    return graph, two_hop

def add_neighbors(graph, two_hop, n_neighbor=2):
    dila_links = []
    for k in two_hop.keys():
        n=0
        candidates_dict = two_hop[k]
        i2hop_nodes = list(candidates_dict.keys())
        np.random.shuffle(i2hop_nodes)
        for v in i2hop_nodes:
            if graph[k, v] == 0:
                dila_links.append((k, v))
                graph[k, v] = candidates_dict[v]
                n += 1
            if n >= n_neighbor: break

    dila_links = np.asarray(dila_links)
    return graph, dila_links

def init_distances(Xgraph):
    # Compute shortest distances
    X_shortestPath=dijkstra(csgraph= csr_matrix(Xgraph), directed=False,  unweighted=True, return_predecessors=False)

    # Deal with unconnected stuff (infinities):
    X_max=np.nanmax(X_shortestPath[X_shortestPath != np.inf])
    X_shortestPath[X_shortestPath > X_max] = X_max

    # Finnally, normalize the distance matrix:
    Cx=X_shortestPath/X_shortestPath.max()

    return Cx


def coupling2clustcoupling(coupling, clust_label, ordered_clust_label):
    clust_coupling = np.empty((len(ordered_clust_label), len(ordered_clust_label),))
    for idx, i in enumerate(ordered_clust_label):
        for jdx, j in enumerate(ordered_clust_label):
            clust_coupling[idx, jdx] = np.mean(coupling[clust_label == i, :][:, clust_label == j])
    return clust_coupling

def clustcoupling2coupling(clust_coupling, clust_label, ordered_clust_label):
    new_coupling = np.empty((len(clust_label), len(clust_label)))
    for idx, i in enumerate(ordered_clust_label):
        for jdx, j in enumerate(ordered_clust_label):
            for m in np.where(clust_label == i)[0]:
                for n in np.where(clust_label == j)[0]:
                    new_coupling[m, n] = clust_coupling[idx, jdx]
    return new_coupling

def fit_spline(X, Y, grid=0.1):
    '''
    Use the P% lowest scatters to fit a spline (Linear GAM)
    '''
    new_X, new_Y, map_idx = np.array([]), np.array([]), np.array([], dtype=np.int8)
    percents = [1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, 0.1] 

    n=-1
    for i in range(10, 100, int(grid * 100)):
        n += 1
        i = i / 100
        grid_idx = np.where((X < i))
        if len(grid_idx[0]) == 0: continue

        percent = percents[n]
        percent_val = np.percentile(Y[grid_idx], percent * 100)
        percent_idx = np.where(Y[grid_idx] <= percent_val)
        new_X = np.concatenate([new_X, X[grid_idx][percent_idx]])
        new_Y = np.concatenate([new_Y, Y[grid_idx][percent_idx]])
        map_idx = np.concatenate([map_idx, grid_idx[0][percent_idx]])

    # using the new X, new_Y to fit a spline
    gam = LinearGAM(s(0, constraints='monotonic_dec')).fit(new_X, new_Y)
    
    pred_Y = gam.predict(X) 
    res = gam.deviance_residuals(X, Y)

    return pred_Y, res, map_idx


def p_value(res, currX, lowest_idx, geo_thres, epsilon=1e-2):
    # prev-k nodes as neighbors, one-tailed z-test
    thres_idx = len(np.where(currX<geo_thres)[0])
    # do not consider self to self pairs
    p_values = []
    for _ in range(thres_idx): p_values.append(1) 
    
    for idx in range(len(currX))[thres_idx:]:
        neighbor_indices = np.arange(idx+1)[thres_idx:]
        new_neighor_indices = np.intersect1d(neighbor_indices, lowest_idx)

        local_std = np.std(res[new_neighor_indices])
        # avoid all zeros -> std = 0. Add epsilon std to the local distribution
        if local_std <= epsilon: local_std = epsilon

        # right tail
        zval = (res[idx]) / local_std
        p_values.append(scipy.stats.norm.cdf(-zval))

    p_values = np.asarray(p_values)
    return p_values

def add_missingness(clust_mats, add_lists, n_bin=30, fit_type="nondec_spline"):
    clust_dist_mat, clust_coupling_mat = clust_mats
    # bin-wise clust_cx_mat
    bins = np.linspace(0, 1, n_bin)

    bin_clust_dist_mat = bins[np.digitize(clust_dist_mat.reshape(-1), bins)]
    bin_clust_dist_mat_flat = bin_clust_dist_mat.reshape(-1)
    clust_coupling_mat_flat = clust_coupling_mat.reshape(-1)
    
    x_arr = np.unique(bin_clust_dist_mat_flat)
    y_arr = []
    for score in x_arr:
        idx = np.where(bin_clust_dist_mat_flat == score)[0]
        zeor_idx = np.where(clust_coupling_mat_flat[idx] == 0)[0]
        y_arr.append(len(zeor_idx))
    y_arr = np.array(y_arr)

    if fit_type == "linearGAM":
        gam = LinearGAM(s(0, constraints='monotonic_inc')).fit(x_arr, y_arr)
        pred_Y = gam.predict(x_arr)
    elif fit_type == "nondec_spline":
        spline = UnivariateSpline(x_arr, y_arr, ) 
        pred_Y1 = spline(x_arr)
        iso_reg = IsotonicRegression(increasing=True).fit(x_arr, pred_Y1)
        pred_Y = iso_reg.predict(x_arr)   
    elif fit_type == "max_nondec_spline":
        nondec_y_arr = [y_arr[0]]
        for i in range(1, len(y_arr)):
            if y_arr[i] < nondec_y_arr[-1]:
                nondec_y_arr.append(nondec_y_arr[-1])
            else:
                nondec_y_arr.append(y_arr[i])
        spline = UnivariateSpline(x_arr, nondec_y_arr, ) 
        pred_Y1 = spline(x_arr)
        iso_reg = IsotonicRegression(increasing=True).fit(x_arr, pred_Y1)
        pred_Y = iso_reg.predict(x_arr)
    else:
        raise ValueError('Fit_type must be in ["linearGAM", "nondec_spline", "max_nondec_spline"], instead of {}.'.format(fit_type))
    
    # add missingness
    avg_dist_list = add_lists["avg_dist_list"]
    avg_coupling_list = add_lists["avg_coupling_list"]
    clust_pair_list = add_lists["clust_pair_list"]
    diff_y = pred_Y - y_arr
    for idx in range(len(x_arr)): 
        nums_zero = max(0, int(diff_y[idx]))
        avg_dist_list += [x_arr[idx]] * nums_zero
        # keep the same length as the avg_cx_list
        avg_coupling_list += [0] * nums_zero
        clust_pair_list += [(-1, -2)] * nums_zero # (-1, -2) is a dummy cluster pair
        
    return (avg_dist_list, avg_coupling_list, clust_pair_list)


def elbow_k(data, cannot_link, k_range = 10):
    from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
    
    def _second_grad(k_arr, step):
        # first grad
        first_grad = (k_arr[1:] - k_arr[:-1])/step 
        # 2nd grad
        second_grad = (first_grad[1:] - first_grad[:-1])/(1+first_grad[1:]*first_grad[:-1])
        second_grad = np.arctan(np.abs(second_grad))
        return second_grad
    
    cl_arr = np.transpose(np.array(cannot_link))
    print("cannot_link shape={}\ndata_shape = {}".format(len(cannot_link), data.shape))

    # calculate # of ambiguity pairs for each k
    y_error = []
    for k in range(1, k_range):
        print("k={}".format(k))
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
    x_step = 1 / data.shape[0]

    # choose the best k by 2nd derivative
    best_k = np.argmax(_second_grad(y_error, step = x_step)) + 2

    
    return best_k, y_error, x_step


def geodistance(data, kmin = 10, kmax = 200, dist_mode = 'distance', metric = 'euclidean'):
    """
    kmax/kmin: param for knn
    dist_mode: 'connectivity' for 0/1; 'distance' for distance measured by metric (euclidean)
    metric: 'euclidean'/'correlation'
    """
    nbrs = NearestNeighbors(n_neighbors=kmin, metric=metric, n_jobs=-1).fit(data)
    knn = nbrs.kneighbors_graph(data, mode = dist_mode)

    connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
    while connected_components != 1:
        if kmin > np.max((kmax, 0.01*len(data))):
            break
        kmin += 2
        nbrs = NearestNeighbors(n_neighbors=kmin, metric=metric, n_jobs=-1).fit(data)
        knn = nbrs.kneighbors_graph(data, mode = dist_mode)
        connected_components = sp.csgraph.connected_components(knn, directed=False)[0]

    # calculate the shortest distance between node pairs
    dist = sp.csgraph.floyd_warshall(knn, directed=False)
    
    dist_max = np.nanmax(dist[dist != np.inf])
    dist[dist > dist_max] = 2*dist_max
    
    # # global max-min normalization
    norm_geo_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))     
    # norm_geo_dist = dist 

    return norm_geo_dist


from .metrics import transfer_accuracy
def sonata_best_acc(x_aligned, y_aligned, label1, label2, alter_mappings, mapping, modality=1):
    acc_best = 0
    x_aligned_best = None
    y_aligned_best = None
    best_mapping = None
    for idx, m in enumerate(alter_mappings):
        if modality == 1:
            this_mapping = np.matmul(m, mapping)
            x_aligned_new = np.matmul(m, x_aligned)
            y_aligned_new = y_aligned
        else:
            this_mapping = np.matmul(mapping, m.T)
            x_aligned_new = x_aligned
            y_aligned_new = np.matmul(m, y_aligned)
        
        acc = transfer_accuracy(x_aligned_new, y_aligned_new, label1, label2)
        if acc > acc_best:
            x_aligned_best = x_aligned_new
            y_aligned_best = y_aligned_new
            best_mapping = this_mapping
            acc_best = acc 
        
    return x_aligned_best, y_aligned_best, best_mapping, acc_best

def silhouette_score_elbow(coupling_iters, ncluster_range=range(2, 10), metric='euclidean'):
    sil = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in ncluster_range:
        clustering = KMeans(n_clusters=k).fit(coupling_iters)
        labels = clustering.labels_
        sil.append(silhouette_score(coupling_iters, labels, metric=metric))
    return sil

def check_diagonal_score(coupling_mat):
    N, N = coupling_mat.shape
    cnt = 0
    for i in range(N):
        cnt += 1 if coupling_mat[i, i] > 0 else 0
    return cnt