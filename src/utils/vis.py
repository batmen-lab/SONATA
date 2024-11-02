import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plt_heatmap(data, title, show=False, save_url=""):
    plt.figure()
    plt.imshow(data, cmap="Blues")
    plt.title(title)
    plt.colorbar(shrink=0.8)
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url)
        plt.close()
        
def plt_domain_by_labels(data_mat, label, title, marker='.', save_url = '', a = 1.0, show = False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    uniq_label = np.unique(label)
    colormap = plt.get_cmap('rainbow', len(uniq_label))
    
    plt.figure(figsize=(5, 5))
    # domain scatters colored by label
    if len(uniq_label) > 10:
        print("Too many labels, use gradient color instead.")
        plt.scatter(data_embed[:,0], data_embed[:,1], c = label, s=50, marker = marker, alpha= a, cmap = plt.cm.get_cmap('Blues'))
    else:
        for idx, l in enumerate(uniq_label):
            plt.scatter(data_embed[np.where(label == l), 0], data_embed[np.where(label == l), 1], 
                        color = colormap(idx), s=50, marker = marker, alpha= a, label = "Type{}".format(l))
            
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title(title, fontdict={'size': 15})  
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 15}) 
    
    if show:
        plt.show()
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
        plt.close() 



def plt_ambiguous_groups_by_labelcolor(data_mat, ambiguous_nodes, ambiguous_labels, marker='.', save_url='', alpha=0.5, show=False, vs_mode='pca'):
    assert len(ambiguous_nodes) == len(ambiguous_labels)
    assert (vs_mode in ["pca","umap", "tsne"]), "visual mode argument has to be either one of 'pca', 'umap' or 'tsne'."
    if vs_mode == 'pca':
        pca = PCA(n_components=2).fit(data_mat)
        data_embed = pca.fit_transform(data_mat)
        x_label = "pca1"
        y_label = "pca2"
    elif vs_mode == 'umap':
        n_neigh = np.ceil(np.sqrt(data_mat.shape[0])).astype(int)
        n_neigh = 20
        data_embed = umap.UMAP(n_components=2, n_neighbors = n_neigh, min_dist = 0.7).fit_transform(data_mat)
        x_label = "umap1"
        y_label = "umap2"
    elif vs_mode == 'tsne':
        data_embed = TSNE(n_components=2).fit_transform(data_mat)
        x_label = "tsne1"
        y_label = "tsne2"

    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    plt.figure(figsize=(5, 5))
    plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                c="grey", alpha=alpha, label='Certain', s=50, marker=marker)
    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f', '#bcbd22']
    for idx, class_label in enumerate(np.unique(ambiguous_labels)):
        class_indices = np.where(ambiguous_labels == class_label)[0]
        plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha,
                    label='Ambiguous Class={}'.format(class_label), s=50, c=color_lst[idx], marker=marker)
        
    
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous groups", fontdict={'fontsize': 10})
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_cannotlink_by_labelcolor(data_mat, ambiguous_nodes, labels, cannot_links, marker='.', save_url='', alpha=0.8, cl_alpha = 0.1, show=False, curve_link=False, label_type='ambiguous', vs_mode='pca'):
    assert (vs_mode in ["pca","umap", "tsne"]), "visual mode argument has to be either one of 'pca', 'umap' or 'tsne'."
    if vs_mode == 'pca':
        pca = PCA(n_components=2).fit(data_mat)
        data_embed = pca.fit_transform(data_mat)
        x_label = "pca1"
        y_label = "pca2"
    elif vs_mode == 'umap':
        n_neigh = np.ceil(np.sqrt(data_mat.shape[0])).astype(int)
        n_neigh = 20
        data_embed = umap.UMAP(n_components=2, n_neighbors = n_neigh, min_dist = 0.7).fit_transform(data_mat)
        x_label = "umap1"
        y_label = "umap2"
    elif vs_mode == 'tsne':
        data_embed = TSNE(n_components=2).fit_transform(data_mat)
        x_label = "tsne1"
        y_label = "tsne2"

    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                       '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
                       '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', 
                       '#FF33D7', '#000000', '#6FFF33', '#FF7F50', '#9FE2BF', '#DFFF00', '#6495ED', '#FFBF00', '#DE3163', '#CCCCFF', 
                       '#40E0D0', '#8A2BE2', '#FF4500', '#2E8B57', '#FF6347', '#4682B4', '#DAA520'
                       ]

    uniq_label = np.unique(labels)
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    # colormap = plt.get_cmap('rainbow', len(uniq_label))
    
    plt.figure(figsize=(5, 5))
    if label_type == 'ambiguous':
        plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                    c="grey", alpha=alpha, label='Certain', s=50, marker=marker)
        for idx, class_label in enumerate(np.unique(labels)):
            class_indices = np.where(labels == class_label)[0]
            plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha,
                        label='Ambiguous Class={}'.format(class_label), s=50, c=color_lst[idx], marker=marker, zorder=10)
    elif label_type == 'cluster':
        for i, label_id in enumerate(uniq_label):
            data_subset = data_embed[labels == label_id]
            plt.scatter(list(data_subset[:, 0]), list(data_subset[:, 1]), s=50, alpha = alpha, marker=marker,
                        color = color_lst[i], label='label{}'.format(label_id), zorder=10)
    else:
        raise ValueError("label type should be either 'ambiguous' or 'cluster'.")
    
    # plt ambiguous_links
    if len(cannot_links) > 0:
        data_pca_ambiguous = data_embed[ambiguous_nodes, :]
        cannot_links = np.transpose(np.array(cannot_links))
        
        if curve_link:
            ## for t_branch only
            for i in range(len(cannot_links[0])):
                rad = 0.4 if data_pca_ambiguous[cannot_links[0][i], 1] > 0 else -0.4
                plt.annotate("",
                        xy=([data_pca_ambiguous[cannot_links[0][i], 0], data_pca_ambiguous[cannot_links[0][i], 1]]),
                        xytext=(data_pca_ambiguous[cannot_links[1][i], 0], data_pca_ambiguous[cannot_links[1][i], 1]),
                        size=20, va="center", ha="center",
                        arrowprops=dict(color='black',
                                        arrowstyle="-",
                                        connectionstyle="arc3, rad={}".format(rad),#  arc3,rad=0.4
                                        linewidth=0.2,
                                        alpha = 0.05
                                        )
                        )
        else:
            plt.plot([data_pca_ambiguous[cannot_links[0], 0], data_pca_ambiguous[cannot_links[1], 0]], 
                    [data_pca_ambiguous[cannot_links[0], 1], data_pca_ambiguous[cannot_links[1], 1]], 
                    c="grey", alpha = cl_alpha, linewidth = 0.2)      
                
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous links", fontdict={'fontsize': 15})
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    if show:
        plt.show() 
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()



def plt_mapping_by_labels(X_new: np.ndarray, y_new: np.ndarray, 
                          label1: np.ndarray, label2: np.ndarray, 
                          title: str = None, save_url: str = '', 
                         a: float = 0.8, show: bool = False) -> None:
    title1="Modality 1"
    title2="Modality 2"
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    uniq_label = np.unique(np.concatenate((label1, label2), axis = 0))
    colormap = plt.get_cmap('rainbow', len(uniq_label))

    plt.figure(figsize=(5, 5))
    if len(uniq_label) > 10:
        print("Too many labels, use gradient color instead.")
        plt.scatter(y_proj[:,0], y_proj[:,1], c = label2, s=50, marker='*', label = title1, alpha = a, cmap = plt.cm.get_cmap('Blues'))
        plt.scatter(X_proj[:,0], X_proj[:,1], c = label1, s=30, label = title2, alpha = a, cmap = plt.cm.get_cmap('Blues'))
    else: 
        for i in range(len(uniq_label)):
            label = uniq_label[i]
            plt.scatter(y_proj[:,0][np.where(label2 ==label)], y_proj[:,1][np.where(label2 ==label)], 
                        color = colormap(i), marker = '*', s=50, alpha = a, label="domain2")
        # domain1
        for i in range(len(uniq_label)):
            label = uniq_label[i]    
            plt.scatter(X_proj[:,0][np.where(label1 ==label)], X_proj[:,1][np.where(label1 ==label)], 
                        color = colormap(i), marker = '.', s=30, alpha= a, label="domain1")

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title(title, fontdict={'size': 15})   
    plt.subplots_adjust(wspace=0.5)
    
    if show:
        plt.show()
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
        plt.close()
