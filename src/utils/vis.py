import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.manifold import TSNE
import umap

def plt_domain_by_label(data_mat, labels, color, title, save_url='', alpha = 0.5, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    markers = ['.', '*', '+', '1', '|']
    uniq_label = np.unique(labels)

    fig = plt.figure(figsize=(3, 3))
    for i in range(len(uniq_label)):
        mark = markers[i]
        label = uniq_label[i]
        plt.scatter(data_embed[:,0][np.where(labels ==label)], data_embed[:,1][np.where(labels ==label)], c = color, marker = mark, s=150, alpha = alpha)

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10) 
    plt.title(title, fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url)
    plt.close()

def plt_domain(data_mat, color, title, save_url='', marker = '.', alpha = 0.5, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    fig = plt.figure(figsize=(3,3))
    # for swiss_roll, do not show labels
    plt.scatter(data_embed[:,0], data_embed[:,1], c = color, marker = marker, s=150, alpha= alpha)

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10) 
    plt.title(title, fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url)
    plt.close()

def plt_domain_by_const_label(data_mat, labels, color, title, save_url='', marker = '.', alpha = 0.5, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    fig = plt.figure(figsize=(3, 3))
    # for constant labels like circle
    plt.scatter(data_embed[:,0], data_embed[:,1], c = labels, marker = marker, s=150, alpha= alpha, cmap = plt.cm.get_cmap(color))

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10) 
    plt.title(title, fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url)
    plt.close()

def plt_domain_by_biolabels(data_mat, label, color, title, y_tick_labels, save_url='', a1 = 0.5, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    uniq_label = np.unique(label)
    colormap = plt.get_cmap('rainbow', len(uniq_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(uniq_label)+1.5), colormap.N) 
    
    #Plot aligned domains, samples colored by domain identity:
    fig = plt.figure(figsize=(7, 3))
    ax0 = plt.subplot(1,2,1)
    plt.scatter(data_embed[:,0], data_embed[:,1], c = color, s=25) 
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)

    ax1 = plt.subplot(1,2,2)
    plt.scatter(data_embed[:,0], data_embed[:,1], c = label, s=25, cmap=colormap, norm=norm) 
    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))   
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    cbaxes = inset_axes(ax1, width="3%", height="100%", loc='right') 
    cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
    cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=y_tick_labels)

    plt.suptitle(title, fontdict={'size': 15})
    plt.subplots_adjust(wspace=0.5)
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_mapping_by_label(X_new, y_new, label1, label2, save_url='', a1=0.3, a2=0.8, show=False):
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    markers = ['.', '*', '+', '1', '|']
    all_label = np.unique(np.concatenate((label1, label2), axis = 0))

    fig = plt.figure(figsize=(3, 3))
    for i in range(len(all_label)):
        mark = markers[i]
        label = all_label[i]
        plt.scatter(y_proj[:,0][np.where(label2 ==label)], y_proj[:,1][np.where(label2 ==label)], 
                    c = "#FF8C00", marker = mark, s=150, alpha = a2, label="domain2")
    for i in range(len(all_label)):
        mark = markers[i]
        label = all_label[i]    
        plt.scatter(X_proj[:,0][np.where(label1 ==label)], X_proj[:,1][np.where(label1 ==label)], 
                    c = "#009ACD", marker = mark, s=100, alpha= a1, label="domain1")
        
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Domains Aligned by SCOT", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_mapping(X_new, y_new, save_url='', a1=0.3, a2=0.8, show=False):
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(y_proj[:,0], y_proj[:,1], c = "#FF8C00", marker = '*', s=150, alpha = a2, label="domain2")
    plt.scatter(X_proj[:,0], X_proj[:,1], c = "#009ACD", marker = '.', s=100, alpha= a1, label="domain1")  
        
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Domains Aligned by SCOT", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_mapping_by_const_label(X_new, y_new, label1, label2, save_url='', a1=0.3, a2=0.8, show=False):
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(y_proj[:,0], y_proj[:,1], c = label2, marker = '*', s=150, alpha = a2, label="domain2", cmap = plt.cm.get_cmap('Oranges') )
    plt.scatter(X_proj[:,0], X_proj[:,1], c = label1, marker = '.', s=100, alpha= a1, label="domain1", cmap = plt.cm.get_cmap('Blues') )
        
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Domains Aligned by SCOT", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_mapping_by_biolabels(X_new, y_new, label1, label2, title1, title2, y_tick_labels, save_url='', XontoY=True, show=False):
    
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    all_label = np.unique(np.concatenate((label1, label2), axis = 0))
    colormap = plt.get_cmap('rainbow', len(all_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(all_label)+1.5), colormap.N) 

    #Plot aligned domains, samples colored by domain identity:
    fig = plt.figure(figsize=(7, 3))
    ax0 = plt.subplot(1,2,1)
    if XontoY:
        plt.scatter(y_proj[:,0], y_proj[:,1], c="#FF8C00", s=25, label = title2)    
        plt.scatter(X_proj[:,0], X_proj[:,1], c="#009ACD", s=25, label = title1)
    else:
        plt.scatter(X_proj[:,0], X_proj[:,1], c="#009ACD", s=25, label = title1)        
        plt.scatter(y_proj[:,0], y_proj[:,1], c="#FF8C00", s=25, label = title2)        
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)

    ax1 = plt.subplot(1,2,2)
    colormap = plt.get_cmap('rainbow', len(all_label))   
    if XontoY:
        plt.scatter(y_proj[:,0], y_proj[:,1], c=label2, s=25, cmap=colormap, label = title2)    
        plt.scatter(X_proj[:,0], X_proj[:,1], c=label1, s=25, cmap=colormap, label = title1)
    else:
        plt.scatter(X_proj[:,0], X_proj[:,1], c=label1, s=25, cmap=colormap, label = title1)        
        plt.scatter(y_proj[:,0], y_proj[:,1], c=label2, s=25, cmap=colormap, label = title2) 

    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))      
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    cbaxes = inset_axes(ax1, width="3%", height="100%", loc='right') 
    cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
    cbar.set_ticks(ticks=(np.arange(0,len(all_label))+1), labels=y_tick_labels)

    plt.suptitle('Domains aligned by SCOT', fontdict={'size': 15}) 
    plt.subplots_adjust(wspace=0.5)
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_cannotlink_by_label(data_mat, ambiguous_nodes, labels, ambiguous_links, save_url='', cl_alpha = 0.1, alpha=0.8, show=False, link_style = False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    markers = ['.', '*', '+', '1', '|']
    uniq_label = np.unique(labels)

    fig = plt.figure(figsize=(3, 3))
    for i in range(len(uniq_label)):
        mark = markers[i]
        label = uniq_label[i]
        plt.scatter(data_embed[:,0][np.where(labels ==label)], data_embed[:,1][np.where(labels ==label)], c = "#009ACD", marker = mark, s=150, alpha = alpha, zorder=10)
    
    # plt ambiguous_links
    if len(ambiguous_nodes) > 0:
        data_pca_ambiguous = data_embed[ambiguous_nodes, :]
        ambiguous_links = np.transpose(np.array(ambiguous_links))
        if link_style:
            for i in range(len(ambiguous_links[0])):
                rad = 0.4 if data_pca_ambiguous[ambiguous_links[0][i], 1] > 0 else -0.4
                plt.annotate("",
                        xy=([data_pca_ambiguous[ambiguous_links[0][i], 0], data_pca_ambiguous[ambiguous_links[0][i], 1]]),
                        xytext=(data_pca_ambiguous[ambiguous_links[1][i], 0], data_pca_ambiguous[ambiguous_links[1][i], 1]),
                        size=20, va="center", ha="center",
                        arrowprops=dict(color='black',
                                        arrowstyle="-",
                                        connectionstyle="arc3, rad={}".format(rad),
                                        linewidth=0.2,
                                        alpha = 0.1
                                        )
                        )
        else:
            plt.plot([data_pca_ambiguous[ambiguous_links[0], 0], data_pca_ambiguous[ambiguous_links[1], 0]], 
                    [data_pca_ambiguous[ambiguous_links[0], 1], data_pca_ambiguous[ambiguous_links[1], 1]], 
                    c="black", alpha = cl_alpha, linewidth = 0.2)
    
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("cell-cell ambiguities", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_cannotlink_by_const_label(data_mat, ambiguous_nodes, labels, cannot_links, save_url='', cl_alpha = 0.1, alpha=0.8, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(data_embed[:,0], data_embed[:,1], c = labels, marker = '.', s=150, alpha= alpha, cmap = plt.cm.get_cmap('Blues'), zorder=10)

    # plt ambiguous_links
    if len(ambiguous_nodes) > 0:
        data_pca_ambiguous = data_embed[ambiguous_nodes, :]
        cannot_links = np.transpose(np.array(cannot_links))
        plt.plot([data_pca_ambiguous[cannot_links[0], 0], data_pca_ambiguous[cannot_links[1], 0]], 
                [data_pca_ambiguous[cannot_links[0], 1], data_pca_ambiguous[cannot_links[1], 1]], 
                c="grey", alpha = cl_alpha, linewidth = 0.2)

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("cell-cell ambiguities", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_cannotlink_by_biolabels(data_mat, ambiguous_nodes, labels, cannot_links, y_tick_labels, save_url='', cl_alpha = 0.1, show=False):
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    uniq_label = np.unique(labels)
    colormap = plt.get_cmap('rainbow', len(uniq_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(uniq_label)+1.5), colormap.N) 

    fig = plt.figure(figsize=(3, 3))
    ax0 = plt.subplot(1,1,1)
    plt.scatter(data_embed[:,0], data_embed[:,1], c = labels, label="Node", s=25, cmap=colormap, norm=norm, zorder=10)

    # plt ambiguous_links
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    cannot_links = np.transpose(np.array(cannot_links))
    plt.plot([data_pca_ambiguous[cannot_links[0], 0], data_pca_ambiguous[cannot_links[1], 0]], 
            [data_pca_ambiguous[cannot_links[0], 1], data_pca_ambiguous[cannot_links[1], 1]], 
            c="grey", alpha = cl_alpha, linewidth = 0.2)
    
    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))   
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("cell-cell ambiguities", fontdict={'fontsize': 15})
    cbaxes = inset_axes(ax0, width="3%", height="100%", loc='right') 
    cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
    cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=y_tick_labels)   
    if show:
        plt.show() 
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_ambiguous_groups_by_label(data_mat, ambiguous_nodes, cluster_labels, cell_labels, save_url='', alpha=0.5, show=False):
    assert len(ambiguous_nodes) == len(cluster_labels)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambiguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    fig = plt.figure(figsize=(3, 3))
    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f', '#bcbd22']
    markers = ['.', '*', '+', '1', '|']
    uniq_cell_label = np.sort(np.unique(cell_labels))

    cell_label_marker = np.empty(cell_labels.shape, dtype='str')
    for i in range(len(uniq_cell_label)):
        cell_label_marker[np.where(cell_labels==uniq_cell_label[i])] = markers[i]

    mscatter(list(data_embed[unambiguous_nodes, 0]), list(data_embed[unambiguous_nodes, 1]), 
                m = list(cell_label_marker[unambiguous_nodes]), c="grey", alpha=alpha, s=150,label='Unambiguous')
    for idx, class_label in enumerate(np.sort(np.unique(cluster_labels))):
        class_indices = np.where(cluster_labels == class_label)[0]
        print('class_label={}\tclass_indices={}'.format(class_label, class_indices))
        mscatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), 
                    m = list(cell_label_marker[ambiguous_nodes][class_indices]), c=color_lst[idx],
                    alpha=alpha, label='Ambiguous Class={}'.format(class_label), s=150)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous groups", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_ambiguous_groups_by_const_label(data_mat, ambiguous_nodes, cluster_labels, save_url='', alpha=0.5, show=False):
    assert len(ambiguous_nodes) == len(cluster_labels)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambiguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(list(data_embed[unambiguous_nodes, 0]), list(data_embed[unambiguous_nodes, 1]), 
                c="grey", alpha=alpha, marker = '.', label='Unambiguous', s=150)
    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f', '#bcbd22']
    for idx, class_label in enumerate(np.unique(cluster_labels)):
        class_indices = np.where(cluster_labels == class_label)[0]
        plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha, marker = '.',
                    label='ambiguous Class={}'.format(class_label), s=150, c=color_lst[idx])
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous groups", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_ambiguous_groups_by_biolabels(data_mat, ambiguous_nodes, ambiguous_labels, save_url='', alpha=0.5, show=False):
    assert len(ambiguous_nodes) == len(ambiguous_labels)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                c="grey", alpha=alpha, label='Certain', s=25)
    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f', '#bcbd22']
    for idx, class_label in enumerate(np.unique(ambiguous_labels)):
        class_indices = np.where(ambiguous_labels == class_label)[0]
        plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha,
                    label='Ambiguous Class={}'.format(class_label), s=25, c=color_lst[idx])
        
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous groups", fontdict={'fontsize': 10})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plt_k_elbow(x_step, yerror, best_k, save_url='', show=False):
    k_range = len(yerror)
    fig = plt.figure(figsize=(3, 3))
    plt.plot([i*x_step for i in range(k_range)], yerror, linewidth=4, color = "#009ACD")
    plt.scatter((best_k-1)*x_step, yerror[best_k-1], color = "red", s=150)
    plt.title('Elbow Method', fontdict={'fontsize': 15})
    plt.xlabel('k / Number of clusters', fontsize=10)
    plt.ylabel('# of uncertain pairs / all possible pairs', fontsize=10)
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight") 
    plt.close()
