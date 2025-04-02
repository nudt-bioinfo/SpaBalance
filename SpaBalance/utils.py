import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch_geometric.utils import to_torch_coo_tensor, add_remaining_self_loops
from torch_geometric.utils import dropout_edge
from sklearn.preprocessing import LabelEncoder

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

def split_adata_ob(ads, ad_ref, ob='obs', key='emb'):
    len_ads = [_.n_obs for _ in ads]
    if ob=='obsm':
        split_obsms = np.split(ad_ref.obsm[key], np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obsms):
            ad.obsm[key] = v
    else:
        split_obs = np.split(ad_ref.obs[key].to_list(), np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obs):
            ad.obs[key] = v
            
def edgeindex2adj(edge_index, num_nodes):
    adj_shape = (num_nodes, num_nodes)  # n*n
    edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
    adj = to_torch_coo_tensor(edge_index, size = adj_shape)
    return adj

def adj2edgeindex(adj):
    edge_index = adj.coalesce().indices().cpu().numpy()
    # Convert the numpy array to a PyTorch LongTensor
    edge_index = torch.tensor(edge_index, dtype=torch.long, device='cuda:0')
    edge_weight = torch.tensor(adj.coalesce().values().cpu().numpy(), dtype=torch.float, device='cuda:0')
    num_nodes = adj.size(0)  # or adj_omics.size(1)
    # Convert edge_index to a dense format tensor
    dense_adj, adj = edgeindex2dense(edge_index, num_nodes)
    return edge_index, edge_weight

def edgeindex2dense(edge_index, num_nodes):
    # Create a sparse matrix
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device='cuda:0')  # Edge indices as torch tensor
    # Assume unweighted edges with weight 1
    edge_attr = torch.ones(edge_index_tensor.size(1)).to('cuda:0')  # If edges have weights, replace with actual weights
    # Create a sparse tensor
    sparse_adj = torch.sparse.FloatTensor(edge_index_tensor, edge_attr, torch.Size([num_nodes, num_nodes])).to('cuda:0')
    # Convert to a dense matrix
    dense_adj = sparse_adj.to_dense()
    adj = torch.eye(num_nodes).to('cuda:0')
    return dense_adj, adj

def drop_edges(adj_omics, drop_rate, device='cuda:0'):
    # Ensure adj_omics is a sparse tensor
    adj_omics = adj_omics.coalesce()  # Coalesce the sparse tensor
    adj_omics_edge_index = adj_omics.indices()  # Convert to edge_index format
    edge_weights = adj_omics.values().cpu().numpy()
    adj_omics_edge_ind, edge_mask = dropout_edge(adj_omics_edge_index, p=drop_rate)
    # Remove the weights of the dropped edges
    filtered_edge_weights = edge_weights[edge_mask.cpu().numpy()]  # Use edge_mask to filter valid weights
    # Assume the number of nodes in the graph is num_nodes (adjust according to your actual data)
    num_nodes = adj_omics.size(0)  # or adj_omics.size(1)
    # Create a sparse tensor using torch.sparse_coo_tensor
    adj_omics_masked = torch.sparse_coo_tensor(
        indices=adj_omics_edge_ind,  # Edge indices
        values=torch.tensor(filtered_edge_weights, device='cuda:0'),  # Edge weights, placed on GPU
        size=(num_nodes, num_nodes),  # Size of the graph
        device='cuda:0'  # Device where the tensor resides
    )
    
    return adj_omics_masked

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     

class OmicsDataset(Dataset):
    def __init__(self, features_omics1, features_omics2):
        self.features_omics1 = features_omics1
        self.features_omics2 = features_omics2

    def __len__(self):
        return len(self.features_omics1)

    def __getitem__(self, idx):
        return self.features_omics1[idx], self.features_omics2[idx]
    
def StrLabel2Idx(string_labels):
    # 创建LabelEncoder对象
    label_encoder = LabelEncoder()
    idx_labels = label_encoder.fit_transform(string_labels)
    
    return np.array(idx_labels)