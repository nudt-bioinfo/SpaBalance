# load necessary packages
import os
import scanpy as sc
import matplotlib.pyplot as plt


# read data
file_fold = 'path/to/the/dataset' #please replace 'file_fold' with the your path
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# Specify data type
data_type = '10x' # '10x', SPOTS', 'Stereo-CITE-seq', and 'Spatial-epigenome-transcriptome'.

# Fix random seed0
from SpaBalance.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)

from SpaBalance.preprocess import clr_normalize_each_cell, pca

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

# ATAC
#from SpaBalance.preprocess import lsi
#adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
#if 'X_lsi' not in adata_omics2.obsm.keys():
#    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
#    lsi(adata_omics2, use_highly_variable=False, n_components=51)

#adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

from SpaBalance.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

# define model
from SpaBalance.Train_model import Train_SpaBalance
model = Train_SpaBalance(data, datatype=data_type, device="cuda:0")# or "cpu"

# train model
output = model.train()
adata = adata_omics1
adata.obsm['SpaBalance'] = output['SpaBalance'].copy()

# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(14, 5))
sc.pp.neighbors(adata, use_rep='SpaBalance', n_neighbors=30)
sc.tl.umap(adata)

# use specified color to plot the UMAP
sc.pl.umap(adata, color='SpaBalance', ax=ax_list[0], title='SpaBalance', s=20, show=False)

# use specified basis to plot the spatial embedding
sc.pl.embedding(adata, basis='spatial', color='SpaBalance', ax=ax_list[1], title='SpaBalance', s=25, show=False)

plt.tight_layout(w_pad=0.3)
# save the figure
output_path = "/your/output/path/spaBalance_results.png"
plt.savefig(output_path, dpi=300)  # save the figure as a PNG image with a resolution of 300 dpi
plt.show()
