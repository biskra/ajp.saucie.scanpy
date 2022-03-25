import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.api as sc
import os
import scanpy.external as sce
import louvain
# import phate
import seaborn as sns
import anndata as ad
import time
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

from matplotlib import cm, colors

from palettable.scientific.sequential import Bilbao_20
from palettable.cartocolors.qualitative import Vivid_3
from palettable.cartocolors.qualitative import Vivid_4
from palettable.cartocolors.qualitative import Vivid_10
from palettable.scientific.diverging import Vik_20
from palettable.scientific.diverging import Berlin_10
#
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

from model import SAUCIE
from loader import Loader

sc.settings.verbosity = 4  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=200,)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_versions()
sc.settings.figdir='./paper.figs/fig1/'

print(os.getcwd())
os.chdir("/home/UTHSCSA/iskra/CytOF_Datasets/10.08-10.10-Debarcoded/GG")
adata = sc.read_10x_mtx('./scRNAseq-10x-outs/iskra_cmis/', var_names='gene_symbols', cache=True)



adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

adata_raw = adata.copy()

sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True, stripplot=False,
             save='.scrnaseq.vio.pre.qc.png')

adata = adata[adata.obs['n_genes'] < 3750, :]
adata = adata[adata.obs['n_genes'] > 750, :]
adata = adata[adata.obs['n_counts'] < 12500, :]
adata = adata[adata.obs['n_counts'] > 1250, :]
adata = adata[adata.obs['percent_mito'] < 0.175, :]

sc.pp.normalize_per_cell(adata)

adata_log1p = sc.pp.log1p(adata, copy=True)
adata_log1p.raw = adata.copy()

# adata_sqrt = adata.copy()
# adata_sqrt.X = np.sqrt(adata.X.copy())
# adata_sqrt.raw = adata.copy()

sc.pp.filter_cells(adata_log1p, min_genes=750)
sc.pp.filter_genes(adata_log1p, min_cells=10)
#
# sc.pp.filter_cells(adata_sqrt, min_genes=1000)
# sc.pp.filter_genes(adata_sqrt, min_cells=3)

# sc.settings.figdir='./scanpy.preprocessing/'
# sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              save='.scrnaseq.vio.post.qc.png')
# plt.savefig('./scanpy.preprocessing/qc.pass.nmui.ncounts.pmt.png')
# plt.close()

sc.pl.scatter(adata, x='n_counts', y='percent_mito')

plt.savefig('./scanpy.preprocessing/qc.pass.ncounts.pmt.png')
plt.close()

sc.pl.scatter(adata, x='n_counts', y='n_genes')

plt.savefig('./scanpy.preprocessing/qc.pass.ncounts.ngenes.png')
plt.close()


rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6a', 'Ly6e', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1']

adata_log1p.obs_names_make_unique()
sce.pp.magic(adata_log1p, n_pca=100, n_jobs=111)

adata_log1p.obsm['MAGIC RNA Marks'] = adata_log1p[:, rna_mark].X

sc.tl.pca(adata_log1p, n_comps=100, random_state=42)
sc.pp.neighbors(adata_log1p, n_neighbors=30, random_state=42)
sc.tl.louvain(adata_log1p, random_state=42)
sc.tl.paga(adata_log1p)
sc.pl.paga(adata_log1p, show=False)
sc.tl.umap(adata_log1p, init_pos='paga', n_components=2, random_state=42)
sc.pl.umap(adata_log1p, color=['louvain', 'Pdgfra', 'Ptprc', 'Acta2'], legend_loc='on data', ncols=2, save='.louvain.feats.png')

sum(adata_log1p.obs['louvain'].isin(['0', '1', '3', '4', '6', '7', '9', '10', '11', '14']))/5130#Fibros
sum(adata_log1p.obs['louvain'].isin(['2', '5', '15', '18', '19', '20']))/5130#Leukos
sum(adata_log1p.obs['louvain'].isin(['8', '12', '13', '16', '17']))/5130#Peris

major_clust = {'0': 'Fibroblasts 61.7%',
               '1': 'Fibroblasts 61.7%',
               '3': 'Fibroblasts 61.7%',
               '4': 'Fibroblasts 61.7%',
               '6': 'Fibroblasts 61.7%',
               '7': 'Fibroblasts 61.7%',
               '9': 'Fibroblasts 61.7%',
               '10': 'Fibroblasts 61.7%',
               '11': 'Fibroblasts 61.7%',
               '14': 'Fibroblasts 61.7%',

               '2': 'Leukocytes 20.6%',
               '5': 'Leukocytes 20.6%',
               '15': 'Leukocytes 20.6%',
               '18': 'Leukocytes 20.6%',
               '19': 'Leukocytes 20.6%',
               '20': 'Leukocytes 20.6%',

               '8': 'Pericytes 16.9%',
               '12': 'Pericytes 16.9%',
               '13': 'Pericytes 16.9%',
               '16': 'Pericytes 16.9%',
               '17': 'Pericytes 16.9%',

               '21': 'Other 0.760%'
                }

adata_log1p.obs['Major Clusters'] = adata_log1p.obs['louvain'].map(major_clust)
sc.pl.umap(adata_log1p, color=['Major Clusters'], ncols=2, save='.mcs.umap.png')

sc.pp.neighbors(adata_log1p, n_neighbors=30, random_state=42, use_rep='MAGIC RNA Marks')

sc.tl.paga(adata_log1p)
sc.pl.paga(adata_log1p, show=False)
sc.tl.umap(adata_log1p, init_pos='paga', random_state=42)
sc.pl.umap(adata_log1p, color=['Major Clusters'], ncols=1, save='.mcs.magic.feats.png')
sc.pl.umap(adata_log1p, color=['louvain'], ncols=1, save='.scs.magic.feats.png')

sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='Major Clusters', standard_scale='var', save='.mcs.png')
sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='louvain', standard_scale='var', dendrogram=True, save='.scs.png')

# sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='louvain', standard_scale='var')