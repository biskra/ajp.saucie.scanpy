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
sc.settings.figdir='./paper.figs/fig2/'

print(os.getcwd())
os.chdir("/home/UTHSCSA/iskra/CytOF_Datasets/10.08-10.10-Debarcoded/GG")


adata1 = sc.read_10x_mtx('./scRNAseq-10x-outs/iskra_cmis/', var_names='gene_symbols', cache=True)

adata2 = sc.read_10x_mtx('./scRNAseq-10x-outs/skelly_csc1/', var_names='gene_symbols', cache=True)

adata3 = sc.read_10x_mtx('./scRNAseq-10x-outs/skelly_csc2/', var_names='gene_symbols', cache=True)

adata4 = sc.read_10x_mtx('./scRNAseq-10x-outs/sham-day7-tip/', var_names='gene_symbols', cache=True)

adata5 = sc.read_10x_mtx('./scRNAseq-10x-outs/sham-day7-gfp', var_names='gene_symbols', cache=True)

adata6 = sc.read_10x_mtx('./scRNAseq-10x-outs/mi-day3-tip', var_names='gene_symbols', cache=True)

adata7 = sc.read_10x_mtx('./scRNAseq-10x-outs/mi-day7-tip', var_names='gene_symbols', cache=True)

adata8 = sc.read_10x_mtx('./scRNAseq-10x-outs/mi-day7-gfp', var_names='gene_symbols', cache=True)

adata = ad.AnnData.concatenate(adata1, adata2, adata3, adata4, adata5, adata6, adata7, adata8)



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

scrnaseq_sample = adata_log1p.obs['batch'].values.astype('U32')

scrnaseq_sample[scrnaseq_sample == '0'] = 'Iskra'
scrnaseq_sample[scrnaseq_sample == '1'] = 'Skelly1'
scrnaseq_sample[scrnaseq_sample == '2'] = 'Skelly2'
scrnaseq_sample[scrnaseq_sample == '3'] = 'FarbehiShamTIPDay7'
scrnaseq_sample[scrnaseq_sample == '4'] = 'FarbehiShamGFPDay7'
scrnaseq_sample[scrnaseq_sample == '5'] = 'FarbehiMITIPDay3'
scrnaseq_sample[scrnaseq_sample == '6'] = 'FarbehiMITIPDay7'
scrnaseq_sample[scrnaseq_sample == '7'] = 'FarbehiMIGFPDay7'
adata_log1p.obs['Samples'] = scrnaseq_sample

# adata_sqrt = adata.copy()
# adata_sqrt.X = np.sqrt(adata.X.copy())
# adata_sqrt.raw = adata.copy()

sc.pp.filter_cells(adata_log1p, min_genes=750)
sc.pp.filter_genes(adata_log1p, min_cells=10)
#
# sc.pp.filter_cells(adata_sqrt, min_genes=1000)
# sc.pp.filter_genes(adata_sqrt, min_cells=3)

# sc.settings.figdir='./scanpy.preprocessing/'
sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True, stripplot=False,
             save='.scrnaseq.vio.post.qc.png')
# plt.savefig('./scanpy.preprocessing/qc.pass.nmui.ncounts.pmt.png')
plt.close()

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
sc.pp.neighbors(adata_log1p, use_rep='MAGIC RNA Marks')

sc.tl.louvain(adata_log1p, resolution=0.5, random_state=42)
sc.tl.paga(adata_log1p, groups='louvain')
sc.pl.paga(adata_log1p)

sc.tl.umap(adata_log1p, random_state=42, init_pos='paga')

sc.pl.umap(adata_log1p, color='louvain', legend_loc='on data', save='.louvain.png')
sc.pl.umap(adata_log1p, color='Samples', save='.sample.png')
sc.pl.umap(adata_log1p, color=['Pecam1', 'Pdgfra', 'Ptprc', 'Acta2'], legend_loc='on data', ncols=2, save='.feats.png')
# sc.pl.umap(adata_log1p, color=['Mki67', 'Ccnb1', 'Birc5', 'mt-Co1'], legend_loc='on data', ncols=2, save='.cc.mt.feats.png')

sum(adata_log1p.obs['louvain'].isin(['1', '18', '20', '25']))/42843 #Endo
sum(adata_log1p.obs['louvain'].isin(['0', '3', '6', '14', '22', '28', '29', '4', '5', '8', '9', '10']))/42843 #Homeostatic Fibro
sum(adata_log1p.obs['louvain'].isin(['13', '17', '32']))/42843 #Pericyte
sum(adata_log1p.obs['louvain'].isin(['12', '15', '16', '19', '26']))/42843 #Homeostatic Leuko
# sum(adata_log1p.obs['louvain'].isin([]))/42843 #Injury Response Fibro
sum(adata_log1p.obs['louvain'].isin(['2', '27']))/42843 #Injury Response Fibroblasts
sum(adata_log1p.obs['louvain'].isin(['7', '11', '21', '23', '24', '30', '31']))/42843 #Injury Response Leukocytes


major_clust = {'1': 'Endothelium 12.4%',
               '18': 'Endothelium 12.4%',
               '20': 'Endothelium 12.4%',
               '25': 'Endothelium 12.4%',

               '0': 'Homeostatic Fibroblasts 51.2%',
               '3': 'Homeostatic Fibroblasts 51.2%',
               '4': 'Homeostatic Fibroblasts 51.2%',
               '5': 'Homeostatic Fibroblasts 51.2%',
               '6': 'Homeostatic Fibroblasts 51.2%',
               '8': 'Homeostatic Fibroblasts 51.2%',
               '9': 'Homeostatic Fibroblasts 51.2%',
               '10': 'Homeostatic Fibroblasts 51.2%',
               '14': 'Homeostatic Fibroblasts 51.2%',
               '22': 'Homeostatic Fibroblasts 51.2%',
               '28': 'Homeostatic Fibroblasts 51.2%',
               '29': 'Homeostatic Fibroblasts 51.2%',

               '13': 'Pericytes 5.20%',
               '17': 'Pericytes 5.20%',
               '32': 'Pericytes 5.20%',

               '12': 'Homeostatic Leukocytes 10.8%',
               '15': 'Homeostatic Leukocytes 10.8%',
               '16': 'Homeostatic Leukocytes 10.8%',
               '19': 'Homeostatic Leukocytes 10.8%',
               '26': 'Homeostatic Leukocytes 10.8%',

               '2': 'Injury Fibroblasts 8.11%',
               '27': 'Injury Fibroblasts 8.11%',

               '7': 'Injury Leukocytes 12.3%',
               '11': 'Injury Leukocytes 12.3%',
               '21': 'Injury Leukocytes 12.3%',
               '23': 'Injury Leukocytes 12.3%',
               '24': 'Injury Leukocytes 12.3%',
               '30': 'Injury Leukocytes 12.3%',
               '31': 'Injury Leukocytes 12.3%',
                }

adata_log1p.obs['Major Clusters'] = adata_log1p.obs['louvain'].map(major_clust)


sc.pl.umap(adata_log1p, color=['Major Clusters'], ncols=2, save='.mcs.umap.png')
sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='Major Clusters', standard_scale='var', save='.major_clusts.png')
sc.pp.neighbors(adata_log1p, n_neighbors=30, random_state=42, use_rep='MAGIC RNA Marks')

sc.tl.paga(adata_log1p)
sc.pl.paga(adata_log1p, show=False)
sc.tl.umap(adata_log1p, init_pos='paga', random_state=42)
sc.pl.umap(adata_log1p, color=['Pdgfra', 'Ptprc', 'Acta2', 'Major Clusters'], ncols=2, save='.mcs.magic.feats.png')

sc.pl.umap(adata_log1p, color=['louvain'], save='.scs.png')
sc.pl.umap(adata_log1p, color=['Major Clusters'], save='.mcs.png')
sc.pl.umap(adata_log1p, color=['Samples'], save='.samples.png')

sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='Major Clusters', standard_scale='var', save='.mcs.png')
sc.pl.dotplot(adata_log1p, var_names=rna_mark, groupby='louvain', standard_scale='var', dendrogram=True, save='.scs.png')

# adata_log1p.write('adata_log1p.fig.2.3.h5ad')