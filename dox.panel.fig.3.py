import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.api as sc
import scanpy.external as sce
import os
import louvain
# import phate
import seaborn as sns
import anndata as ad
import time
import random
import tensorflow as tf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from matplotlib import cm, colors

from palettable.scientific.sequential import Bilbao_20
from palettable.cartocolors.qualitative import Vivid_3
from palettable.cartocolors.qualitative import Vivid_4
from palettable.cartocolors.qualitative import Bold_4
from palettable.cartocolors.qualitative import Bold_6
from palettable.cartocolors.qualitative import Vivid_10
from palettable.scientific.diverging import Vik_20
from palettable.scientific.diverging import Berlin_10

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

from model import SAUCIE
from loader import Loader

sc.settings.verbosity = 4  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, dpi_save=400)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_versions()
sc.settings.figdir='./paper.figs/fig3/'

print(os.getcwd())
os.chdir("/home/UTHSCSA/iskra/CytOF_Datasets/10.08-10.10-Debarcoded/GG")

mapping = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
       'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn', 'IdU',
       'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
       'aSMA', 'CD31', 'VEGFR2', 'ActCaspase3', 'P53', 'MTCO1',
       'HSP60', 'Ki67', 'pRb', 'CyclinB1', 'P21', 'bCatTotal',
       'bCatActive', 'pHistoneH3', 'Sample', 'Batch']

cytof_marks = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
       'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn', 'IdU',
       'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
       'aSMA', 'CD31', 'VEGFR2', 'ActCaspase3', 'P53', 'MTCO1',
       'HSP60', 'Ki67', 'pRb', 'CyclinB1', 'P21', 'bCatTotal',
       'bCatActive', 'pHistoneH3']

expr2 = pd.read_csv("10.08.2019.gg.in.vivo.b4.csv", sep=',')

expr2_adata = ad.AnnData(expr2[mapping[:-2]])
expr2_adata.obs['Sample'] = expr2['Sample'].values
expr2_adata.obs['Batch'] = 1

expr2_adata.var_names_make_unique()
sc.pp.filter_cells(expr2_adata, min_genes=1)

vivo_adata1 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[0]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

vivo_adata2 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[1]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

vivo_adata3 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[2]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

vivo_adata4 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[3]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

vivo_adata5 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[4]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

vivo_adata6 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[5]],
                              n_obs=7500,
                              random_state=42,
                              copy=True)

expr2_adata = ad.AnnData.concatenate(vivo_adata1,
                                     vivo_adata2,
                                     vivo_adata3,
                                     vivo_adata4,
                                     vivo_adata5,
                                     vivo_adata6,
                                     )

expr2_adata.obs['n_counts'] = expr2_adata.X.sum(axis=1)

sc.pl.violin(expr2_adata, ['n_counts', 'n_genes'],
             jitter=0.4, multi_panel=True, stripplot=False,
             save='.vivo.pre.qc.plots.png'
             )

sc.pp.neighbors(expr2_adata, n_neighbors=30, random_state=42, method='umap')
sc.tl.louvain(expr2_adata, resolution=1, random_state=42)

sc.tl.paga(expr2_adata, groups='louvain')
sc.pl.paga(expr2_adata, plot=True, random_state=42)
sc.tl.umap(expr2_adata, init_pos='paga', random_state=42)


sc.pl.umap(expr2_adata, color='louvain', ncols=3, legend_loc='on data', save='.unsupervised.louvain.png')
sc.pl.umap(expr2_adata, color=['CD146', 'CD140a', 'CD45', 'CD31'], ncols=2, save='.key_markers.png')
# sc.pl.umap(expr2_adata, color=['aSMA', 'HSP60', 'MTCO1'], ncols=3)
# sc.pl.umap(expr2_adata, color=['n_counts', 'n_genes'], ncols=3)


sum(expr2_adata.obs['louvain'].isin(['0', '1', '2', '4', '5', '6', '11', '14', '18', '20']))/45000 #Endo
sum(expr2_adata.obs['louvain'].isin(['3', '9', '15']))/45000 #Fibro
sum(expr2_adata.obs['louvain'].isin(['7', '10', '12']))/45000 #Peri
sum(expr2_adata.obs['louvain'].isin(['8', '13', '17', '19']))/45000#Leuko
sum(expr2_adata.obs['louvain'].isin(['16']))/45000#Transitional

major_clust = {'0': 'Endothelium 61.4%',
               '1': 'Endothelium 61.4%',
               '2': 'Endothelium 61.4%',
               '4': 'Endothelium 61.4%',
               '5': 'Endothelium 61.4%',
               '6': 'Endothelium 61.4%',
               '11': 'Endothelium 61.4%',
               '14': 'Endothelium 61.4%',
               '18': 'Endothelium 61.4%',
               '20': 'Endothelium 61.4%',

               '3':  'Fibroblasts 15.8%',
               '9':  'Fibroblasts 15.8%',
               '15':  'Fibroblasts 15.8%',

               '8': 'Leukocytes 8.70%',
               '13': 'Leukocytes 8.70%',
               '17': 'Leukocytes 8.70%',
               '19': 'Leukocytes 8.70%',

               '7': 'Pericytes 13.3%',
               '10': 'Pericytes 13.3%',
               '12': 'Pericytes 13.3%',

               '16': 'Ambiguous Cells 0.0910%'
                }

expr2_adata.obs['Major Clusters'] = expr2_adata.obs['louvain'].map(major_clust)

sum(expr2_adata.obs['louvain'].isin(['8']))/45000 #Peri

sub_clust =   {'0': 'Vimentin Low Endothelium 33.6%',
               '1': 'Vimentin Low Endothelium 33.6%',
               '2': 'Vimentin Low Endothelium 33.6%',
               '4': 'Vimentin Hi Endothelium 22.6%',
               '5': 'Vimentin Hi Endothelium 22.6%',
               '6': 'Vimentin Hi Endothelium 22.6%',
               '11': 'CD90+ Endothelium 3.23%',
               '14': 'Ly6C/Ly6AE Low Endothelium 2.10%',
               '18': 'Ly6C/Ly6AE Low Endothelium 2.10%',
               '20': 'IdU+ Endothelium 0.156%',

               '3':  'PDGFRa Hi Fibroblasts 9.22%',
               '9':  'PDGFRa Hi Ly6C/Ly6AE Hi Fibroblasts 5.62%',
               '15':  'PDGFRa Low Fibroblasts 0.0911%',

               '8': 'FolR2 Hi Leukocytes 5.84%',
               '13': 'CD11b- IAIE Hi Leukocytes 1.57%',
               '17': 'CD11b+ IAIE Low Leukocytes 0.827%',
               '19': 'CD90+ Leukocytes 0.464%',

               '7': 'Non-Muscular Pericytes 5.96%',
               '10': 'Ly6C/Ly6AE High Pericytes 4.51%',
               '12': 'Ly6C/Ly6AE Low Pericytes 2.78%',

               '16': 'Ambiguous Cells 0.0910%'
                }


expr2_adata.obs['Subclusters'] = expr2_adata.obs['louvain'].map(sub_clust)

sc.plotting.dotplot(expr2_adata, groupby='Major Clusters', var_names=cytof_marks, standard_scale='var', save='.vivo.mcs.dotplot.png')
sc.plotting.dotplot(expr2_adata, groupby='Subclusters', var_names=cytof_marks, standard_scale='var', save='.vivo.subclusters.dotplot.png')
sc.plotting.dotplot(expr2_adata, groupby='louvain', var_names=cytof_marks, standard_scale='var', dendrogram=True, save='.vivo.louvain.dotplot.png')


# sc.pl.umap(expr2_adata, color=['Major Clusters',  'SAUCIE Clusters', 'Subclusters'], ncols=1)
sc.pl.umap(expr2_adata, color='Major Clusters', save='.major_clusters.png')
sc.pl.umap(expr2_adata, color='Subclusters', save='.subclusters.png')


major_clust = {'0': 'Endothelium',
               '1': 'Endothelium',
               '2': 'Endothelium',
               '4': 'Endothelium',
               '5': 'Endothelium',
               '6': 'Endothelium',
               '11': 'Endothelium',
               '14': 'Endothelium',
               '18': 'Endothelium',
               '20': 'Endothelium',

               '3':  'Fibroblasts',
               '9':  'Fibroblasts',
               '15':  'Fibroblasts',

               '8': 'Leukocytes',
               '13': 'Leukocytes',
               '17': 'Leukocytes',
               '19': 'Leukocytes',

               '7': 'Pericytes',
               '10': 'Pericytes',
               '12': 'Pericytes',

               '16': 'Ambiguous Cells'
                }

expr2_adata.obs['Major Clusters'] = expr2_adata.obs['louvain'].map(major_clust)

# expr2_adata.write_h5ad('expr2_adata.fig.3.h5ad')
# per_endo = [0.6636*100, 0.6567*100, 0.7437*100, 0.5937*100, 0.4793*100, 0.5461*100]
# per_fibro = [0.1584*100, 0.1376*100, 0.1023*100, 0.1646*100, 0.1956*100, 0.1865*100]
# per_leuko = [0.05026*100, 0.06547*100, 0.0511*100, 0.0979*100, 0.1396*100, 0.1177*100]
# per_peri = [0.1177*100, 0.1337*100, 0.09653*100, 0.1352*100, 0.1713*100, 0.1408*100]
# per_trans = [0.01000*100, 0.006533*100, 0.0064*100, 0.00853*100, 0.01413*100, 0.0088*100]
#
# per_list = [[per_endo, per_fibro, per_leuko, per_peri, per_trans]]
# df = pd.DataFrame(index=['B1F1', 'B1F2', 'B1F3', 'B2M1', 'B2M2', 'B2M3'], columns=['Endothelium %', 'Fibroblast %', 'Leukocyte %', 'Pericyte %', 'Transitional %'])
# df['Endothelium %'] = [0.6636*100, 0.6567*100, 0.7437*100, 0.5937*100, 0.4793*100, 0.5461*100]
# df['Fibroblast %'] = [0.1584*100, 0.1376*100, 0.1023*100, 0.1646*100, 0.1956*100, 0.1865*100]
# df['Leukocyte %'] = [0.05026*100, 0.06547*100, 0.0511*100, 0.0979*100, 0.1396*100, 0.1177*100]
# df['Pericyte %'] = [0.1177*100, 0.1337*100, 0.09653*100, 0.1352*100, 0.1713*100, 0.1408*100]
# df['Transitional %'] = [0.01000*100, 0.006533*100, 0.0064*100, 0.00853*100, 0.01413*100, 0.0088*100]
# df['Batch'] = [1, 1, 1, 2, 2, 2]
#
# df['']