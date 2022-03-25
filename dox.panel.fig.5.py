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
sc.settings.figdir='./paper.figs/fig5/'

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
expr3 = pd.read_csv("10.08.2019.gg.in.vitro.b4.csv", sep=',')

expr3_adata = ad.AnnData(expr3[mapping[:-2]])
expr3_adata.obs['Sample'] = expr3['Sample'].values
expr3_adata.obs['Batch'] = 1


vitro_adata1 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[0]],
                              n_obs=11250,
                              random_state=42,
                              copy=True)

vitro_adata2 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[1]],
                              n_obs=11250,
                              random_state=42,
                              copy=True)

vitro_adata3 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[2]],
                              n_obs=11250,
                              random_state=42,
                              copy=True)

vitro_adata4 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[3]],
                              n_obs=11250,
                              random_state=42,
                              copy=True)

expr3_adata = ad.AnnData.concatenate(vitro_adata1,
                                     vitro_adata2,
                                     vitro_adata3,
                                     vitro_adata4)

expr3_adata.var_names_make_unique()
sc.pp.filter_cells(expr3_adata, min_genes=1)

expr3_adata.obs['n_counts'] = expr3_adata.X.sum(axis=1)
sc.pl.violin(expr3_adata, ['n_counts', 'n_genes'],
             jitter=0.4, multi_panel=True, stripplot=False,
              save='vitro.pre.qc.plots.png'
             )


sc.pp.neighbors(expr3_adata, n_neighbors=30, random_state=42, method='umap')
sc.tl.louvain(expr3_adata, resolution=1, random_state=42)

sc.tl.paga(expr3_adata, groups='louvain')
sc.pl.paga(expr3_adata, plot=True, random_state=42)
sc.tl.umap(expr3_adata, init_pos='paga', random_state=42)


sc.pl.umap(expr3_adata, color='louvain', ncols=3, legend_loc='on data', save='.unsupervised.louvain.png')
sc.pl.umap(expr3_adata, color=['CD31', 'CD140a', 'CD45'], ncols=3, save='.key_markers.png')

sc.pl.dotplot(expr3_adata, groupby='louvain',
              var_names=cytof_marks, dendrogram=True,
              standard_scale='var', save='.invitro.louvain.png')


sum(expr3_adata.obs['louvain'].isin(['10', '13', '14', '15']))/45000 #Endo-like
sum(expr3_adata.obs['louvain'].isin(['0', '1', '9']))/45000 #Leuko
sum(expr3_adata.obs['louvain'].isin(['2', '3', '4', '5', '6', '7', '8', '11', '12']))/45000 #Fibro

major_clust = {'0':  'Leukocytes 30.5%',
               '1':  'Leukocytes 30.5%',
               '9':  'Leukocytes 30.5%',

               '10': 'Endothelial-like Cells 5.6%',
               '13': 'Endothelial-like Cells 5.6%',
               '14': 'Endothelial-like Cells 5.6%',
               '15': 'Endothelial-like Cells 5.6%',

               '2': 'Fibroblasts 63.9%',
               '3': 'Fibroblasts 63.9%',
               '4': 'Fibroblasts 63.9%',
               '5': 'Fibroblasts 63.9%',
               '6': 'Fibroblasts 63.9%',
               '7': 'Fibroblasts 63.9%',
               '8': 'Fibroblasts 63.9%',
               '11': 'Fibroblasts 63.9%',
               '12': 'Fibroblasts 63.9%',
              }

expr3_adata.obs['Major Clusters'] = expr3_adata.obs['louvain'].map(major_clust)


major_clust = {'0':  'FolR2 Hi Leukocytes 13.9%',
               '1':  'FolR2 Low Leukocytes 10.8%',
               '9':  'Replicating Leukocytes 5.78%',

               '10': 'CD140a Hi Endothelial-like Cells 3.04%',
               '13': 'ActCaspase3 Hi Endothelial-like Cells 1.56%',
               '14': 'CD90 Hi Endothelial-like Cells 0.511%',
               '15': 'Replicating Endothelial-like Cells 0.482%',

               '2': 'Ly6AE Low Fibroblasts 19.3%',
               '3': 'Ly6AE Hi Fibroblasts 37.9%',
               '4': 'Ly6AE Hi Fibroblasts 37.9%',
               '5': 'Ly6AE Hi Fibroblasts 37.9%',
               '6': 'Postn Hi Fibroblasts 7.98%',
               '7': 'Ly6AE Low Fibroblasts 19.3%',
               '8': 'Marker Low Fibroblasts 7.10%',
               '11': 'Ly6AE Low Fibroblasts 19.3%',
               '12': 'Replicating Fibroblasts 1.83%',
              }

expr3_adata.obs['Subclusters'] = expr3_adata.obs['louvain'].map(major_clust)
expr3_adata.obs['Condition'] = [i[:2] for i in np.asarray(expr3_adata.obs['Sample'])]

sc.pl.umap(expr3_adata, color='louvain', ncols=3, legend_loc='on data', save='.invitro.louvain.png')
sc.pl.umap(expr3_adata, color='Major Clusters', ncols=3, save='.invitro.mcs.png')
# sc.pl.paga(expr3_adata, plot=True, random_state=42, color='Major Clusters', save='.invitro.paga.mcs.png')
sc.pl.umap(expr3_adata, color='Subclusters', ncols=3, save='.invitro.subclusters.png')
sc.pl.dotplot(expr3_adata, groupby='Subclusters', var_names=cytof_marks, dendrogram=True, standard_scale='var', save='.invitro.subclusters.png')

expr3_adata.obs['Subclusters'] = [i[:-6] for i in expr3_adata.obs['Subclusters']]

prop_freq = pd.DataFrame(index=np.unique(expr3_adata.obs['Condition']), columns=np.unique(expr3_adata.obs['Subclusters']))

for i in np.unique(expr3_adata.obs['Condition']):
    for j in np.unique(expr3_adata.obs['Subclusters']):
        prop_freq[j].loc[i] = sum(expr3_adata[expr3_adata.obs['Condition']==i].obs['Subclusters']==j)/11250

prop_freq = prop_freq.T

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_20.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.T.astype,
               # vmin=-1.0,
               # vmax=1.0,
               cbar_kws={"ticks":[0, 0.25, 0.5, 0.75, 1], "label":"Frequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.75, top=.90, bottom=.15)
#
# cg.ax_row_dendrogram.set_visible(False)
# dendro_box = cg.ax_row_dendrogram.get_position()
# dendro_box.x0 = 0.205
# dendro_box.x1 = 0.225
#cg.cax.set_position(dendro_box)
cg.cax.yaxis.set_ticks_position("left")
cg.cax.yaxis.set_label_position("left")

for a in cg.ax_row_dendrogram.collections:
    a.set_linewidth(2)

for a in cg.ax_col_dendrogram.collections:
    a.set_linewidth(2)

plt.savefig('./paper.figs/fig5/in.vitro.freq.png', dpi=400)
plt.close()


prop_freq = prop_freq.loc[prop_freq.index[cg.dendrogram_row.reordered_ind]]
prop_freq = prop_freq[prop_freq.columns[cg.dendrogram_col.reordered_ind]]


cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_20.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               annot=prop_freq.round(3),
               # vmin=-1.0,
               # vmax=1.0,
               cbar_kws={"ticks":[0, 0.25, 0.5, 0.75, 1], "label":"Frequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.75, top=.90, bottom=.15)
#
# cg.ax_row_dendrogram.set_visible(False)
# dendro_box = cg.ax_row_dendrogram.get_position()
# dendro_box.x0 = 0.205
# dendro_box.x1 = 0.225
#cg.cax.set_position(dendro_box)
cg.cax.yaxis.set_ticks_position("left")
cg.cax.yaxis.set_label_position("left")

for a in cg.ax_row_dendrogram.collections:
    a.set_linewidth(2)

for a in cg.ax_col_dendrogram.collections:
    a.set_linewidth(2)

plt.savefig('./paper.figs/fig5/in.vitro.freq.png', dpi=400)
plt.close()