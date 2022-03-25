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

from palettable.scientific.sequential import Bilbao_10
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
sc.settings.figdir='./paper.figs/fig7/'

print(os.getcwd())
os.chdir("/home/UTHSCSA/iskra/CytOF_Datasets/10.08-10.10-Debarcoded/GG")
#
# mc_recon_expr = pd.read_csv('saucie.recon.clust.df.v2.csv')


clustering_channels = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU']

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


# expr2 = pd.read_csv("10.08.2019.gg.in.vivo.b4.csv", sep=',')
#
# expr2_adata = ad.AnnData(expr2[mapping[:-2]])
# expr2_adata.obs['Sample'] = expr2['Sample'].values
# expr2_adata.obs['Batch'] = 1
# # expr2_adata = expr2_adata[expr2_adata[:, 'CD31'].X < 1.44]
# expr2_adata.obs['n_counts'] = expr2_adata.X.sum(axis=1)
#
# expr2_adata.var_names_make_unique()
# sc.pp.filter_cells(expr2_adata, min_genes=1)

expr3 = pd.read_csv("10.15.2019.gg.DOX.tx.b4.csv", sep=',')

expr3_adata = ad.AnnData(expr3[mapping[:-2]])

expr3_adata.obs['Sample'] = expr3['Sample'].values
expr3_adata.obs['Batch'] = 1
# expr2_adata = expr2_adata[expr2_adata[:, 'CD31'].X < 1.44]
expr3_adata.obs['n_counts'] = expr3_adata.X.sum(axis=1)

expr3_adata.var_names_make_unique()
sc.pp.filter_cells(expr3_adata, min_genes=1)

#
# vivo_adata1 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[0]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)
#
# vivo_adata2 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[1]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)
#
# vivo_adata3 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[2]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)
#
# vivo_adata4 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[3]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)
#
# vivo_adata5 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[4]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)
#
# vivo_adata6 = sc.pp.subsample(expr2_adata[expr2_adata.obs['Sample'] == np.unique(expr2_adata.obs['Sample'])[5]],
#                               n_obs=15000,
#                               random_state=42,
#                               copy=True)



vivo_adata7 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[0]],
                              n_obs=20000, #66689
                              random_state=42,
                              copy=True)

vivo_adata8 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[1]],
                              n_obs=20000,#53005
                              random_state=42,
                              copy=True)

vivo_adata9 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[2]],
                              n_obs=20000,#55269
                              random_state=42,
                              copy=True)
#
# vivo_adata10 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[3]],
#                               n_obs=1500,#1700 exclude
#                               random_state=42,
#                               copy=True)
#
# vivo_adata11 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[4]],
#                               n_obs=1500,#1564 exclude
#                               random_state=42,
#                               copy=True)

vivo_adata12 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[5]],
                              n_obs=20000,#50162
                              random_state=42,
                              copy=True)



vivo_adata13 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[6]],
                              n_obs=20000,
                              random_state=42,
                              copy=True)

vivo_adata14 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[7]],
                              n_obs=20000,
                              random_state=42,
                              copy=True)

vivo_adata15 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[8]],
                              n_obs=20000,
                              random_state=42,
                              copy=True)

vivo_adata16 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[9]],
                              n_obs=20000,
                              random_state=42,
                              copy=True)
#
# vivo_adata17 = sc.pp.subsample(expr3_adata[expr3_adata.obs['Sample'] == np.unique(expr3_adata.obs['Sample'])[10]],
#                               n_obs=8000,#8439 Exclude
#                               random_state=42,
#                               copy=True)


expr_adata = ad.AnnData.concatenate(
                                     # vivo_adata1,
                                     # vivo_adata2,
                                     # vivo_adata3,
                                     # vivo_adata4,
                                     # vivo_adata5,
                                     # vivo_adata6,
                                     vivo_adata7,
                                     vivo_adata8,
                                     vivo_adata9,
                                     # vivo_adata10,
                                     # vivo_adata11,
                                     vivo_adata12,
                                     vivo_adata13,
                                     vivo_adata14,
                                     vivo_adata15,
                                     vivo_adata16,
                                     # vivo_adata17
                                     )


# sc.pl.violin(expr2_adata, ['n_counts', 'n_genes'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              # save='.vivo.pre.qc.plots.png'
#              )
expr_adata.obs['n_counts'] = expr_adata.X.sum(axis=1)


sc.pp.neighbors(expr_adata, n_neighbors=30, random_state=42, method='umap')
sc.tl.louvain(expr_adata, resolution=1, random_state=42)

sc.tl.paga(expr_adata, groups='louvain')
sc.pl.paga(expr_adata, plot=True, random_state=42, save='.paga.connectivity.png')
sc.tl.umap(expr_adata, init_pos='paga', random_state=42)


sc.pl.umap(expr_adata, color='louvain', ncols=3, legend_loc='on data', save='.unsupervised.louvain.png')
sc.pl.umap(expr_adata, color=['CD146', 'CD140a', 'CD45', 'CD31'], ncols=2, save='.key_markers.png')
sc.pl.umap(expr_adata, color=['aSMA', 'HSP60', 'MTCO1', 'IdU'], ncols=2, save='.fn.markers.png')


sc.plotting.dotplot(expr_adata,
                    groupby='louvain',
                    var_names=cytof_marks,
                    standard_scale='var',
                    dendrogram=True,
                    save='.dox.louvain.dotplot.png')

group_anno = {
               'F1.10.08.19.gg': 'Control',
               'F2.10.08.19.gg': 'Control',
               'F3.10.08.19.gg': 'Control',
               'M1.10.08.19.gg': 'Control',
               'M2.10.08.19.gg': 'Control',
               'M3.10.08.19.gg': 'Control',
               'F24D1F1': 'DOX 24 Hour',
               'F24D2F2': 'DOX 24 Hour',
               'F24D3F3': 'DOX 24 Hour',
               'M24D1M1': 'DOX 24 Hour',
               'M24D2M2': 'DOX 24 Hour',
               'M24D3M3': 'DOX 24 Hour',
               'F72D1M7': 'DOX 72 Hour',
               'F72D2M8': 'DOX 72 Hour',
               'M72D1M7': 'DOX 72 Hour',
               'M72D2M8': 'DOX 72 Hour',
               'M72D3M9': 'DOX 72 Hour',
}

expr_adata.obs['Groups'] = pd.Categorical(expr_adata.obs['Sample'].map(group_anno))

mouse_anno = {
               'F1.10.08.19.gg': 'C-F1',
               'F2.10.08.19.gg': 'C-F2',
               'F3.10.08.19.gg': 'C-F3',
               'M1.10.08.19.gg': 'C-M1',
               'M2.10.08.19.gg': 'C-M2',
               'M3.10.08.19.gg': 'C-M3',
               'F24D1F1': 'D24-F1',
               'F24D2F2': 'D24-F2',
               'F24D3F3': 'D24-F3',
               'M24D1M1': 'D24-M1',
               'M24D2M2': 'D24-M2',
               'M24D3M3': 'D24-M3',
               'F72D1M7': 'D72-F7',
               'F72D2M8': 'D72-F8',
               'M72D1M7': 'D72-M7',
               'M72D2M8': 'D72-M8',
               'M72D3M9': 'D72-M9',
}

expr_adata.obs['Mouse'] = pd.Categorical(expr_adata.obs['Sample'].map(mouse_anno))


sc.pl.umap(expr_adata, color=['n_counts', 'n_genes', 'Groups'], save='.counts.genes.png', ncols=3)

sum(expr_adata.obs['louvain'].isin(['0', '1', '2', '5', '9', '13' '22']))/160000


sum(expr_adata.obs['louvain'].isin(['4', '14', '16', '18', '21']))/160000
sum(expr_adata.obs['louvain'].isin(['3', '16']))/160000

sum(expr_adata.obs['louvain'].isin(['7', '8', '10']))/160000
sum(expr_adata.obs['louvain'].isin(['11']))/160000

sum(expr_adata.obs['louvain'].isin(['6', '12', '20']))/160000

sum(expr_adata.obs['louvain'].isin(['15', '17', '19']))/160000

major_clust = {'0': 'Homeostatic Endothelium 47.8%',
               '1': 'Homeostatic Endothelium 47.8%',
               '2': 'Homeostatic Endothelium 47.8%',
               '5': 'Homeostatic Endothelium 47.8%',
               '9': 'Homeostatic Endothelium 47.8%',
               '13': 'Homeostatic Endothelium 47.8%',
               '22': 'Homeostatic Endothelium 47.8%',

               '4': 'Homeostatic Leukocytes 11.2%',
               '14': 'Homeostatic Leukocytes 11.2%',
               '18': 'Homeostatic Leukocytes 11.2%',
               '21': 'Homeostatic Leukocytes 11.2%',

               '3': 'Injury Leukocytes 8.36%',
               '16': 'Injury Leukocytes 8.36%',

               '7':  'Homeostatic Fibroblasts 15.1%',
               '8':  'Homeostatic Fibroblasts 15.1%',
               '10':  'Homeostatic Fibroblasts 15.1%',

               '11':  'Injury Fibroblasts 3.23%',

                '6': 'Pericytes 9.36%',
               '12': 'Pericytes 9.36%',
               '20': 'Pericytes 9.36%',

               '15':  'Ambiguous Cells 4.42%',
               '17':  'Ambiguous Cells 4.42%',
               '19':  'Ambiguous Cells 4.42%',
               }

expr_adata.obs['Major Clusters'] = expr_adata.obs['louvain'].map(major_clust)

sum(expr_adata.obs['louvain'].isin(['11']))/160000
sum(expr_adata.obs['louvain'].isin(['6', '12', '20']))/160000
sum(expr_adata.obs['louvain'].isin(['7', '8', '10']))/160000


sub_clust = {
               '0': 'VEGFR2 Low Endothelium 15.0%',
               '13': 'VEGFR2 Low Endothelium 15.0%',

               '1': 'VEGFR2 Hi Endothelium 34.9%',
               '2': 'VEGFR2 Hi Endothelium 34.9%',
               '5': 'VEGFR2 Hi Endothelium 34.9%',
               '9': 'VEGFR2 Hi Endothelium 34.9%',

               '22': 'IdU+ Endothelium 48.1%',

               '4': 'Homeostatic Leukocytes 9.39%',
               '14': 'Homeostatic Leukocytes 9.39%',
               '18': 'Homeostatic Leukocytes 9.39%',
               '21': 'Homeostatic Leukocytes 9.39%',

               '3': 'P53+ Leukocytes 6.59%',
               '16': 'IdU+ Injury Leukocytes 1.77%',

               '7':  'Homeostatic Fibroblasts 15.1%',
               '8':  'Homeostatic Fibroblasts 15.1%',
               '10':  'Homeostatic Fibroblasts 15.1%',

               '11':  'Myofibroblasts 3.23%',

               '6': 'Non-Muscular Pericytes 9.36%',
               '12': 'Ly6C/Ly6AE High Pericytes 9.36%',
               '20': 'Ly6C/Ly6AE Low Pericytes 9.36%',

               '15':  'Ambiguous Cells 4.42%',
               '17':  'Ambiguous Cells 4.42%',
               '19':  'Ambiguous Cells 4.42%',
              }


expr_adata.obs['Subclusters'] = expr_adata.obs['louvain'].map(sub_clust)

sc.plotting.dotplot(expr_adata,
                    groupby='louvain',
                    var_names=cytof_marks,
                    standard_scale='var',
                    save='.dox.louvain.dotplot.png',
                    dendrogram=True)

sc.plotting.dotplot(expr_adata,
                    groupby='Major Clusters',
                    var_names=cytof_marks,
                    standard_scale='var',
                    save='.dox.Mcs.dotplot.png',
                    dendrogram=True)

sc.plotting.dotplot(expr_adata,
                    groupby='Subclusters',
                    var_names=cytof_marks,
                    standard_scale='var',
                    save='.dox.subclusters.dotplot.png',
                    dendrogram=True)

sc.pl.umap(expr_adata, color='louvain', ncols=3, legend_loc='on data', save='.unsupervised.louvain.png')
sc.pl.umap(expr_adata, color='Major Clusters', ncols=3, save='.major_clusters.png')
sc.pl.umap(expr_adata, color='Subclusters', ncols=3,  save='.subclusters.png')

sc.pl.paga(expr_adata, plot=True, color='Major Clusters', random_state=42, save='.paga.connectivity.png')

expr_adata.obs['Subclusters'] = [i[:-6] for i in expr_adata.obs['Subclusters']]
expr_adata.obs['Major Clusters'] = [i[:-6] for i in expr_adata.obs['Major Clusters']]

sns.set(font_scale=2.25)

add = expr_adata
sample_grouping = 'Sample'
cluster_grouping = 'Subclusters'
savefile = './paper.figs/fig7/dox.freq.mouse.sub_clusters.freq.png'

# indices =

prop_freq = pd.DataFrame(index=add.obs[sample_grouping].cat.categories, columns=np.unique(add.obs[cluster_grouping].to_numpy()))

for i in add.obs[sample_grouping].cat.categories:
    for j in np.unique(add.obs[cluster_grouping]):
        prop_freq[j].loc[i] = sum(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)/len(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)

prop_freq = prop_freq.T

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(17.5, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.T.astype,
               # vmin=-1.0,
               # vmax=1.0,
               standard_scale=1,
               cbar_kws={"ticks":[0,  0.5,  1], "label":"Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.close()

prop_freq = prop_freq.loc[prop_freq.index[cg.dendrogram_row.reordered_ind]]
prop_freq = prop_freq[prop_freq.columns[cg.dendrogram_col.reordered_ind]]

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(17.5, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.round(3),
               standard_scale=1,
               cbar_kws={"ticks": [0, 0.5, 1], "label": "Standard Scaled\nFrequency of Cells"})


plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.savefig(savefile, dpi=400)
plt.close()


add = expr_adata
sample_grouping = 'Sample'
cluster_grouping = 'Major Clusters'
savefile = './paper.figs/fig7/dox.freq.mouse.major_clusters.freq.png'

prop_freq = pd.DataFrame(index=add.obs[sample_grouping].cat.categories, columns=np.unique(add.obs[cluster_grouping].to_numpy()))

for i in add.obs[sample_grouping].cat.categories:
    for j in np.unique(add.obs[cluster_grouping]):
        prop_freq[j].loc[i] = sum(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)/len(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)

prop_freq = prop_freq.T

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.T.astype,
               # vmin=-1.0,
                standard_scale=1,
                cbar_kws={"ticks": [0, 0.5, 1], "label": "Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.close()

prop_freq = prop_freq.loc[prop_freq.index[cg.dendrogram_row.reordered_ind]]
prop_freq = prop_freq[prop_freq.columns[cg.dendrogram_col.reordered_ind]]

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.round(3),
               # vmin=-1.0,
               # vmax=0.75,
               standard_scale=0,
               cbar_kws={"ticks": [0,  0.5, 1], "label": "Standard Scaled\nCell Frequency"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.savefig(savefile, dpi=400)
plt.close()



add = expr_adata
sample_grouping = 'Groups'
cluster_grouping = 'Subclusters'
savefile = './paper.figs/fig7/dox.freq.groups.sub_clusters.freq.png'

prop_freq = pd.DataFrame(index=np.unique(add.obs[sample_grouping]), columns=np.unique(add.obs[cluster_grouping]))

for i in np.unique(add.obs[sample_grouping]):
    for j in np.unique(add.obs[cluster_grouping]):
        prop_freq[j].loc[i] = sum(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)/len(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)


prop_freq = prop_freq.T

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.T.astype,
               # vmin=-1.0,
               # vmax=1.0,
               standard_scale=1,
               cbar_kws={"ticks":[0,  0.5,  1], "label":"Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.45, top=.90, bottom=.15)
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

plt.close()

prop_freq = prop_freq.loc[prop_freq.index[cg.dendrogram_row.reordered_ind]]
prop_freq = prop_freq[prop_freq.columns[cg.dendrogram_col.reordered_ind]]

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.round(3),
               # vmin=-1.0,
               # vmax=0.25,
               standard_scale=1,
               cbar_kws={"ticks":[0,  0.5,  1], "label":"Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.45, top=.90, bottom=.15)
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

plt.savefig(savefile, dpi=400)
plt.close()

add = expr_adata
sample_grouping = 'Groups'
cluster_grouping = 'Major Clusters'
savefile = './paper.figs/fig7/dox.freq.groups.major_clusters.freq.png'

prop_freq = pd.DataFrame(index=np.unique(add.obs[sample_grouping]), columns=np.unique(add.obs[cluster_grouping]))

for i in np.unique(add.obs[sample_grouping]):
    for j in np.unique(add.obs[cluster_grouping]):
        prop_freq[j].loc[i] = sum(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)/len(add[add.obs[sample_grouping]==i].obs[cluster_grouping]==j)

prop_freq = prop_freq.T

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.T.astype,
               # vmin=-1.0,
               standard_scale=1,
               cbar_kws={"ticks":[0,  0.5,  1], "label":"Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.close()

prop_freq = prop_freq.loc[prop_freq.index[cg.dendrogram_row.reordered_ind]]
prop_freq = prop_freq[prop_freq.columns[cg.dendrogram_col.reordered_ind]]

cg = sns.clustermap(data=prop_freq.astype('float64'),
               cmap=Bilbao_10.mpl_colormap,
               figsize=(15, 15),
               method='ward',
               row_cluster=True,
               col_cluster=True,
               # annot=prop_freq.round(3),
               # vmin=-1.0,
               standard_scale=1,
               cbar_kws={"ticks":[0,  0.5,  1], "label":"Standard Scaled\nFrequency of Cells"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.55, top=.90, bottom=.15)
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

plt.savefig(savefile, dpi=400)
plt.close()

sc.write(filename='fig.7.h5ad', adata=expr_adata)