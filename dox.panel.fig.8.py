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
from tensorflow.python.framework import ops
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
sc.settings.figdir='./paper.figs/fig8/'

print(os.getcwd())
os.chdir("/home/UTHSCSA/iskra/CytOF_Datasets/10.08-10.10-Debarcoded/GG")


clustering_channels = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU']

mapping = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
            'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn',
            'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
            'aSMA', 'CD31', 'VEGFR2', 'P53', 'MTCO1',
            'HSP60', 'Ki67', 'CyclinB1', 'P21', 'bCatTotal', 'IdU', 'Sample', 'Batch']

cytof_marks = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
       'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn', 'IdU',
       'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
       'aSMA', 'CD31', 'VEGFR2', 'ActCaspase3', 'P53', 'MTCO1',
       'HSP60', 'Ki67', 'pRb', 'CyclinB1', 'P21', 'bCatTotal',
       'bCatActive', 'pHistoneH3']

rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU', 'Sample', 'Batch']

ct_adata = sc.read_h5ad('fig.7.h5ad')
#
# ct_adata = sc.pp.subsample(ct_adata,
#                               n_obs=42843,
#                               random_state=42,
#                               copy=True)

sc_adata = sc.read_h5ad('adata_log1p.fig.2.3.h5ad')


sc_expr = sc_adata.to_df()
sc_expr = sc_expr.astype('float64', copy=True)

sc_expr['Batch'] = 0
sc_expr['Sample'] = 'scRNAseq'
sc_expr['IdU'] = sc_expr['Mki67'].values
sc_expr['Ly6AE'] = sc_expr['Ly6a'].values + sc_expr['Ly6e'].values

sc_expr = sc_expr[rna_mark]
sc_expr.columns = rna_mark

ct_expr = ct_adata.to_df()
ct_expr = ct_expr.astype('float64', copy=True)

ct_expr['Batch'] = 1
ct_expr['Sample'] = 'CyTOF'


ct_expr = ct_expr[mapping]
ct_expr.columns = rna_mark


expr = pd.concat([sc_expr[rna_mark],
                  ct_expr[rna_mark].sample(replace=True, n=sc_expr.shape[0])
                  # expr4[rna_mark]
                  ], sort=False)
expr = expr.fillna(0)


exprcc = expr[expr.columns[expr.dtypes == 'float64']]
cc = exprcc.columns
cc_r = cc+"_r"
data = exprcc.values[:, 0:(exprcc.shape[1])]


##
##
##400k steps was used for batch correction on in vitro+in vivo dataset
##140k steps was used for batch correction of in vivo only dataset
##
# # ##
train, test = train_test_split(expr, train_size=0.80, test_size=0.20)

train_b = train['Batch'].values
train_data = train[train.columns[train.dtypes == 'float64']]
train_data = train_data.values[:, 0:(train_data.shape[1])]

test_b = test['Batch'].values
test_data = test[test.columns[test.dtypes == 'float64']]
test_data = test_data.values[:, 0:(test_data.shape[1])]

trainset = Loader(train_data, labels=np.int32(train_b), shuffle=True)
testset = Loader(test_data, labels=np.int32(test_b), shuffle=True)
# #
loadtrain = Loader(data, labels=np.int32(expr['Batch']), shuffle=False)
# loadeval = Loader(data, labels=np.int32(expr['Batch']), shuffle=False)

# eset = expr
nminibatches = 1000
steps = 1

expr = pd.concat([sc_expr[rna_mark], ct_expr[rna_mark]], sort=False)

expr = expr.fillna(0)

exprcc = expr[expr.columns[expr.dtypes == 'float64']]
cc = exprcc.columns
cc_r = cc + "_r"
data = exprcc.values[:, 0:(exprcc.shape[1])]
loadeval = Loader(data, labels=np.int32(expr['Batch']), shuffle=False)


## Keep constant for batch correction
l1 = 1024
l2 = 512
l3 = 256
l4 = 2

lb = 0.14

saucie = SAUCIE(data.shape[1],
                    layers=[l1, l2, l3, l4],
                    learning_rate=0.001,
                    lambda_b=lb,
                    # limit_gpu_fraction=0.20,
                    no_gpu=True
                    )

st = time.time()
saucie.train(load=trainset, steps=nminibatches, batch_size=256)
print("----Training Step: " + str((nminibatches)) + " ------Runtime:" + str(time.time() - st))
print("--- Train Loss: "+str(saucie.get_loss(trainset))+" Test Loss: "+str(saucie.get_loss(testset)))


reconstruction = saucie.get_reconstruction(loadtrain)
expr_train = pd.DataFrame(reconstruction[0], columns=cc_r)
expr_train.to_csv('./paper.figs/fig8/diss.saucie-cd.trainset.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')

reconstruction = saucie.get_reconstruction(loadeval)

expr_bc = pd.DataFrame(reconstruction[0], columns=cc_r)

embedding = saucie.get_embedding(loadeval)

expr_bc['SAUCIE1'] = embedding[0][:, 0]
expr_bc['SAUCIE2'] = embedding[0][:, 1]
expr_bc['Batch'] = expr['Batch'].values
expr_bc['Sample'] = expr['Sample'].values
expr_bc.to_csv('./paper.figs/fig8/paper.d3.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')

plt.figure(num=777,
           figsize=[14, 12])
ax = plt.subplot(111)
cmap = cm.get_cmap(Vivid_10.mpl_colormap, len(np.unique(expr_bc['Batch'])))
plt.scatter(x=expr_bc['SAUCIE1'],
            y=expr_bc['SAUCIE2'],
            s=0.1,
            marker=',',
            # alpha=0.1,
            c=pd.Categorical(expr_bc['Batch']).codes,
            cmap=cmap)
plt.xlabel("SAUCIE1")
plt.ylabel("SAUCIE2")
# plt.xlim([-3.25, 3.25])
# plt.ylim([-4,4])
plt.axis('tight')
plt.setp(ax.spines.values(), linewidth=2)
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

for j in np.unique(expr_bc['Batch']):
    plt.annotate(s=str(j),
                 xy=(expr_bc[expr_bc['Batch'] == j]['SAUCIE1'].median(),
                     expr_bc[expr_bc['Batch'] == j]['SAUCIE2'].median()),
                 size=15)

# plt.axis('off')
plt.title("Batch Distribution SAUCIE-B", fontsize=45, pad=15)
plt.xlabel("SAUCIE1", fontsize=45, labelpad=15)
plt.ylabel("SAUCIE2", fontsize=45, labelpad=15)
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    right=False,
    left=False,
    top=False,
    labelbottom=False,
    labelleft=False
)
plt.savefig('./paper.figs/fig8/gg_matrix_batch.'+str(lb)+'.'+str(nminibatches)+'.'+str(steps)+".png",
            dpi=400)
plt.close()


expr_eval = pd.read_csv('./paper.figs/fig8/paper.d3.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')
expr = pd.read_csv('./paper.figs/fig8/diss.saucie-cd.trainset.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')
expr = expr.drop('Unnamed: 0',  axis=1)
expr_eval = expr_eval.drop('Unnamed: 0',  axis=1)

clustering_channels = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU']

cc_r = [clustering_channels + "_r" for clustering_channels in clustering_channels]
cc_n = [clustering_channels + "_n" for clustering_channels in clustering_channels]


data = expr.values[:, 0:(expr.shape[1])]

ops.reset_default_graph()

sess = tf.InteractiveSession()

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

lc = 0.035
ld = 0.070
layer_c = 2
saucie = SAUCIE(data.shape[1],
                layers=[512, 256, 128, 2],
                learning_rate=0.001,
                layer_c=layer_c,
                lambda_c=lc,
                lambda_d=ld,
                no_gpu=True)


train, test = train_test_split(expr, train_size=0.80, test_size=0.20)

train_data = train[cc_r]
train_data = train_data.values[:, 0:(train_data.shape[1])]

test_data = test[cc_r]
test_data = test_data.values[:, 0:(test_data.shape[1])]

trainset = Loader(train_data, shuffle=True)
testset = Loader(test_data,  shuffle=True)

data = expr[cc_r]
loadeval = Loader(expr_eval[cc_r], shuffle=False)

# eset = expr
nminibatches = 3750
steps = 1


for steps in (range(0, 1)): ###Set range to (1,2) for 1 iteration
    st = time.time()
    saucie.train(load=trainset, steps=nminibatches, batch_size=256)
    print("----Training Step: " + str((nminibatches * (steps+1))) + " ------Runtime:" + str(time.time() - st))
    # st = time.time()
    print("lr=e-3 --- Train Loss: "+str(saucie.get_loss(trainset))+" Test Loss: "+str(saucie.get_loss(testset)))
    st = time.time()
    # number_of_clusters, clusters = saucie.get_clusters(loadeval)
    # print("----Generated clusters----" + str(((steps + 1) * nminibatches)) + " ------Runtime:" + str(time.time() - st))
    # # print("Loss (lr=1e-4): " + str(saucie.get_loss(loadtrain)))
#
# reconstruction = saucie.get_reconstruction(loadeval)
#
# reconstruction


st = time.time()

embedding = saucie.get_embedding(loadeval)
print("----Generated embedding----"+str(((steps+1)*nminibatches))+" ------Runtime:"+str(time.time() - st))

expr_eval['SAUCIE1'] = embedding[:, 0]
expr_eval['SAUCIE2'] = embedding[:, 1]

st = time.time()

number_of_clusters, clusters = saucie.get_clusters(loadeval)
print("----Generated clusters----"+str(((steps+1)*nminibatches))+" ------Runtime:"+str(time.time() - st))

expr_eval['saucie.clusters'] = pd.Categorical(clusters.astype('int'))
# expr = expr[expr['Batch'] == 1]

clust_name = 'saucie.clusters'

eset = expr_eval.copy()
eset[clust_name] = pd.Categorical(eset[clust_name]).codes

clustdf = pd.DataFrame(index=(np.unique(eset[clust_name])), columns=cc_r)

for j in np.unique(eset[clust_name]):
    clustdf.loc[j] = eset[eset[clust_name] == j][cc_r].median(axis=0).astype('float64')


sns.set_context("poster", font_scale=1.25)
cg = sns.clustermap(data=clustdf.astype('float64').T,
                    cmap=Vik_20.mpl_colormap,
                    z_score=0,
                    figsize=(15, 15),
                    # linewidths=0.75,
                    method='ward',
                    vmin=-2.5,
                    vmax=2.5,
                    cbar_kws={"ticks":[-2, -1, 0, 1, 2], "label":"z-score"})
# cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=12)
# cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=12)

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg.fig.subplots_adjust(right=.85, top=.90, bottom=.15)
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

plt.savefig('./paper.figs/fig8/cmap.cmis.'+str(lc)+'.'+str(ld)+'.cc_r.'+str((steps+1)*nminibatches)+'.png', dpi=400)
plt.close()


plt.figure(num=777,
          figsize=[14, 12])
ax = plt.subplot(111)
cmap = cm.get_cmap(Vivid_10.mpl_colormap, len(np.unique(expr_eval['saucie.clusters'])))
plt.scatter(x=expr_eval['SAUCIE1'],
            y=expr_eval['SAUCIE2'],
            s=1,
            c=pd.Categorical(expr_eval['saucie.clusters']).codes,
            cmap=cmap)
plt.xlabel("SAUCIE1")
plt.ylabel("SAUCIE2")
# plt.xlim([-3.25, 3.25])
# plt.ylim([-4,4])
plt.axis('tight')
plt.setp(ax.spines.values(), linewidth=2)
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

for j in np.unique(expr_eval['saucie.clusters']):
    plt.annotate(s=str(j),
                 xy=(expr_eval[expr_eval['saucie.clusters'] == j]['SAUCIE1'].median(),
                 expr_eval[expr_eval['saucie.clusters'] == j]['SAUCIE2'].median()),
                 size=25)

# plt.axis('off')
plt.title("Deterministic SAUCIE", fontsize=45, pad=15)
plt.xlabel("SAUCIE1", fontsize=45, labelpad=15)
plt.ylabel("SAUCIE2", fontsize=45, labelpad=15)
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    right=False,
    left=False,
    top=False,
    labelbottom=False,
    labelleft=False
)
plt.savefig('./paper.figs/fig8/cmis.'+str(lc)+'.'+str(ld)+'.cc_r.'+str((steps+1)*nminibatches)+'.png', dpi=400)
plt.close()
#
#
# expr_eval.to_csv('./paper.figs/fig8/d3.cytof-mes.cd.recon.clust.df.csv')
expr_eval = pd.read_csv('./paper.figs/fig8/d3.cytof-mes.cd.recon.clust.df.csv')
mc_recon_expr = expr_eval.copy()

sc_recon = mc_recon_expr[mc_recon_expr['Batch'] == 0].copy()
cc_recon = mc_recon_expr[mc_recon_expr['Batch'] == 1].copy()


major_clust = {'0': 'Homeostatic Endothelium',
               '1': 'Homeostatic Endothelium',
               '2': 'Homeostatic Endothelium',
               '5': 'Homeostatic Endothelium',
               '9': 'Homeostatic Endothelium',
               '13': 'Homeostatic Endothelium',
               '22': 'Homeostatic Endothelium',

               '4': 'Homeostatic Leukocytes',
               '14': 'Homeostatic Leukocytes',
               '18': 'Homeostatic Leukocytes',
               '21': 'Homeostatic Leukocytes',

               '3': 'Injury Leukocytes',
               '16': 'Injury Leukocytes',

               '7':  'Homeostatic Fibroblasts',
               '8':  'Homeostatic Fibroblasts',
               '10':  'Homeostatic Fibroblasts',

               '11':  'Injury Fibroblasts',

                '6': 'Pericytes',
               '12': 'Pericytes',
               '20': 'Pericytes',

               '15':  'Ambiguous Cells',
               '17':  'Ambiguous Cells',
               '19':  'Ambiguous Cells',
               }


saucie_anno = {0: 'Fibroblasts-1',
               1: 'Fibroblasts-1',
               2: 'Endothelium-1',
               3: 'Fibrocytes-1',
               4: 'Leukocytes-1',
               5: 'Leukocytes-1',
               6: 'Fibroblasts-1',
               7: 'Fibrocytes-2',
               8: 'Pericytes-1',
               9: 'Pericytes-1',
               10: 'Myofibroblasts-1',
               11: 'Myofibroblasts-1',
               12: 'Myofibroblasts-1',
               13: 'Myofibroblasts-1',
               14: 'Myofibroblasts-1'
               }



ct_adata = sc.read_h5ad('fig.7.h5ad')
ct_adata.obs['Major Clusters'] = ct_adata.obs['louvain'].map(major_clust)

sc_adata = sc.read_h5ad('adata_log1p.fig.2.3.h5ad')


sc_expr = sc_adata.to_df()
sc_expr = sc_expr.astype('float64', copy=True)

sc_expr['Batch'] = 0
sc_expr['Sample'] = 'scRNAseq'
sc_expr['IdU'] = sc_expr['Mki67'].values
sc_expr['Ly6AE'] = sc_expr['Ly6a'].values + sc_expr['Ly6e'].values

sc_expr = sc_expr[rna_mark]
sc_expr.columns = rna_mark

ct_expr = ct_adata.to_df()
ct_expr = ct_expr.astype('float64', copy=True)

ct_expr['Batch'] = 1
ct_expr['Sample'] = 'CyTOF'

sc_adata.obs['SAUCIE1'] = sc_recon['SAUCIE1'].values
sc_adata.obs['SAUCIE2'] = sc_recon['SAUCIE2'].values
sc_adata.obs['SAUCIE Clusters - scRNAseq'] = pd.Categorical(sc_recon['saucie.clusters'].values)
sc_adata.obs['SAUCIE Clusters - scRNAseq'] = pd.Categorical(sc_adata.obs['SAUCIE Clusters - scRNAseq'].map(saucie_anno))
sc_adata.obs['Major Clusters - scRNAseq'] = sc_adata.obs['Major Clusters'].copy()

# sc.pl.umap(sc_adata, color=['SAUCIE Clusters - scRNAseq'], save='.sc.'+str(nminibatches)+'.saucie.clust.png')
# sc.pl.umap(sc_adata, color=['Major Clusters'], save='.sc.'+str(nminibatches)+'.major.clust.png')

ct_adata.obs['SAUCIE1'] = cc_recon['SAUCIE1'].values
ct_adata.obs['SAUCIE2'] = cc_recon['SAUCIE2'].values
ct_adata.obs['SAUCIE Clusters - CyTOF'] = pd.Categorical(cc_recon['saucie.clusters'].values)
ct_adata.obs['SAUCIE Clusters - CyTOF'] = pd.Categorical(ct_adata.obs['SAUCIE Clusters - CyTOF'].map(saucie_anno))
ct_adata.obs['Major Clusters - CyTOF'] = ct_adata.obs['Major Clusters'].copy()

# sc_adata = sc_adata[sc_adata.obs['SAUCIE Clusters - scRNAseq'].isin([1, 2, 3, 4, 5, 7, 8, 9])]
# ct_adata = ct_adata[ct_adata.obs['SAUCIE Clusters - CyTOF'].isin([1, 2, 3, 4, 5, 7, 8, 9])]
ctsub_adata = ct_adata
scsub_adata = sc_adata

# sc.pl.umap(ctsub_adata, color=['SAUCIE Clusters - CyTOF'], save='.cc.'+str(nminibatches)+'.sub.saucie.clust.png')
# sc.pl.umap(ctsub_adata, color=['Major Clusters'], save='.cc.'+str(nminibatches)+'.sub.major.clust.png')

sc.pl.scatter(adata=scsub_adata, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters - scRNAseq',
              save='.scsub_adata.saucie.space.png', palette=sc.plotting.palettes.vega_20_scanpy)

sc.pl.scatter(adata=scsub_adata, x='SAUCIE1', y='SAUCIE2', color='Major Clusters - scRNAseq',
              save='.scsub_adata.mcs.saucie.space.png', palette=sc.plotting.palettes.vega_20_scanpy)

sc.pl.scatter(adata=ctsub_adata, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters - CyTOF',
              save='.ctsub_adata.saucie.space.png', palette=sc.plotting.palettes.vega_20_scanpy)

sc.pl.scatter(adata=ctsub_adata, x='SAUCIE1', y='SAUCIE2', color='Major Clusters - CyTOF',
              save='.ctsub_adata.mcs.saucie.space.png', palette=sc.plotting.palettes.vega_20_scanpy)

# sc.pl.dotplot(ct_adata, var_names=cytof_marks, groupby='SAUCIE Clusters - CyTOF', standard_scale='var', dendrogram=True, save='.saucie.cytof.marks.png')
sc.pl.dotplot(ctsub_adata, var_names=cytof_marks, groupby='SAUCIE Clusters - CyTOF', standard_scale='var', dendrogram=False, save='.sub.saucie.cytof.marks.png')

rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6a', 'Ly6e', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1']

sc.pl.dotplot(scsub_adata, var_names=rna_mark, groupby='SAUCIE Clusters - scRNAseq', standard_scale='var', dendrogram=False, save='.sub.saucie.rna.marks.png')

