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
sc.settings.set_figure_params(dpi=100, dpi_save=400)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_versions()
sc.settings.figdir='./paper.figs/fig6/'

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

# sc.pp.normalize_per_cell(adata)

adata_log1p = sc.pp.log1p(adata, copy=True)
adata_log1p.raw = adata.copy()

scrnaseq_sample = adata_log1p.obs['batch'].values.astype('U16')

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

sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True, stripplot=False,
             save='.scrnaseq.vio.post.qc.png')
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

rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6a', 'Ly6e', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1']


expr1 = adata_log1p.to_df()
expr1 = expr1.astype('float64', copy=True)
expr1['Batch'] = 0
expr1['Sample'] = 'scRNAseq'
expr1['IdU'] = expr1['Mki67'].values
expr1['Ly6AE'] = expr1['Ly6a'].values + expr1['Ly6e'].values

# mapping = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
#        'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn', 'IdU',
#        'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
#        'aSMA', 'CD31', 'VEGFR2', 'ActCaspase3', 'P53', 'MTCO1',
#        'HSP60', 'Ki67', 'pRb', 'CyclinB1', 'P21', 'bCatTotal',
#        'bCatActive', 'pHistoneH3', 'Sample', 'Batch']


mapping = ['CD45', 'CD11b', 'FolR2', 'IAIE', 'TNFa', 'CD140a',
       'Vimentin', 'CD90', 'Ly6C', 'Ly6AE', 'Postn',
       'IL6', 'CD9', 'VEGF', 'CD146', 'CD200', 'Notch3',
       'aSMA', 'CD31', 'VEGFR2', 'P53', 'MTCO1',
       'HSP60', 'Ki67', 'CyclinB1', 'P21', 'bCatTotal', 'IdU', 'Sample', 'Batch']

# rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
#            'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn', 'IdU',
#            'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
#            'Acta2', 'Pecam1', 'Kdr', 'Casp3', 'Trp53', 'mt-Co1',
#            'Hspd1', 'Mki67', 'Rb1', 'Ccnb1', 'Cdkn1a', 'Ctnnb1',
#            'Dkk3', 'Birc5', 'Sample', 'Batch']

rna_mark = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU', 'Sample', 'Batch']

# r_r = [rna_mark+'_r' for rna_mark in rna_mark]
expr3 = pd.read_csv("10.08.2019.gg.in.vitro.b4.csv", sep=',')

expr3_adata = ad.AnnData(expr3[mapping[:-2]])
expr3_adata.obs['Sample'] = expr3['Sample'].values
expr3_adata.obs['Batch'] = 1

expr3_adata.var_names_make_unique()
sc.pp.filter_cells(expr3_adata, min_genes=1)
# expr3_adata = expr3_adata[expr3_adata[:, 'CD31'].X < 1.44]

expr3_adata.obs['n_counts'] = expr3_adata.X.sum(axis=1)

# sc.pl.violin(expr3_adata, ['n_counts', 'n_genes'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              )

#
# sc.pp.filter_cells(expr3_adata, min_counts=30)
# sc.pp.filter_cells(expr3_adata, max_counts=60)
# sc.pp.filter_cells(expr3_adata, min_genes=15)
# sc.pp.filter_cells(expr3_adata, max_genes=25)

# sc.pl.violin(expr3_adata, ['n_genes', 'n_counts'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              save='.cytof.invivo.vio.post.qc.png')


re_expr1 = expr3_adata.to_df()
re_expr1['Sample'] = expr3_adata.obs['Sample']
re_expr1['Batch'] = expr3_adata.obs['Batch']
re_expr1 = re_expr1[mapping]
re_expr1.columns = rna_mark

#
# expr3 = pd.read_csv("10.08.2019.gg.fc.invitro.csv")
#
# expr3_adata = ad.AnnData(expr3[mapping[:-2]])
# expr3_adata.obs['Sample'] = expr3['Sample'].values
# expr3_adata.obs['Batch'] = 2
#
# expr3_adata.var_names_make_unique()
# sc.pp.filter_cells(expr3_adata, min_genes=1)
# #
# expr3_adata.obs['n_counts'] = expr3_adata.X.sum(axis=1)
# # sc.pl.violin(expr3_adata, ['n_counts', 'n_genes'],
# #              jitter=0.4, multi_panel=True, stripplot=False)
#
# sc.pl.violin(expr3_adata, ['n_genes', 'n_counts'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              save='.cytof.invitro.vio.pre.qc.png')
#
# sc.pp.filter_cells(expr3_adata, min_counts=50)
# sc.pp.filter_cells(expr3_adata, max_counts=85)
# sc.pp.filter_cells(expr3_adata, min_genes=20)
# sc.pp.filter_cells(expr3_adata, max_genes=32)
#
# sc.pl.violin(expr3_adata, ['n_genes', 'n_counts'],
#              jitter=0.4, multi_panel=True, stripplot=False,
#              save='.cytof.invitro.vio.post.qc.png')
# #
# # sc.pl.violin(expr3_adata, ['n_counts', 'n_genes'],
# #              jitter=0.4, multi_panel=True, stripplot=False)
#
# re_expr3 = expr3_adata.to_df()
# re_expr3['Sample'] = expr3_adata.obs['Sample']
# re_expr3['Batch'] = expr3_adata.obs['Batch']
# re_expr3.columns = rna_mark

expr = pd.concat([expr1[rna_mark].sample(replace=True, n=re_expr1.shape[0]),
                  re_expr1[rna_mark],
                  # re_expr3[rna_mark].sample(replace=True, n=re_expr1.shape[0]),
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

expr = pd.concat([expr1[rna_mark],
                  re_expr1[rna_mark],
                  # expr4[rna_mark]
                  ], sort=False)

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

lb = 0.10

saucie = SAUCIE(data.shape[1],
                    layers=[l1, l2, l3, l4],
                    learning_rate=0.001,
                    lambda_b=lb,
                    # limit_gpu_fraction=0.20,
                    no_gpu=True
                    )


for steps in (range(0, 1)): ###Set range to (1,2) for 1 iteration

    st = time.time()
    saucie.train(load=trainset, steps=nminibatches, batch_size=256)
    print("----Training Step: " + str((nminibatches)) + " ------Runtime:" + str(time.time() - st))
    print("--- Train Loss: "+str(saucie.get_loss(trainset))+" Test Loss: "+str(saucie.get_loss(testset)))


    reconstruction = saucie.get_reconstruction(loadtrain)
    expr_train = pd.DataFrame(reconstruction[0], columns=cc_r)
    expr_train.to_csv('./paper.figs/fig6/diss.saucie-cd.trainset.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')

    reconstruction = saucie.get_reconstruction(loadeval)

    expr_bc = pd.DataFrame(reconstruction[0], columns=cc_r)

    embedding = saucie.get_embedding(loadeval)

    expr_bc['SAUCIE1'] = embedding[0][:, 0]
    expr_bc['SAUCIE2'] = embedding[0][:, 1]
    expr_bc['Batch'] = expr['Batch'].values
    expr_bc['Sample'] = expr['Sample'].values

    expr_bc.to_csv('./paper.figs/fig6/diss.cytof-mes.equal.train.'+str(nminibatches)+'.'+str(lb)+'.'+str(steps)+'.csv')

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
    plt.savefig('./paper.figs/fig6/gg_matrix_batch.'+str(lb)+'.'+str(nminibatches)+'.'+str(steps)+".png",
                dpi=400)
    plt.close()

    eset = expr_bc[expr_bc['Batch'] == 0].copy()
    plt.figure(num=777,
               figsize=[14, 12])
    ax = plt.subplot(111)
    # cmap = cm.get_cmap(Vivid_10.mpl_colormap, len(np.unique(expr_bc['Batch'])))
    plt.scatter(x=eset['SAUCIE1'],
                y=eset['SAUCIE2'],
                s=1,
                marker=',',
                # alpha=0.1,
                # c=pd.Categorical(expr_bc['Batch']).codes,
                # cmap=cmap
                )
    plt.xlabel("SAUCIE1")
    plt.ylabel("SAUCIE2")
    # plt.xlim([-3.25, 3.25])
    # plt.ylim([-4,4])
    plt.axis('tight')
    plt.setp(ax.spines.values(), linewidth=2)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    #
    # for j in np.unique(expr_bc['Batch']):
    #     plt.annotate(s=str(j),
    #                  xy=(expr_bc[expr_bc['Batch'] == j]['SAUCIE1'].median(),
    #                      expr_bc[expr_bc['Batch'] == j]['SAUCIE2'].median()),
    #                  size=15)

    # plt.axis('off')
    plt.title("scRNAseq Distribution SAUCIE-B", fontsize=45, pad=15)
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
    plt.savefig('./paper.figs/fig6/scrnaseq.dist.'+str(lb)+'.'+str(nminibatches)+'.'+str(steps)+".png",
                dpi=400)
    plt.close()

    fname = './paper.figs/fig6/feat.'
    for i in ['Ptprc_r', 'Notch3_r', 'Pdgfra_r', 'Vim_r']:
        plt.figure(num=777, figsize=[12, 9])
        plt.scatter(x=expr_bc['SAUCIE1'],
                    y=expr_bc['SAUCIE2'],
                    c=expr_bc[i].astype('float64'),
                    marker=",",
                    s=0.1,
                    cmap=Bilbao_20.mpl_colormap)
        # plt.scatter(x=expr['SAUCIE1'],
        #             y=expr['SAUCIE2'],
        #             c=expr[i].astype('float64'),
        #             marker=",",
        #             s=0.1,
        #             cmap=Bilbao_20.mpl_colormap)
        plt.axis('tight')
        plt.axis('off')
        plt.title(str(i), fontsize=50)
        # plt.xlabel(str(i), fontsize=15, fontweight='bold')
        # plt.ylabel("Pecam1_r", fontsize=15, fontweight='bold')
        plt.tick_params(
            axis='both',
            which='both',
            bottom=True,
            right=False,
            left=True,
            top=False,
            labelbottom=True,
            labelleft=True
        )
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=50)
        # tick_locator = ticker.MaxNLocator(nbins=3)
        # cb.locator = tick_locator
        cb.update_ticks()
        # plt.savefig(fname+str(i)+str(lb)+'.'+str(nminibatches)+'.'+str(steps)+".cbar.png", dpi=400)
        plt.savefig(fname + i + ".stable.params.cbar.png", dpi=400)
        plt.close()



expr_eval = pd.read_csv('./paper.figs/fig6/diss.cytof-mes.equal.train.1000.0.1.0.csv')
expr = pd.read_csv('./paper.figs/fig6/diss.saucie-cd.trainset.1000.0.1.0.csv')
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

lc = 0.10
ld = 0.20
layer_c = 0
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
nminibatches = 1500
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

expr_eval['saucie.clusters'] = clusters.astype('int')
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

plt.savefig('./paper.figs/fig6/cmap.cmis.'+str(lc)+'.'+str(ld)+'.cc_r.'+str((steps+1)*nminibatches)+'.png', dpi=400)
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
plt.savefig('./paper.figs/fig6/cmis.'+str(lc)+'.'+str(ld)+'.cc_r.'+str((steps+1)*nminibatches)+'.png', dpi=400)
plt.close()


expr_eval.to_csv('./paper.figs/fig6/diss.ch2.sc.cytof-mes.cd.recon.clust.df.csv')

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

rna_marks1 = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6a', 'Ly6e', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1']


clustering_channels = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6AE', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1', 'IdU']

cc = ['Ptprc', 'Itgam', 'Folr2', 'H2-Ab1', 'Tnf', 'Pdgfra',
           'Vim', 'Thy1', 'Ly6c1', 'Ly6a', 'Ly6e', 'Postn',
           'Il6', 'Cd9', 'Vegfa', 'Mcam', 'Cd200', 'Notch3',
           'Acta2', 'Pecam1', 'Kdr', 'Trp53', 'mt-Co1',
           'Hspd1', 'Mki67',  'Ccnb1', 'Cdkn1a', 'Ctnnb1']

cc_r = [clustering_channels + "_r" for clustering_channels in clustering_channels]

mc_recon_expr = pd.read_csv('./paper.figs/fig6/diss.ch2.sc.cytof-mes.cd.recon.clust.df.csv')


sc_recon = mc_recon_expr[mc_recon_expr['Batch'] == 0].copy()
cc_recon = mc_recon_expr[mc_recon_expr['Batch'] == 1].copy()

expr3 = pd.read_csv("10.08.2019.gg.in.vitro.b4.csv", sep=',')

expr3_adata = ad.AnnData(expr3[mapping[:-2]])
expr3_adata.obs['Sample'] = expr3['Sample'].values
expr3_adata.obs['Batch'] = 1

expr3_adata.obs['n_counts'] = expr3_adata.X.sum(axis=1)

expr3_adata.var_names_make_unique()
sc.pp.filter_cells(expr3_adata, min_genes=1)

expr3_adata.obs['SAUCIE Clusters'] = pd.Categorical(cc_recon['saucie.clusters'])
expr3_adata.obs['SAUCIE1'] = cc_recon['SAUCIE1'].values
expr3_adata.obs['SAUCIE2'] = cc_recon['SAUCIE2'].values

adata_log1p.obs['SAUCIE Clusters'] = pd.Categorical(sc_recon['saucie.clusters'])
adata_log1p.obs['SAUCIE1'] = sc_recon['SAUCIE1'].values
adata_log1p.obs['SAUCIE2'] = sc_recon['SAUCIE2'].values

for j in cc_r:
    expr3_adata.obs[j] = cc_recon[j].values
    adata_log1p.obs[j] = sc_recon[j].values

sc.pl.scatter(adata=adata_log1p, x='SAUCIE1', y='SAUCIE2', color='Pecam1')
sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color='CD31')

from sklearn.linear_model import LinearRegression

expr = pd.read_csv("./paper.figs/fig6/diss.cytof-mes.equal.train.1000.0.1.0.csv")
expr = expr.drop('Unnamed: 0',  axis=1)



model_y = pd.DataFrame(data=adata_log1p[:, rna_marks1].X, columns=rna_marks1)


gosc_expr = expr[expr['Sample'] == 'scRNAseq']
gosc_expr = gosc_expr[cc_r]

for i in model_y.columns:
    X = gosc_expr[cc_r]
    # y = gosc_expr[gosc_expr.columns[len(cc_r):]]
    y = model_y[i]

    clf = LinearRegression(fit_intercept=True)
    clf.fit(X, y)

    ont_score = pd.Series(clf.predict(expr[cc_r]))

    expr[i] = ont_score.copy()
    print(str(i) + "\n")

for j in cc:
    expr3_adata.obs[j] = expr[expr['Sample']!='scRNAseq'][j].values

from scipy.stats import pearsonr

corr_mat = pd.DataFrame(columns=rna_marks1, index=cytof_marks)

for i in rna_marks1:
    for j in cytof_marks:
        corr_mat[i].loc[j] = (pearsonr(x=expr3_adata[:, j].X, y=expr3_adata.obs[i]))[0]



sc.pl.scatter(adata=adata_log1p, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters',
              save='.vitro.saucie.scrnaseq.no.anno.png', palette=sc.plotting.palettes.vega_20_scanpy)

sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters',
              save='.vitro.saucie.cytof.no.anno.png', palette=sc.plotting.palettes.vega_20_scanpy)

# sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color='IdU')

sum(adata_log1p.obs['SAUCIE Clusters'].isin([0]))/42843##Fibro
sum(adata_log1p.obs['SAUCIE Clusters'].isin([1]))/42843##Endo-like
sum(adata_log1p.obs['SAUCIE Clusters'].isin([2, 3]))/42843##Leuko


saucie_sc = {0: 'Fibroblasts 64.0%',
             1: 'Endothelium 13.1%',
             2: 'Leukocytes 22.9%',
             3: 'Leukocytes 22.9%',
            }

adata_log1p.obs['SAUCIE Clusters - scRNAseq'] = adata_log1p.obs['SAUCIE Clusters'].map(saucie_sc)

sum(expr3_adata.obs['SAUCIE Clusters'].isin([0]))/168459##Fibro
sum(expr3_adata.obs['SAUCIE Clusters'].isin([1]))/168459##Endo-like
sum(expr3_adata.obs['SAUCIE Clusters'].isin([2,3]))/168459##Leuko
# sum(expr3_adata.obs['SAUCIE Clusters'].isin([1, 6, 8, 9, 10]))/369405##Homeo Leuko

saucie_cc = {0: 'Fibroblasts 52.6%',
             1: 'Endothelium 6.36%',
             2: 'Leukocytes 41.1%',
             3: 'Leukocytes 41.1%',
            }

expr3_adata.obs['SAUCIE Clusters - CytOF'] = expr3_adata.obs['SAUCIE Clusters'].map(saucie_cc)

sc.pl.scatter(adata=adata_log1p, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters - scRNAseq', save='.vivo.saucie.mc.sc.png', palette=sc.plotting.palettes.vega_20_scanpy)
sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color='SAUCIE Clusters - CytOF', save='.vivo.saucie.mc.cc.png', palette=sc.plotting.palettes.vega_20_scanpy)
sc.pl.scatter(adata=adata_log1p, x='SAUCIE1', y='SAUCIE2', color='Postn', save='.vivo.saucie.postn.sc.png')
sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color='Sample', save='.vivo.saucie.samples.cc.png', palette=sc.plotting.palettes.vega_20_scanpy)


for i in ['n_counts', 'n_genes', 'SAUCIE Clusters - scRNAseq']:
    sc.pl.scatter(adata=adata_log1p, x='SAUCIE1', y='SAUCIE2', color=i, save='.vivo.saucie.scrnaseq.'+str(i)+'.png')

for i in ['n_counts', 'n_genes', 'SAUCIE Clusters - CytOF']:
    sc.pl.scatter(adata=expr3_adata, x='SAUCIE1', y='SAUCIE2', color=i, save='.vivo.saucie.cytof.'+str(i)+'.png')

sc.plotting.dotplot(adata_log1p, groupby='SAUCIE Clusters - scRNAseq', var_names=rna_marks1, standard_scale='var', save='.vivo.mcs.scrnaseq.png')
sc.plotting.dotplot(expr3_adata, groupby='SAUCIE Clusters - CytOF', var_names=cytof_marks, standard_scale='var', save='.vivo.mcs.cytof.png')

for i in np.unique(expr3_adata.obs['Sample']):
    sc.pl.scatter(expr3_adata[expr3_adata.obs['Sample']==i], x='SAUCIE1', y='SAUCIE2', color='IdU', save='.vivo.saucie.'+str(i)+'.png')

import seaborn as sns

sns.set_context('paper', font_scale=1.5)
##Plot 1
cg = sns.clustermap(data=corr_mat.astype('float64'),
               cmap=Vik_20.mpl_colormap,
               figsize=(25, 15),
               method='ward',
               # row_cluster=False,
               # col_cluster=False,
               annot=corr_mat.astype('float64').round(1),
               vmin=-1.0,
               vmax=1.0,
               cbar_kws={"ticks":[-1, -0.5, 0, 0.5, 1], "label":"Pearson R"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
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

plt.savefig('./paper.figs/fig6/corr_mat_cytof_vs_xcript.png', dpi=400)
plt.close()

corr_mat = corr_mat.loc[corr_mat.index[cg.dendrogram_row.reordered_ind]]
corr_mat = corr_mat[corr_mat.columns[cg.dendrogram_col.reordered_ind]]


cg = sns.clustermap(data=corr_mat.astype('float64'),
               cmap=Vik_20.mpl_colormap,
               figsize=(25, 15),
               method='ward',
               # row_cluster=False,
               # col_cluster=False,
               annot=corr_mat.astype('float64').round(1),
               vmin=-1.0,
               vmax=1.0,
               cbar_kws={"ticks":[-1, -0.5, 0, 0.5, 1], "label":"Pearson R"})

plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
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

plt.savefig('./paper.figs/fig6/clustered_corr_mat_cytof_vs_xcript.png', dpi=400)
plt.close()
