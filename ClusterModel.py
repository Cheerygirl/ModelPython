#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import pandas as pd
import numpy as np
import datetime
from time import time
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from scipy.spatial import distance

##########Get Index###############
ImprotanceIndex = pd.read_csv('RelationIndex.csv',encoding='gbk',index_col=0)

##########Index Prepare###########
ChainNum = len(ImprotanceIndex)
NanNum = ImprotanceIndex.isnull().sum()
Nanpct = NanNum / ChainNum
NanIndex = pd.DataFrame({'NanNum':NanNum,'Nanpct':Nanpct})
##check Nanpct > 0 ###
print NanIndex[NanIndex.Nanpct>0]

############
Amtlist = ['BillRctAmt1yr','BillRctAmtHfyr','BillRctAmtQt']
ImprotanceIndex[Amtlist] = ImprotanceIndex[Amtlist].fillna(0)

###数据过滤###
ImprotanceIndex.drop(['BillChainFirstDate','BillChainLastDate'],axis=1,inplace=True)

####只下过一次单的企业####
ImprotanceIndex = ImprotanceIndex[ImprotanceIndex['BillNum']>1]
ImprotanceIndex = ImprotanceIndex[ImprotanceIndex['BillChainDay']>1]

bp2list =['BillChainStdGap','BillChainStdGapU','BillKurtamt','BillSkewamt']
ImprotanceIndex[bp2list] = ImprotanceIndex[bp2list].fillna(0)

##########Model Index############
var_list = [u'Bill25amt', u'Bill75amt', u'BillAmt', u'BillAvgamt', u'BillCvamt',
            u'BillIQRamt', u'BillKurtamt', u'BillMaxamt', u'BillMedianamt',
            u'BillMinamt', u'BillNum', u'BillRctAmt1yr', u'BillRctAmtHfyr',
            u'BillRctAmtQt', u'BillSig1', u'BillSig2', u'BillSig3', u'BillSkewamt',u'BillStdamt']

ModelIndex = ImprotanceIndex[var_list]
ModelIndexCorr = ModelIndex.corr()
ModelIndexCorr.to_csv('ModelIndexCorr.csv',encoding='utf8')

for i in ModelIndex:
    fig = sns.distplot(ModelIndex[i]).get_figure()
    fig.savefig(i)
    fig.clear()

##################
model_var = [ u'BillAmt',  u'BillCvamt',u'BillIQRamt', u'BillKurtamt', u'BillMaxamt', u'BillMedianamt',
            u'BillMinamt', u'BillNum', u'BillSig1', u'BillSig2', u'BillSig3', u'BillSkewamt']
ModelIndex = ModelIndex[model_var]
ModelIndex.to_csv('ModelIndex.csv',encoding='gbk')

####数据标准化####
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelIndex)
RbTrainIndex = rbscale.transform(ModelIndex)
RbIndex = pd.DataFrame(RbTrainIndex,columns=ModelIndex.columns,index=ModelIndex.index)

'''
qtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ModelIndex)
QtnTrainIndex = qtnscale.transform(ModelIndex)
QtnIndex = pd.DataFrame(QtnTrainIndex,columns=ModelIndex.columns,index=ModelIndex.index)

qtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ModelIndex)
QtuTrainIndex = qtuscale.transform(ModelIndex)
QtuIndex = pd.DataFrame(QtuTrainIndex,columns=ModelIndex.columns,index=ModelIndex.index)
'''

#####样本距离分析####
''''
euclid = distance.pdist(RbTrainIndex,metric= 'euclidean')
euclid_tran = distance.squareform(euclid)

manhat = distance.pdist(RbTrainIndex,metric= 'cityblock')
manhat_tran = distance.squareform(manhat)

stdeuc = distance.pdist(RbTrainIndex,metric= 'seuclidean', V=None)
stdeuc_tran = distance.squareform(stdeuc)

sqseuc = distance.pdist(RbTrainIndex,metric= 'sqeuclidean')
sqseuc_tran = distance.squareform(sqseuc)

cosine = distance.pdist(RbTrainIndex,metric= 'cosine')
cosine_tran = distance.squareform(cosine)

correl = distance.pdist(RbTrainIndex,metric= 'correlation')
correl_tran = distance.squareform(correl)

hammig = distance.pdist(RbTrainIndex,metric= 'hamming')
hammig_tran = distance.squareform(hammig)

jaccad = distance.pdist(RbTrainIndex,metric= 'jaccard')
jaccad_tran = distance.squareform(jaccad)

chebys = distance.pdist(RbTrainIndex,metric= 'chebyshev')
chebys_tran = distance.squareform(chebys)

canber = distance.pdist(RbTrainIndex,metric= 'canberra')
canber_tran = distance.squareform(canber)

braycu = distance.pdist(RbTrainIndex,metric= 'braycurtis')
braycu_tran = distance.squareform(braycu)
'''
mahala = distance.pdist(RbTrainIndex,metric='mahalanobis', VI=None)
mahala_tran = distance.squareform(mahala)

mahala_tran = pd.DataFrame(mahala_tran,columns=RbIndex.index,index=RbIndex.index)
mahala_tran.to_csv('Mahala.csv',encoding='utf8')

Nearest = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6','7','8','9','10','chain_name'])
for i in mahala_tran.index:
    chain_nearest = mahala_tran.nsmallest(11,i,'first').index.tolist()
    chain_nearest.append(i)
    Nearest = Nearest.append(pd.Series(chain_nearest, index=['0', '1', '2', '3', '4', '5', '6','7','8','9','10',
                                                             'chain_name']),ignore_index=True)
Nearest.set_index('chain_name')
Nearest.to_csv('Nearest.csv',encoding='gbk')

#######成分分析########
# PCA on the fruit dataset (for comparison)
pca = PCA(n_components = 6,whiten=True).fit(RbTrainIndex)
RbIndex_pca = pca.transform(RbTrainIndex)
RbIndex_pcacom = pca.components_
RbIndex_pcaexr = pca.explained_variance_ratio_
RbIndex_pca_df = pd.DataFrame(RbIndex_pca,index=ModelIndex.index)
RbIndex_pcacom_df = pd.DataFrame(RbIndex_pcacom,columns=ModelIndex.columns)
RbIndex_pcaexr_df = pd.DataFrame(RbIndex_pcaexr)
RbIndex_pca_df.to_csv('RbIndex_pca.csv',encoding='utf8')
RbIndex_pcacom_df.to_csv('RbIndex_pcacom.csv',encoding='utf8')
RbIndex_pcaexr_df.to_csv('RbIndex_pcaexr.csv',encoding='utf8')

# FA
fa = FactorAnalysis(n_components=6).fit(RbTrainIndex)
RbIndex_fa = fa.transform(RbIndex)
RbIndex_facom = fa.components_
RbIndex_fanv = fa.noise_variance_
RbIndex_fa_df = pd.DataFrame(RbIndex_fa,index=ModelIndex.index)
RbIndex_facom_df = pd.DataFrame(RbIndex_facom,columns=ModelIndex.columns)
RbIndex_fanv_df = pd.DataFrame(RbIndex_fanv,index=ModelIndex.columns)
RbIndex_fa_df.to_csv('RbIndex_fa.csv',encoding='utf8')
RbIndex_facom_df.to_csv('RbIndex_facom.csv',encoding='utf8')
RbIndex_fanv_df.to_csv('RbIndex_fanv.csv',encoding='utf8')

'''
# Multidimensional scaling (MDS) on the fruit dataset
mds = MDS(n_components = 2)
RbIndex_mds = mds.fit_transform(RbTrainIndex)
RbIndex_mds_df = pd.DataFrame(RbIndex_mds,index=ModelIndex.index)
RbIndex_mds_df.to_csv('RbIndex_mds.csv',encoding='utf8')

# t-SNE on the fruit dataset
tsne = TSNE(random_state = 0)
RbIndex_tsne = tsne.fit_transform(RbTrainIndex)
RbIndex_tsne_df = pd.DataFrame(RbIndex_tsne,index=ModelIndex.index)
RbIndex_tsne_df.to_csv('RbIndex_tsne.csv',encoding='utf8')
'''

##################################聚类##################################
###K-Means###
n_samples,n_features = RbTrainIndex.shape
digits = 12
#labels = TrainIndex.labels
sample_size = n_samples

def bench_k_means(estimator,name,data):
    t0 = time()
    estimator.fit(data)
    parameter = [name,estimator.inertia_,(time()-t0),
                 metrics.silhouette_score(data, estimator.labels_, metric='sqeuclidean',sample_size=sample_size),
                 metrics.calinski_harabaz_score(data, estimator.labels_)]
    return parameter
#    print('Name: %-9s' % name)
#    print('Inertia: %i' % estimator.inertia_)
#    print('Time: %.2fs' % (time() - t0))
#    print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(data, estimator.labels_, metric='sqeuclidean',sample_size=sample_size))
#    print("Calinski-Harabaz Index: %0.3f"
#          % metrics.calinski_harabaz_score(data, estimator.labels_))
#    print(82 * '_')

KMeansScore = pd.DataFrame(columns=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters'])
for i in range(3,digits,1):
    n_digits = i
#    print("n_digits: %d, \t n_samples %d, \t n_features %d"  % (n_digits, n_samples, n_features))
    prRandom = bench_k_means(KMeans(init='random',n_clusters=i,n_init=20),name='Random',data=RbTrainIndex)
    prRandom.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prRandom, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prKplus = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='k-means++',data=RbTrainIndex)
    prKplus.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prKplus, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    pca = PCA(n_components=i,whiten=True).fit(RbTrainIndex)
    prPCA = bench_k_means(KMeans(init=pca.components_,n_clusters=i,n_init=1),name='PCA-based',data=RbTrainIndex)
    prPCA.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prPCA, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReducedTrainIndex = PCA(n_components=5,whiten=True).fit_transform(RbTrainIndex)
    prReDuc = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='ReDuc_PCA',data=ReducedTrainIndex)
    prReDuc.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prReDuc, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedTrainIndex)
    ReQtuTrainIndex = ReQtuscale.transform(ReducedTrainIndex)
    prReQtu = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='ReQtu_PCA',data=ReQtuTrainIndex)
    prReQtu.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prReQtu, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReQtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ReducedTrainIndex)
    ReQtnTrainIndex = ReQtnscale.transform(ReducedTrainIndex)
    prReQtn = bench_k_means(KMeans(init='k-means++', n_clusters=i, n_init=20), name='ReQtn_PCA', data=ReQtnTrainIndex)
    prReQtn.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prReQtn, index=['name', 'SSE', 'time', 'silhouette_score', 'calinski_harabaz_score', 'n_clusters']),
        ignore_index=True)

KMeansScore.to_csv('KMeansScore.csv',encoding='utf8')

##affinity propagation clustering algorithm##
def bench_af(estimator,name,data):
    af = estimator.fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    parameter = [name,n_clusters_,metrics.silhouette_score(data, labels, metric='sqeuclidean'),
                 metrics.calinski_harabaz_score(data, labels)]
    return parameter
#    print ('Name: %-9s' % name)
#    print('Estimated number of clusters: %d' % n_clusters_)
#    print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(data, labels, metric='sqeuclidean'))
#    print("Calinski-Harabaz Index: %0.3f"
#          % metrics.calinski_harabaz_score(data, labels))
#    print(82 * '_')

AffScore = pd.DataFrame(columns=['name','n_clusters','silhouette_score','calinski_harabaz_score'])

prAff = bench_af(AffinityPropagation(),name='Index',data=RbTrainIndex)
AffScore = AffScore.append(
    pd.Series(prAff, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReducedTrainIndex = PCA(n_components=5, whiten=True).fit_transform(RbTrainIndex)
prAfRed = bench_af(AffinityPropagation(),name='PCA_Red',data=ReducedTrainIndex)
AffScore = AffScore.append(
    pd.Series(prAfRed, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedTrainIndex)
ReQtuTrainIndex = ReQtuscale.transform(ReducedTrainIndex)
prAfQtu = bench_af(AffinityPropagation(),name='PCA_Qtu',data=ReQtuTrainIndex)
AffScore = AffScore.append(
    pd.Series(prAfQtu, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReQtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ReducedTrainIndex)
ReQtnTrainIndex = ReQtnscale.transform(ReducedTrainIndex)
prAfQtn = bench_af(AffinityPropagation(),name='PCA_Qtn',data=ReQtnTrainIndex)
AffScore = AffScore.append(
    pd.Series(prAfQtn, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

AffScore.to_csv('AffScore.csv',encoding='utf8')

###Hierarchical clustering####
def bench_agg(estimator,name,data):
    agg = estimator.fit(data)
    labels = agg.labels_
    leaves = agg.n_leaves_
    parameter = [name,leaves,metrics.silhouette_score(data, labels, metric='sqeuclidean'),
                 metrics.calinski_harabaz_score(data, labels)]
    return parameter
#    print ('Name: %-9s' % name)
#    print('Number of leaves: %d' % leaves )
#    print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(data, labels, metric='sqeuclidean'))
#    print("Calinski-Harabaz Index: %0.3f"
#          % metrics.calinski_harabaz_score(data, labels))
#    print(82 * '_')

HierarScore = pd.DataFrame(columns=['name','leaves','silhouette_score','calinski_harabaz_score','n_cluster'])
for i in range(3,digits,1):
    n_digits = i
#    print("n_digits: %d, \t n_samples %d, \t n_features %d"  % (n_digits, n_samples, n_features))
    prWardAgg = bench_agg(AgglomerativeClustering(linkage='ward',n_clusters=i),name='WardAgg',data=RbTrainIndex)
    prWardAgg.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prWardAgg, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prAvgAgg = bench_agg(AgglomerativeClustering(linkage='average',n_clusters=i),name='AvgAgg',data=RbTrainIndex)
    prAvgAgg.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prAvgAgg, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prComlAvg = bench_agg(AgglomerativeClustering(linkage='complete', n_clusters=i), name='ComlAgg', data=RbTrainIndex)
    prComlAvg.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prComlAvg, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReducedTrainIndex = PCA(n_components=5,whiten=True).fit_transform(RbTrainIndex)
    prWardRed = bench_agg(AgglomerativeClustering(linkage='ward',n_clusters=i),name='WardRed',data=ReducedTrainIndex)
    prWardRed.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prWardRed, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prAvgRed = bench_agg(AgglomerativeClustering(linkage='average',n_clusters=i),name='AvgRed',data=ReducedTrainIndex)
    prAvgRed.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prAvgRed, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prComlRed = bench_agg(AgglomerativeClustering(linkage='complete', n_clusters=i), name='ComlRed', data=ReducedTrainIndex)
    prComlRed.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prComlRed, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedTrainIndex)
    ReQtuTrainIndex = ReQtuscale.transform(ReducedTrainIndex)
    prWardQtu = bench_agg(AgglomerativeClustering(linkage='ward',n_clusters=i),name='WardQtu',data=ReQtuTrainIndex)
    prWardQtu.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prWardQtu, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prAvgQtu = bench_agg(AgglomerativeClustering(linkage='average',n_clusters=i),name='AvgQtu',data=ReQtuTrainIndex)
    prAvgQtu.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prAvgQtu, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prComlQtu = bench_agg(AgglomerativeClustering(linkage='complete', n_clusters=i), name='ComlQtu', data=ReQtuTrainIndex)
    prComlQtu.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prComlQtu, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReQtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ReducedTrainIndex)
    ReQtnTrainIndex = ReQtnscale.transform(ReducedTrainIndex)
    prWardQtn = bench_agg(AgglomerativeClustering(linkage='ward',n_clusters=i),name='WardQtn',data=ReQtnTrainIndex)
    prWardQtn.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prWardQtn, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prAvgQtn = bench_agg(AgglomerativeClustering(linkage='average',n_clusters=i),name='AvgQtn',data=ReQtnTrainIndex)
    prAvgQtn.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prAvgQtn, index=['name','leaves','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)
    prComlQtn = bench_agg(AgglomerativeClustering(linkage='complete', n_clusters=i), name='ComlQtn', data=ReQtnTrainIndex)
    prComlQtn.append(n_digits)
    HierarScore = HierarScore.append(
        pd.Series(prComlQtn, index=['name', 'leaves', 'silhouette_score', 'calinski_harabaz_score', 'n_clusters']),
        ignore_index=True)

HierarScore.to_csv('HierarScore.csv',encoding='utf8')

#plt.figure()
#dendrogram(ward(RbTrainIndex))
#plt.show()

'''
###DBSCAN,维度数最好不要超过6个####
def bench_dbs(estimator,name,data):
    dbs = estimator.fit(data)
    labels = dbs.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    parameter = [name,n_clusters_,metrics.silhouette_score(data, labels),
                 metrics.calinski_harabaz_score(data, labels)]
    return parameter
#    print ('Name: %-9s' % name)
#    print('Estimated number of cluster: %d' % n_clusters_ )
#    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
#    print("Calinski-Harabaz Index: %0.3f" % metrics.calinski_harabaz_score(data, labels))
#    print(82 * '_')
DBSCANScore = pd.DataFrame(columns=['name','n_cluster','silhouette_score','calinski_harabaz_score'])

prDBSCAN = bench_dbs(DBSCAN(eps=1e-2, min_samples=10),name='Index',data=RbTrainIndex)
DBSCANScore = DBSCANScore.append(
    pd.Series(prDBSCAN, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReducedTrainIndex = PCA(n_components=5, whiten=True).fit_transform(RbTrainIndex)
prDBSCANRed = bench_dbs(DBSCAN(eps=1e-3, min_samples=10),name='PCA_Red',data=ReducedTrainIndex)
DBSCANScore = DBSCANScore.append(
    pd.Series(prDBSCANRed, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedTrainIndex)
ReQtuTrainIndex = ReQtuscale.transform(ReducedTrainIndex)
prDBSCANQtu = bench_dbs(DBSCAN(eps=1e-2, min_samples=10),name='PCA_Qtu',data=ReQtuTrainIndex)
DBSCANScore = DBSCANScore.append(
    pd.Series(prDBSCANQtu, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

ReQtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ReducedTrainIndex)
ReQtnTrainIndex = ReQtnscale.transform(ReducedTrainIndex)
prDBSCANQtn = bench_dbs(DBSCAN(eps=1e-3, min_samples=100),name='PCA_Qtn',data=ReQtnTrainIndex)
DBSCANScore = DBSCANScore.append(
    pd.Series(prDBSCANQtn, index=['name','n_clusters','silhouette_score','calinski_harabaz_score']),
    ignore_index=True)

DBSCANScore.to_csv('DBSCANScore.csv',encoding='utf8')
'''