#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import pandas as pd
import numpy as np
from time import time
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import distance

##########Get Index###############
Index = pd.read_csv('Index.csv',encoding='gbk',index_col=0)
Y = pd.read_csv('ModelY.csv',index_col=0)

Unitype = list(enumerate(np.unique(Y.com_index)))
UnitypeDict = {name: i for i,name in Unitype}
Y.com_index = Y.com_index.map(lambda x:UnitypeDict[x]).astype(int)
IndexY = pd.concat([Index,Y],axis=1)

##########Index Prepare###########
FeatureList = [ u'Chain25PayDay',u'Chain50PayDay', u'Chain75PayDay',u'ChainAmtPayDay', u'ChainAvgPayDay',
                   u'ChainCvPayDay', u'ChainIQRPayDay', u'ChainKurtPayDay', u'ChainPayDay1sig',
                   u'ChainPayDay2sig', u'ChainPayDay3sig',u'ChainSkewPayDay',u'ChainCrAmt', u'ChainCrNum',
                   u'ChainMaxPayDay',u'ChainMinPayDay',u'ChainNumPayDay',u'ChainStdPayDay',
                   u'PayDayAvgMonth_case_dur', u'PayDayAvgMonth_ts_len', u'PayDayAvgMonth_tsd_len',
                   u'PayDayAvgMonth_ts_pct', u'PayDayAvgMonth_tsd_pct', u'PayDayAvgMonth_tsdts_pct',
                   u'PayDayAvgWeek_case_dur', u'PayDayAvgWeek_ts_len', u'PayDayAvgWeek_tsd_len',
                   u'PayDayAvgWeek_ts_pct', u'PayDayAvgWeek_tsd_pct', u'PayDayAvgWeek_tsdts_pct',
                   u'PayDayAvgQuater_case_dur', u'PayDayAvgQuater_ts_len', u'PayDayAvgQuater_tsd_len',
                   u'PayDayAvgQuater_ts_pct', u'PayDayAvgQuater_tsd_pct', u'PayDayAvgQuater_tsdts_pct',
                   u'PayDayAvgYr_case_dur', u'PayDayAvgYr_ts_len', u'PayDayAvgYr_tsd_len',
                   u'PayDayAvgYr_ts_pct', u'PayDayAvgYr_tsd_pct', u'PayDayAvgYr_tsdts_pct',
                   u'PayDayAvgMonth_Avg', u'PayDayAvgMonth_Max', u'PayDayAvgMonth_Min',
                   u'PayDayAvgMonth_Std', u'PayDayAvgMonth_Sum', u'PayDayMaxMonth_Avg',
                   u'PayDayMaxMonth_Max', u'PayDayMaxMonth_Min',u'PayDayMaxMonth_Std',
                   u'PayDayMaxMonth_Sum',  u'PayDayMinMonth_Avg', u'PayDayMinMonth_Max',
                   u'PayDayMinMonth_Min', u'PayDayMinMonth_Std', u'PayDayMinMonth_Sum',
                   u'PayDayAvgWeek_Avg', u'PayDayAvgWeek_Max', u'PayDayAvgWeek_Min', u'PayDayAvgWeek_Std',
                   u'PayDayAvgWeek_Sum', u'PayDayMaxWeek_Avg', u'PayDayMaxWeek_Max', u'PayDayMaxWeek_Min',
                   u'PayDayMaxWeek_Std', u'PayDayMaxWeek_Sum', u'PayDayMinWeek_Avg', u'PayDayMinWeek_Max',
                   u'PayDayMinWeek_Min', u'PayDayMinWeek_Std', u'PayDayMinWeek_Sum',
                   u'PayDayAvgQuater_Avg', u'PayDayAvgQuater_Max', u'PayDayAvgQuater_Min',
                   u'PayDayAvgQuater_Std', u'PayDayAvgQuater_Sum', u'PayDayMaxQuater_Avg',
                   u'PayDayMaxQuater_Max', u'PayDayMaxQuater_Min', u'PayDayMaxQuater_Std',
                   u'PayDayMaxQuater_Sum', u'PayDayMinQuater_Avg', u'PayDayMinQuater_Max',
                   u'PayDayMinQuater_Min', u'PayDayMinQuater_Std', u'PayDayMinQuater_Sum',
                   u'PayDayAvgYr_Avg', u'PayDayAvgYr_Max', u'PayDayAvgYr_Min', u'PayDayAvgYr_Std',
                   u'PayDayAvgYr_Sum', u'PayDayMaxYr_Avg', u'PayDayMaxYr_Max', u'PayDayMaxYr_Min',
                   u'PayDayMaxYr_Std', u'PayDayMaxYr_Sum', u'PayDayMinYr_Avg', u'PayDayMinYr_Max',
                   u'PayDayMinYr_Min', u'PayDayMinYr_Std', u'PayDayMinYr_Sum']
FeatureYList = [ u'Chain25PayDay',u'Chain50PayDay', u'Chain75PayDay',u'ChainAmtPayDay', u'ChainAvgPayDay',
                   u'ChainCvPayDay', u'ChainIQRPayDay', u'ChainKurtPayDay', u'ChainPayDay1sig',
                   u'ChainPayDay2sig', u'ChainPayDay3sig',u'ChainSkewPayDay',u'ChainCrAmt', u'ChainCrNum',
                   u'ChainMaxPayDay',u'ChainMinPayDay',u'ChainNumPayDay',u'ChainStdPayDay',
                   u'PayDayAvgMonth_case_dur', u'PayDayAvgMonth_ts_len', u'PayDayAvgMonth_tsd_len',
                   u'PayDayAvgMonth_ts_pct', u'PayDayAvgMonth_tsd_pct', u'PayDayAvgMonth_tsdts_pct',
                   u'PayDayAvgWeek_case_dur', u'PayDayAvgWeek_ts_len', u'PayDayAvgWeek_tsd_len',
                   u'PayDayAvgWeek_ts_pct', u'PayDayAvgWeek_tsd_pct', u'PayDayAvgWeek_tsdts_pct',
                   u'PayDayAvgQuater_case_dur', u'PayDayAvgQuater_ts_len', u'PayDayAvgQuater_tsd_len',
                   u'PayDayAvgQuater_ts_pct', u'PayDayAvgQuater_tsd_pct', u'PayDayAvgQuater_tsdts_pct',
                   u'PayDayAvgYr_case_dur', u'PayDayAvgYr_ts_len', u'PayDayAvgYr_tsd_len',
                   u'PayDayAvgYr_ts_pct', u'PayDayAvgYr_tsd_pct', u'PayDayAvgYr_tsdts_pct',
                   u'PayDayAvgMonth_Avg', u'PayDayAvgMonth_Max', u'PayDayAvgMonth_Min',
                   u'PayDayAvgMonth_Std', u'PayDayAvgMonth_Sum', u'PayDayMaxMonth_Avg',
                   u'PayDayMaxMonth_Max', u'PayDayMaxMonth_Min',u'PayDayMaxMonth_Std',
                   u'PayDayMaxMonth_Sum',  u'PayDayMinMonth_Avg', u'PayDayMinMonth_Max',
                   u'PayDayMinMonth_Min', u'PayDayMinMonth_Std', u'PayDayMinMonth_Sum',
                   u'PayDayAvgWeek_Avg', u'PayDayAvgWeek_Max', u'PayDayAvgWeek_Min', u'PayDayAvgWeek_Std',
                   u'PayDayAvgWeek_Sum', u'PayDayMaxWeek_Avg', u'PayDayMaxWeek_Max', u'PayDayMaxWeek_Min',
                   u'PayDayMaxWeek_Std', u'PayDayMaxWeek_Sum', u'PayDayMinWeek_Avg', u'PayDayMinWeek_Max',
                   u'PayDayMinWeek_Min', u'PayDayMinWeek_Std', u'PayDayMinWeek_Sum',
                   u'PayDayAvgQuater_Avg', u'PayDayAvgQuater_Max', u'PayDayAvgQuater_Min',
                   u'PayDayAvgQuater_Std', u'PayDayAvgQuater_Sum', u'PayDayMaxQuater_Avg',
                   u'PayDayMaxQuater_Max', u'PayDayMaxQuater_Min', u'PayDayMaxQuater_Std',
                   u'PayDayMaxQuater_Sum', u'PayDayMinQuater_Avg', u'PayDayMinQuater_Max',
                   u'PayDayMinQuater_Min', u'PayDayMinQuater_Std', u'PayDayMinQuater_Sum',
                   u'PayDayAvgYr_Avg', u'PayDayAvgYr_Max', u'PayDayAvgYr_Min', u'PayDayAvgYr_Std',
                   u'PayDayAvgYr_Sum', u'PayDayMaxYr_Avg', u'PayDayMaxYr_Max', u'PayDayMaxYr_Min',
                   u'PayDayMaxYr_Std', u'PayDayMaxYr_Sum', u'PayDayMinYr_Avg', u'PayDayMinYr_Max',
                   u'PayDayMinYr_Min', u'PayDayMinYr_Std', u'PayDayMinYr_Sum',u'dis_index',u'com_index',u'con_index']
FeatureYIndex = IndexY[FeatureYList]

ChainNum = len(FeatureYIndex)
NanNum = FeatureYIndex.isnull().sum()
Nanpct = NanNum / ChainNum
NanIndex = pd.DataFrame({'NanNum':NanNum,'Nanpct':Nanpct})
##check Nanpct > 0 ###
print NanIndex[NanIndex.Nanpct>0]

############
FeatureYIndex = FeatureYIndex[(FeatureYIndex['ChainNumPayDay']>=1)]
FeatureYIndex[FeatureList] = FeatureYIndex[FeatureList].fillna(0)
FeatureYIndex.to_csv('ModelIndex_dis.csv',encoding='gbk')

ModelSample = FeatureYIndex.dropna(axis=0,subset=['dis_index'])
OtherSample = FeatureYIndex[pd.isnull(FeatureYIndex['dis_index'])]

######Index Selection Scale#######
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelSample[FeatureList])
RbIndex = rbscale.transform(ModelSample[FeatureList])
RbIndex = pd.DataFrame(RbIndex,columns=ModelSample[FeatureList].columns,index=ModelSample[FeatureList].index)

RbIndexCorr = RbIndex.corr()
RbIndexCorr.to_csv('RbIndexCorr.csv',encoding='utf8')
RbIndex['dis_index'] = ModelSample['dis_index']

#################Index Selection##################
######1#####
#VarTIndexSelect = ModelSample.drop(['con_index','com_index','dis_index'],axis=1)
#VarTYSelect = ModelSample[['con_index','com_index','dis_index']]

from sklearn.feature_selection import VarianceThreshold
VarT = VarianceThreshold(threshold=(0.8*(1-0.8)))
VarTvari = VarT.fit(RbIndex[FeatureList]).variances_
VarTvari = pd.DataFrame(VarTvari,index=RbIndex[FeatureList].columns)
VarTvari.columns = ['variances']
VarTFeatureList = VarTvari[VarTvari.variances>=0.2].index.tolist()
VarTFeatureList.append('dis_index')
VarTSample = ModelSample[VarTFeatureList]
VarTFeatureList.pop()

#####2#####
#UniIndexSelect = IndexVarT.drop(['con_index','com_index','dis_index'],axis=1)
#UniYSelect = IndexVarT[['con_index','com_index','dis_index']]

from sklearn.feature_selection import f_classif, mutual_info_classif,chi2
mi = mutual_info_classif(VarTSample[VarTFeatureList],VarTSample.dis_index)
mi = pd.Series(mi,index=VarTSample[VarTFeatureList].columns)
ff,fp = f_classif(VarTSample[VarTFeatureList],VarTSample.dis_index)
ff = pd.Series(ff,index=VarTSample[VarTFeatureList].columns)
fp = pd.Series(fp,index=VarTSample[VarTFeatureList].columns)
SelectIndex1 = pd.DataFrame({'mi':mi,'ff':ff,'fp':fp})
SelectIndex1.to_csv('SelectIndex1.csv',encoding='gbk')
UniFeatureList = SelectIndex1[(SelectIndex1['fp']<=0.05)].index.tolist()
UniFeatureList.append('dis_index')
UniSample = VarTSample[UniFeatureList]
UniFeatureList.pop()

######3-RFE#####
#RFEIndexSelect = IndexUni.drop(['con_index','com_index','dis_index'],axis=1)
#RFEYSelect = IndexUni[['con_index','com_index','dis_index']]

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def RFESelectIndex(estimator,Index,Y):
    t0 = time()
    Ref = RFE(estimator=estimator)
    Ref.fit(Index,Y)
    IndexRank = Ref.ranking_
    IndexRank = pd.Series(IndexRank,index=Index.columns)
    print time()-t0
    return IndexRank

LR = LogisticRegression()
RankLR = RFESelectIndex(LR,UniSample[UniFeatureList],UniSample.dis_index)

DT = DecisionTreeClassifier()
RankDT = RFESelectIndex(DT,UniSample[UniFeatureList],UniSample.dis_index)

RF = RandomForestClassifier()
RankRF = RFESelectIndex(RF,UniSample[UniFeatureList],UniSample.dis_index)

SVM = SVC(kernel="linear")
RankSVM = RFESelectIndex(SVM,UniSample[UniFeatureList],UniSample.dis_index)

SelectIndex2 = pd.DataFrame({'LR':RankLR,'DT':RankDT,'RF':RankRF,'SVC':RankSVM})
SelectIndex2.to_csv('SelectIndex2.csv',encoding='gbk')

######3-RFECV########
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

def RFECVSelectIndex(estimator,Index,Y,Fold=5):
    Rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5),scoring='accuracy')
    Rfecv.fit(Index,Y)
    scores = Rfecv.grid_scores_
    IndexRank = Rfecv.ranking_
    IndexRank = pd.Series(IndexRank,index=Index.columns)
    scores = pd.Series(scores,index=range(1, len(scores) + 1))
    return IndexRank,scores

LR = LogisticRegression()
RankcvLR,ScorecvLR = RFECVSelectIndex(LR,UniSample[UniFeatureList],UniSample.dis_index)

DT = DecisionTreeClassifier()
RankcvDT,ScorecvDT = RFECVSelectIndex(DT,UniSample[UniFeatureList],UniSample.dis_index)

RF = RandomForestClassifier()
RankcvRF,ScorecvRF = RFECVSelectIndex(RF,UniSample[UniFeatureList],UniSample.dis_index)

SVM = SVC(kernel='linear')
RankcvSVM,ScorecvSVM = RFECVSelectIndex(SVM,UniSample[UniFeatureList],UniSample.dis_index)

SelectIndex3 = pd.DataFrame({'LR':RankcvLR,'DT':RankcvDT,'RF':RankcvRF,'SVC':RankcvSVM})
SelectIndex3.to_csv('SelectIndex3.csv',encoding='gbk')
SelectNum = pd.DataFrame({'LR':ScorecvLR,'DT':ScorecvDT,'RF':ScorecvRF,'SVC':ScorecvSVM})
SelectNum.to_csv('SelectNum.csv',encoding='gbk')

#########3-SFM#####################
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

def SFMIndexSelect(estimator,Index,Y):
    Sfm = SelectFromModel(estimator=estimator,prefit=True,threshold='mean')
    Sfm.fit(RFEIndexSelect,RFEYSelect.dis_index)
    Support = Sfm.get_support()
    Support = pd.Series(Support,index=Index.columns)
    return Support

Lasso = LassoCV()
LassoSupport = SFMIndexSelect(Lasso,UniSample[UniFeatureList],UniSample.dis_index)

Svm = LinearSVC(C=0.01, penalty="l2", dual=False,multi_class='ovr')
SvmSupport = SFMIndexSelect(Svm,UniSample[UniFeatureList],UniSample.dis_index)

ET = ExtraTreesClassifier()
ETSupport = SFMIndexSelect(ET,UniSample[UniFeatureList],UniSample.dis_index)

LR = LogisticRegression()
LRSupport = SFMIndexSelect(LR,UniSample[UniFeatureList],UniSample.dis_index)

RF = RandomForestClassifier()
RFSupport = SFMIndexSelect(RF,UniSample[UniFeatureList],UniSample.dis_index)

SelectIndex4 = pd.DataFrame({'Lasso':LassoSupport,'LR':LRSupport,'ET':ETSupport,'RF':RFSupport,'SVM':SvmSupport})
SelectIndex4.to_csv('SelectIndex4.csv',encoding='gbk')

###################
ModelSelectList = ['ChainMaxPayDay','PayDayMaxQuater_Std','ChainCvPayDay','PayDayMaxMonth_Std',
                   'PayDayMaxMonth_Max','PayDayMaxWeek_Max','PayDayMaxQuater_Max','ChainStdPayDay',
                   'PayDayMaxWeek_Std','PayDayAvgQuater_Std','PayDayAvgMonth_Max','PayDayAvgMonth_Std',
                   'PayDayAvgWeek_Max','PayDayAvgWeek_Std','PayDayAvgQuater_Max','PayDayMaxQuater_Avg',
                   'con_index','com_index','dis_index']
ModelSelectIndex = UniSample[ModelSelectList]
ModelSelectIndexCorr = ModelSelectIndex.corr()
ModelSelectIndexCorr.to_csv('ModelSelectIndexCorr_con.csv',encoding='gbk')

##########Model Index############
model_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std','dis_index']
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']

ModelIndex = ModelSample[model_var]
ModelIndex.to_csv('ModelIndex.csv',encoding='gbk')

for i in ModelIndex:
    fig = sns.distplot(ModelIndex[i]).get_figure()
    fig.savefig(i)
    fig.clear()

######Cluster Feature Scale#######
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelIndex[feature_var])
RbCluster = rbscale.transform(ModelIndex[feature_var])

RbCluster = pd.DataFrame(RbCluster,columns=ModelIndex[feature_var].columns,index=ModelIndex.index)
RbClusterCorr = RbCluster.corr()
RbClusterCorr.to_csv('RbClusterCorr.csv',encoding='utf8')

#####样本距离分析####
mahala = distance.pdist(RbCluster,metric='mahalanobis', VI=None)
mahala_tran = distance.squareform(mahala)

mahala_tran = pd.DataFrame(mahala_tran,columns=RbCluster.index,index=RbCluster.index)
mahala_tran.to_csv('Mahala.csv',encoding='utf8')

Nearest = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6','7','8','9','10','chain_name'])
for i in mahala_tran.index:
    chain_nearest = mahala_tran.nsmallest(11,i,'first').index.tolist()
    chain_nearest.append(i)
    Nearest = Nearest.append(pd.Series(chain_nearest, index=['0', '1', '2', '3', '4', '5', '6','7','8','9','10',
                                                             'chain_name']),ignore_index=True)
Nearest = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6','7','8','9','10','chain_name'])
for i in mahala_tran.index:
    chain_nearest = mahala_tran.nsmallest(11,i,'first').index.tolist()
    chain_nearest.append(i)
    Nearest = Nearest.append(pd.Series(chain_nearest, index=['0', '1', '2', '3', '4', '5', '6','7','8','9','10',
                                                             'chain_name']),ignore_index=True)
Nearest.set_index('chain_name')
Nearest.to_csv('Nearest.csv',encoding='gbk')

###########成分分析########
# PCA on the fruit dataset (for comparison)
pca = PCA(n_components = 4,whiten=True).fit(RbCluster)
RbIndex_pca = pca.transform(RbCluster)
RbIndex_pcacom = pca.components_
RbIndex_pcaexr = pca.explained_variance_ratio_
RbIndex_pca_df = pd.DataFrame(RbIndex_pca,index=RbCluster.index)
RbIndex_pcacom_df = pd.DataFrame(RbIndex_pcacom,columns=RbCluster.columns)
RbIndex_pcaexr_df = pd.DataFrame(RbIndex_pcaexr)
RbIndex_pca_df.to_csv('RbIndex_pca.csv',encoding='utf8')
RbIndex_pcacom_df.to_csv('RbIndex_pcacom.csv',encoding='utf8')
RbIndex_pcaexr_df.to_csv('RbIndex_pcaexr.csv',encoding='utf8')

# FA
fa = FactorAnalysis(n_components=4).fit(RbCluster)
RbIndex_fa = fa.transform(RbCluster)
RbIndex_facom = fa.components_
RbIndex_fanv = fa.noise_variance_
RbIndex_fa_df = pd.DataFrame(RbIndex_fa,index=RbCluster.index)
RbIndex_facom_df = pd.DataFrame(RbIndex_facom,columns=RbCluster.columns)
RbIndex_fanv_df = pd.DataFrame(RbIndex_fanv,index=RbCluster.columns)
RbIndex_fa_df.to_csv('RbIndex_fa.csv',encoding='utf8')
RbIndex_facom_df.to_csv('RbIndex_facom.csv',encoding='utf8')
RbIndex_fanv_df.to_csv('RbIndex_fanv.csv',encoding='utf8')

##############K-Means聚类##############################
n_samples,n_features = RbCluster.shape
digits = 20
#labels = TrainIndex.labels
sample_size = n_samples

from sklearn.cluster import KMeans
def bench_k_means(estimator,name,data):
    t0 = time()
    estimator.fit(data)
    parameter = [name,estimator.inertia_,(time()-t0),
                 metrics.silhouette_score(data, estimator.labels_, metric='sqeuclidean',sample_size=sample_size),
                 metrics.calinski_harabaz_score(data, estimator.labels_)]
    return parameter

KMeansScore = pd.DataFrame(columns=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters'])
for i in range(3,digits,1):
    n_digits = i
#    prRandom = bench_k_means(KMeans(init='random',n_clusters=i,n_init=20),name='Random',data=RbCluster)
    prKplus = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='k-means++',data=RbCluster)
    prKplus.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prKplus, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

    ReducedTrainIndex = PCA(n_components=2,whiten=True).fit_transform(RbCluster)
    prReDuc = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='ReDuc_PCA',data=ReducedTrainIndex)
    prReDuc.append(n_digits)
    KMeansScore = KMeansScore.append(
        pd.Series(prReDuc, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
        ignore_index=True)

#    ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedTrainIndex)
#    ReQtuTrainIndex = ReQtuscale.transform(ReducedTrainIndex)
#    prReQtu = bench_k_means(KMeans(init='k-means++',n_clusters=i,n_init=20),name='ReQtu_PCA',data=ReQtuTrainIndex)
#    prReQtu.append(n_digits)
#    KMeansScore = KMeansScore.append(
#        pd.Series(prReQtu, index=['name','SSE','time','silhouette_score','calinski_harabaz_score','n_clusters']),
#        ignore_index=True)

#    ReQtnscale = preprocessing.QuantileTransformer(output_distribution='normal').fit(ReducedTrainIndex)
#    ReQtnTrainIndex = ReQtnscale.transform(ReducedTrainIndex)
#    prReQtn = bench_k_means(KMeans(init='k-means++', n_clusters=i, n_init=20), name='ReQtn_PCA', data=ReQtnTrainIndex)
#    prReQtn.append(n_digits)
#    KMeansScore = KMeansScore.append(
#        pd.Series(prReQtn, index=['name', 'SSE', 'time', 'silhouette_score', 'calinski_harabaz_score', 'n_clusters']),
#        ignore_index=True)

KMeansScore.to_csv('KMeansScore.csv',encoding='utf8')

########分类算法##################
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as conti_randuni

X_train,X_test,Y_train,Y_test = train_test_split(ModelIndex[feature_var],ModelIndex.dis_index,test_size=0.2,stratify=ModelIndex.dis_index,random_state=0)
SKF = StratifiedKFold(n_splits=6,shuffle=False,random_state=0)
ClassfyResult = pd.DataFrame(columns=['Name','Time','ScoreMu','ScoreStd','ScoreOpt','ParameterOpt'])

#####Sample Number Evaluation######
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plt_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_size=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(
        estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_size)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,"o-",color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,"o-",color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt

digits=load_digits()
X,y=digits.data,digits.target
title="Learning Curves(Nativr Bayes)"

cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
estimator=GaussianNB()
plt_learning_curve(estimator,title,X,y,ylim=(0.7,1.0),cv=cv,n_jobs=1)
title="Learnming Curves (SVM,RBF kernel,$\gamma=0.001$)"
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
estimator=SVC(gamma=0.001)
plt_learning_curve(estimator,title,X,y,(0.7,1.01),cv=cv,n_jobs=1)
plt.show()

######Classify Feature Scale#######
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(X_train)
RbX_train = rbscale.transform(X_train)
RbX_test = rbscale.transform(X_test)
RbX_train = pd.DataFrame(RbX_train,columns=X_train.columns,index=X_train.index)
RbX_test = pd.DataFrame(RbX_test,columns=X_test.columns,index=X_test.index)

def bench_classify(estimator,name,t_index,t_y,parameters):
    t0 = time()
    scores = cross_val_score(estimator,t_index,t_y,cv=SKF)
    if pd.isnull(parameters):
        result = [name,(time()-t0),scores.mean(),scores.std(),None,None]
        print(name, " Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    else:
        grid = GridSearchCV(estimator, param_grid=parameters)
        grid.fit(t_index, t_y)
        result = [name,(time()-t0),scores.mean(),scores.std(),grid.best_score_,grid.best_params_]

        Scores = grid.grid_scores_
        pd.DataFrame(Scores).to_csv(name+'Score.csv',encoding='gbk')
        print(name," Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

    return result

###Every Parameter by pic###
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
digits = load_digits()
param_range=np.logspace(-6,-1,5)
train_scores,test_scores=validation_curve(SVC(),X,y,param_name="gamma",param_range=param_range,
                                          cv=10,scoring="accuracy",n_jobs=1)
train_scores_mean=np.mean(train_scores,axis=1)
train_scores_std=np.std(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)
test_scores_std=np.std(test_scores,axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0,1.1)
lw=2
plt.semilogx(param_range,train_scores_mean,label="Training score",color="darkorange",lw=lw)
plt.fill_between(param_range,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,
                 alpha=0.2,color="darkorange",lw=lw)
plt.semilogx(param_range,test_scores_mean,label="Cross-validation Score",color="navy",lw=lw)
plt.fill_between(param_range,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,
                 alpha=0.2,color="navy",lw=lw)
plt.legend(loc="best")
plt.show()

#######################LR#####################
LR = LogisticRegression()

#1####Optimized Parameter####
#parameter prepare
class_weight = ['balanced',None]
solver = ['newton-cg','lbfgs','liblinear','sag','saga']
C = range(1,10,1)
parameters = {'class_weight':class_weight,'solver':solver,'C':C}
LRResult1 = bench_classify(LR,"LR",RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(LRResult1,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

#2#####parameter prepare#####
class_weight = ['balanced',None]
solver = ['newton-cg','lbfgs','sag','saga']
C = np.arange(0.1,15,0.1)
multi_class = ['ovr','multinomial']
parameters = {'class_weight':class_weight,'solver':solver,'C':C,'multi_class':multi_class}
LRResult2 = bench_classify(LR,'LR',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(LRResult2,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

#3##parameter prepare##
penalty = ['l1','l2']
class_weight = ['balanced',None]
solver = ['saga']
C = range(1,20,1)
multi_class = ['ovr','multinomial']
parameters = {'penalty':penalty,'class_weight':class_weight,'solver':solver,'C':C,'multi_class':multi_class}
LRResult3 = bench_classify(LR,"LR",RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(LRResult3,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####GaussianNB#####
GNB = GaussianNB()
GNBResult = bench_classify(GNB,"GNB",RbX_train,Y_train,None)
ClassfyResult = ClassfyResult.append(pd.Series(GNBResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

#####KNN####
KNN = KNeighborsClassifier()
####Optimized Parameter###
###parameter prepare###
n_neighbors = range(2,4,1)
weights = ['uniform','distance']
algorithm = ['auto','ball_tree','kd_tree','brute']
p = range(1,5,1)
metric = ['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis']
parameters = {'n_neighbors':n_neighbors,'weights':weights,'algorithm':algorithm,'p':p}
KNNResult = bench_classify(KNN,'KNN',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(KNNResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####DT######
DT = DecisionTreeClassifier()
###Optimized Parameter###
##parameter prepare##
criterion = ['gini','entropy']
splitter = ['best','random']
max_features = ['sqrt','log2',None]
max_depth = range(3,10,1)
min_sample_split = range(2,4,1)
min_sample_leaf = range(1,10,1)
bootstrap = [True,False]
parameters = {'criterion':criterion,'splitter':splitter,'max_depth':max_depth,'max_feature':max_features,
              'min_samples_split':min_sample_split,'min_sample_leaf':min_sample_leaf,'bootstrap':bootstrap}
DTResult = bench_classify(DT,'DT',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(DTResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

#####ET######
ET = ExtraTreesClassifier()
###Optimized Parameter###
##parameter prepare##
n_estimators = range(10,20,1)
criterion = ['gini','entropy']
max_features = ['sqrt','log2',None]
max_depth = range(3,10,1)
min_sample_split = range(2,4,1)
min_sample_leaf = range(1,10,1)
bootstrap = [True,False]
class_weight = ['balanced','balanced_subsample',None]
parameters = {'n_estimators':n_estimators,'criterion':criterion,'max_features':max_features,
              'max_depth':max_depth,'min_samples_split':min_sample_split,'class_weight':class_weight,
              'min_sample_leaf':min_sample_leaf,'bootstrap':bootstrap}
ETResult = bench_classify(ET,'ET',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(ETResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####RF######
RF = RandomForestClassifier()
###Optimized Parameter###
##parameter prepare##
n_estimators = range(10,20,1)
criterion = ['gini','entropy']
max_features = ['sqrt','log2',None]
max_depth = range(3,10,1)
min_sample_split = range(2,4,1)
class_weight = ['balanced','balanced_subsample',None]
##cross validation##
parameters = {'n_estimators':n_estimators,'criterion':criterion,'max_features':max_features,
              'max_depth':max_depth,'min_samples_split':min_sample_split,'class_weight':class_weight}
RFResult = bench_classify(RF,'RF',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(RFResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####LDA######
LDA = LinearDiscriminantAnalysis()
###Optimized Parameter###
##parameter prepare##
solver = ['svd','lsqr','eigen']
parameters = {'solver':solver}
LDAResult = bench_classify(LDA,'LDA',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(LDAResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####QDA######
QDA = QuadraticDiscriminantAnalysis()
QDAResult = bench_classify(QDA,'QDA',RbX_train,Y_train,None)
ClassfyResult = ClassfyResult.append(pd.Series(QDAResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

####SVM####
SVM = SVC()
###Optimized Parameter###
##parameter prepare##
C = range(1,10,1)
kernel = ['linear','poly','rbf','sigmoid']
decision_function_shape = ['ovo','ovr']
parameters = {'C':C,'kernel':kernel,'decision_function_shape':decision_function_shape}
SVMResult = bench_classify(SVM,'SVM',RbX_train,Y_train,parameters)
ClassfyResult = ClassfyResult.append(pd.Series(SVMResult,index=['Name','Time','ScoreMu','ScoreStd',
                                                                'ScoreOpt','ParameterOpt']),
                                     ignore_index=True)

ClassfyResult.to_csv('ClassfyResults.csv',encoding='gbk')

#########More for RF & ET########
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

def EnsembleModel(estimator,name,X,Y):
    t0 = time()
    scores = cross_val_score(estimator,X,Y,cv=SKF)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    estimator.fit(X, Y)
    oobscore = estimator.oob_score_

    ######Feature Select########
    FeatureImprotance = pd.Series(estimator.feature_importances_,index=X.columns)
    FeatureImprotance.sort_values(inplace=True,ascending=False)
    obb_score = []
    cv_score = []
    feature = []
    for i in range(1,len(FeatureImprotance)+1):
        Feature = FeatureImprotance[:i].index.tolist()
        feature.append(Feature)
        FeatureX = X[Feature]
        scores = cross_val_score(estimator,FeatureX,Y,cv=SKF)
        cv_score.append(np.mean(scores))
        estimator.fit(FeatureX,Y)
        obb_score.append(estimator.oob_score_)
    pd.DataFrame({'feature':feature,'cv_score':cv_score,'obb_score':obb_score}).to_csv('FeatureSelect.csv',encoding='gbk')

    ########Model parameters#######
    estimator.set_params(random_state=0, oob_score=True)
    parameters = {'criterion': ('gini', 'entropy'), 'max_features': ('sqrt', 'log2', None, 1, 2, 3, 4),
                  'max_depth': [None, 3, 5, 8, 10],'n_estimators':[10,20,30,40,50]}
    grid = GridSearchCV(estimator, param_grid=parameters)
    grid.fit(X,Y)
    grid.grid_scores_
    grid.cv_results_
    result = [name, (time() - t0), scores.mean(), scores.std(), oobscore, grid.best_score_, grid.best_params_]
    print result

######Ensemble Model Feature Prepare#######
X_EMtrain,X_EMtest,Y_EMtrain,Y_EMtest = train_test_split(UniSample[UniFeatureList],UniSample.dis_index,test_size=0.2,stratify=UniSample.dis_index,random_state=0)
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(X_EMtrain)
RbX_EMtrain = rbscale.transform(X_EMtrain)
RbX_EMtest = rbscale.transform(X_EMtest)
RbX_EMtrain = pd.DataFrame(RbX_EMtrain,columns=X_EMtrain.columns,index=X_EMtrain.index)
RbX_EMtest = pd.DataFrame(RbX_EMtest,columns=X_EMtest.columns,index=X_EMtest.index)

############RF############
RF = RandomForestClassifier(random_state=0,oob_score=True)
EnsembleModel(RF,'RF',RbX_EMtrain,Y_EMtrain)

############ET############
ET = ExtraTreesClassifier(random_state=0,oob_score=True,bootstrap=True)
EnsembleModel(ET,'ET',RbX_EMtrain,Y_EMtrain)


'''
print(__doc__)

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
'''