#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

###########################################################Index#####################################################
###get index###
Index = pd.read_csv('ModelIndex.csv',encoding='gbk',index_col=0)

####model var###
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']
ModelIndex = Index[feature_var].fillna(0)
modelindex = ModelIndex.index
modelcolumns = ModelIndex.columns

####scale####
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelIndex)
ModelIndex = rbscale.transform(ModelIndex)

##################################################Cluster Model#####################################################
n_digits = 7
###variable preprocess####
pca = PCA(n_components=5,whiten=True).fit(ModelIndex)
ReducedModelIndex = pca.transform(ModelIndex)
ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedModelIndex)
ReducedModelIndex = ReQtuscale.transform(ReducedModelIndex)

Cluster = KMeans(init='k-means++',n_clusters=n_digits,n_init=20).fit(ReducedModelIndex)
FinalOutput = pd.DataFrame(ModelIndex,columns=modelcolumns,index=modelindex)
FinalOutput['predict_index'] = Cluster.predict(ReducedModelIndex)
FinalOutput.to_csv('duration_result.csv',encoding='gbk')


###########################################Same Project Predict####################################################
ProjectIndex = pd.read_csv('Index.csv',encoding='gbk',index_col=0)

####Index Prepare####
PredictIndex = ProjectIndex[ProjectIndex['ChainNumPayDay']>=1]

####model var#####
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']
ModelIndex = PredictIndex[feature_var].fillna(0)
modelindex = ModelIndex.index
modelcolumns = ModelIndex.columns

###variable preprocess###
ModelIndex = rbscale.transform(ModelIndex)
ReducedModelIndex = pca.transform(ModelIndex)
ReducedModelIndex = ReQtuscale.transform(ReducedModelIndex)

###Results####
FinalOutput = pd.DataFrame(ModelIndex,columns=modelcolumns,index=modelindex)
FinalOutput['predict_index'] = Cluster.predict(ReducedModelIndex)
FinalOutput.to_csv('project_result.csv',encoding='gbk')


###########################################New Project Predict#####################################################
ProjectIndex = pd.read_csv('Index.csv',encoding='gbk',index_col=0)

####Index Prepare####
PredictIndex = ProjectIndex[ProjectIndex['ChainNumPayDay']>=1]

####model var#####
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']
ModelIndex = PredictIndex[feature_var].fillna(0)
modelindex = ModelIndex.index
modelcolumns = ModelIndex.columns

###variable preprocess###
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelIndex)
ModelIndex = rbscale.transform(ModelIndex)
ReducedModelIndex = pca.transform(ModelIndex)
ReQtuscale = preprocessing.QuantileTransformer(output_distribution='uniform').fit(ReducedModelIndex)
ReducedModelIndex = ReQtuscale.transform(ReducedModelIndex)

###Results####
FinalOutput = pd.DataFrame(ModelIndex,columns=modelcolumns,index=modelindex)
FinalOutput['predict_index'] = Cluster.predict(ReducedModelIndex)
FinalOutput.to_csv('project_result.csv',encoding='gbk')