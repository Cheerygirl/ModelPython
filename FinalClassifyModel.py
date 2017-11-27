#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import pandas as pd
from sklearn import preprocessing

################################################Index#########################################################
###get index###
Index = pd.read_csv('Index.csv',encoding='gbk',index_col=0)
Y = pd.read_csv('ModelY.csv',index_col=0)
IndexY = pd.concat([Index,Y],axis=1)

####model var#####
model_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std','dis_index']
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']

##########Index Prepare###########
ChainNum = len(IndexY)
NanNum = IndexY.isnull().sum()
Nanpct = NanNum / ChainNum
NanIndex = pd.DataFrame({'NanNum':NanNum,'Nanpct':Nanpct})
##check Nanpct > 0 ###
print NanIndex[NanIndex.Nanpct>0]

############
IndexY = IndexY[IndexY['ChainNumPayDay']>=1]
############
IndexY[feature_var] = IndexY[feature_var].fillna(0)

#####Get Model Index##########
ModelIndex = IndexY[model_var]
ModelIndex.to_csv('ModelIndex_dis.csv',encoding='gbk')

ModelSample = ModelIndex.dropna(axis=0,subset=['dis_index'])
OtherSample = ModelIndex[pd.isnull(ModelIndex['dis_index'])]

###################################################Final Model######################################################
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train,X_test,Y_train,Y_test = train_test_split(ModelSample[feature_var],ModelSample['dis_index'],
                                                 test_size=0.2,stratify=ModelSample['dis_index'],random_state=0)

###train sample scale###
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(X_train)
RbX_train = rbscale.transform(X_train)
RbX_test = rbscale.transform(X_test)

LR = LogisticRegression(penalty='l1',multi_class='multinomial',C=3,solver='saga',class_weight=None)
LR.fit(RbX_train, Y_train)
test_expected = Y_test
test_predicted = LR.predict(RbX_test)
print(metrics.classification_report(test_expected, test_predicted))
print(metrics.confusion_matrix(test_expected, test_predicted))
print(metrics.accuracy_score(test_expected,test_predicted),metrics.f1_score(test_expected,test_predicted),
      metrics.jaccard_similarity_score(test_expected,test_predicted),metrics.precision_score(test_expected,test_predicted),
      metrics.recall_score(test_expected,test_predicted))
print(metrics.zero_one_loss(test_expected,test_predicted),metrics.log_loss(test_expected,test_predicted),
      metrics.hamming_loss(test_expected,test_predicted),metrics.hinge_loss(test_expected,test_predicted))
train_expected = Y_train
train_predited = LR.predict(RbX_train)
print(metrics.classification_report(train_expected, train_predited))
print(metrics.confusion_matrix(train_expected, train_predited))
'''
#####ROC#####
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
###Y prepare###
ModelY = ModelSample.dis_index
ModelY = label_binarize(ModelY,classes=[1,2,3])
n_classes = ModelY.shape[1]

X_train,X_test,Y_train,Y_test = train_test_split(ModelSample[feature_var],ModelY,test_size=0.2,random_state=0)

###train sample scale###
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(X_train)
RbX_train = rbscale.transform(X_train)
RbX_test = rbscale.transform(X_test)

LR = OneVsRestClassifier(LogisticRegression(penalty='l1',multi_class='multinomial',C=3,solver='saga',class_weight=None))
LR.fit(RbX_train, Y_train)
test_score = LR.decision_function(RbX_test)
test_expected = Y_test
test_predicted = LR.predict(RbX_test)
print(metrics.classification_report(test_expected, test_predicted))
print(metrics.accuracy_score(test_expected,test_predicted),metrics.jaccard_similarity_score(test_expected,test_predicted))
print(metrics.zero_one_loss(test_expected,test_predicted),metrics.log_loss(test_expected,test_predicted),
      metrics.hamming_loss(test_expected,test_predicted))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_expected[:, i], test_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(test_expected.ravel(), test_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

####multi-class ROC####
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
'''
#####predict data######
ModelSample[feature_var] = rbscale.transform(ModelSample[feature_var])
ModelSample['predict_index'] = LR.predict(ModelSample[feature_var])
OtherSample[feature_var] = rbscale.transform(OtherSample[feature_var])
OtherSample['predict_index'] = LR.predict(OtherSample[feature_var])

ModelResults = pd.concat([ModelSample,OtherSample])
ModelResults.to_csv('ModelResults.csv',encoding='gbk')

###########################################Same Project Predict####################################################
ProjectIndex = pd.read_csv('Index.csv',encoding='gbk',index_col=0)

####Index Prepare####
PredictIndex = ProjectIndex[ProjectIndex['ChainNumPayDay']>=1]

####model var#####
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']
ModelIndex = PredictIndex[feature_var].fillna(0)

###Robost Scale###
ModelIndex = rbscale.transform(ModelIndex)

###Results####
ModelIndex['predict_index'] = LR.predict(ModelIndex)

###########################################New Project Predict#####################################################
ProjectIndex = pd.read_csv('Index.csv',encoding='gbk',index_col=0)

####Index Prepare####
PredictIndex = ProjectIndex[ProjectIndex['ChainNumPayDay']>=1]

####model var#####
feature_var = ['ChainMaxPayDay','ChainCvPayDay','ChainStdPayDay','PayDayAvgQuater_Std']
ModelIndex = PredictIndex[feature_var].fillna(0)

###Robost Scale###
rbscale = preprocessing.RobustScaler(quantile_range=(25, 75)).fit(ModelIndex)
ModelIndex = rbscale.transform(ModelIndex)

###Results####
ModelIndex['predict_index'] = LR.predict(ModelIndex)