#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'ctt'

import numpy as np
import pandas as pd
import scipy.spatial.distance as spd
from scipy.cluster.hierarchy import dendrogram,linkage,cophenet,inconsistent,fcluster
from sklearn.cluster import DBSCAN,SpectralClustering,AffinityPropagation
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from time import time

##########Get Index##############
model_data=pd.read_csv('af_result.csv',encoding='utf8')

#######unhandle NAN#######

###max-min scale###
def preprocessing(df,nlist):
    data=df[nlist]
    n_min=data.min(axis=0,skipna=True)
    n_max=data.max(axis=0, skipna=True)
    df[nlist]=(data-n_min)/(n_max-n_min)
    return df

##
def numvar(x,y,list):
    demo1=x[list]
    demo2=y[list]
    ed=np.square(demo1-demo2).tolist()
    dist=np.nan_to_num(ed).sum()
    return dist

#dist中含Nan的距离
#两个记录同为类别变量，不同为0，相同不为0
def discvar(x,y,list):
    demo1 = x[list]
    demo2 = y[list]
    # hd=spd.hamming(demo1,demo2)
    # dist=hd*len(list)
    dist=sum(~(demo1==demo2).values)
    return dist

#两个变量里Nan的总记录数
def count_nanfea(x,y,list):
    x_nan= x[list]
    y_nan =y[list]
    nan_num=(x_nan+y_nan).isnull().sum()
    return nan_num

def mix_distance(x,y,nlist,clist):
    nl=len(nlist)
    cl=len(clist)
    n_nan=count_nanfea(x,y,nlist)
    c_nan = count_nanfea(x, y, clist)
    n_f=nl-n_nan
    c_f=cl-c_nan
    if (n_f+c_f)==0:
        pdist=np.nan
    else:
        if n_f>0:
            ndist=numvar(x,y,nlist)
        else:
            ndist=0
        cdist=discvar(x,y,clist)
        pdist=(ndist+cdist-c_nan)/(n_f+c_f)
    return pdist

def dis_matrix(df, nlist, clist):
    data_len=len(df)
    df=preprocessing(df, nlist)
    mdist=np.zeros([data_len,data_len])
    for i in range(data_len-1):
        x = df.loc[i]
        for j in range(i+1,data_len):
            y=df.loc[j]
            pdst=mix_distance(x,y, nlist, clist)
            mdist[i,j]=pdst
            mdist[j,i]=pdst
    return mdist

num_fea=['TrendBill','TrendPay','PayConcertrantion','PayDispersed','pay_intime','pay_stable']
disc_fea=['TradeFreq','TradeImportance','TradeDuration','LowSeasonBill','LowSeasonPay','MainCycleBill','MainCyclePay','PeakSeasonBill','PeakSeasonPay']
t0=time()
dist_matrix=dis_matrix(model_data,num_fea,disc_fea)
dist_matrix=pd.DataFrame(data=dist_matrix,columns=range(len(model_data)))
dist_matrix.to_csv('model_dist_matrix1.csv',encoding='gbk')
print (time()-t0)

dist_matrix=pd.read_excel('model_dist_matrix1.xlsx')
simi=dist_matrix.fillna(dist_matrix.max()).values

db = DBSCAN(eps=0.1,metric='precomputed').fit(simi)

sc = SpectralClustering(n_clusters=3,affinity='precomputed').fit(simi)

af = AffinityPropagation(affinity='precomputed').fit(simi)

metrics.silhouette_score(X=simi,metric ='precomputed', labels=af.labels_)

def af_damp(dist):
    damp=[]
    for i in np.arange(0.5,1,0.1):
        af = AffinityPropagation(damping=i,affinity='precomputed').fit(dist)
        label=len(np.unique(af.labels_))
        sil=metrics.silhouette_score(X=dist,metric ='precomputed', labels=af.labels_)
        WCD=inner_distance(dist, af.labels_)
        BCD_matrix = outer_distance(dist, af.labels_)
        BCD = np.mean(BCD_matrix)
        line=[i,label,sil,WCD,BCD]
        damp.append(line)
    return damp

# calculate within-cluster dispersion
def inner_distance(dist_matrix,sample_label):
    labels=np.unique(sample_label)
    sample_label=pd.DataFrame(data=sample_label,columns=['lab'])
    inner_dist=0
    for label in labels:
        group_dist=0
        ind=sample_label[sample_label.lab==label].index
        size=len(ind)
        for x in itertools.combinations(ind,2):
            group_dist += dist_matrix[x[0],x[1]]
        group_dist=group_dist/((size-1)/2)
        inner_dist += group_dist
    inner_dist=inner_dist/len(sample_label)
    return inner_dist

def outer_distance(dist_matrix,sample_label):
    labels=np.unique(sample_label)
    sample_label=pd.DataFrame(data=sample_label,columns=['lab'])
    BCD=[]
    for i in range(len(labels)-1):
        group1=sample_label[sample_label.lab==labels[i]].index
        for j in range(i+1,len(labels)):
            group2=sample_label[sample_label.lab==labels[j]].index
            dist=0
            for x in itertools.product(group1,group2):
                dist += dist_matrix[x[0],x[1]]
            BCD.append(dist/(len(group1)*len(group2)))
    return BCD

WCD=inner_distance(simi,af.labels_)
BCD_matrix=outer_distance(simi,af.labels_)
BCD=np.mean(BCD_matrix)

def search_method(dist_m,list):
    for m in list:
        Z = linkage(dist_m, m)
        c, coph_dists = cophenet(Z, dist_m)
        print "%s : %f" %(m,c)

search_method(BCD_matrix,['single','complete','average','weighted'])
Z = linkage(BCD_matrix,'average')
dn=dendrogram(Z)

# def h_cluster(dist_m,method='single'):
#     Z = linkage(dist_m, method)
#     # fig = plt.figure(figsize=(25, 10))
#     # dn = dendrogram(Z)
#     plt.title('Hierarchical Clustering Dendrogram (truncated)')
#     plt.xlabel('sample index or (cluster size)')
#     plt.ylabel('distance')
#     dendrogram(
#         Z,
#         truncate_mode='lastp',  # show only the last p merged clusters
#         p=12,  # show only the last p merged clusters
#         leaf_rotation=90.,
#         leaf_font_size=12.,
#         show_contracted=True,  # to get a distribution impression in truncated branches
#     )
#     plt.show()
#     c, coph_dists = cophenet(Z, BCD_matrix)
#     print "Cophenetic Correlation Coefficient:",c
#     return Z

def Elbow(Z,lastp=10):
    last = Z[-lastp:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print "clusters:", k

Elbow(Z,lastp=10)
incons = inconsistent(Z, 5)
fcluster(Z, 6, criterion='maxclust')