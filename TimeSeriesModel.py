#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import mysql.connector
import pandas as pd
import numpy as np
import math
import scipy.stats
import scipy.signal
from scipy.stats import norm,linregress

########################Get raw data#######################
RAWBILL = ('''SELECT buyer_name,seller_name,seller_id,order_no,bill_time,
bill_amount,proof_no,proj_no,match_pay_time,paid_amount,due_time,is_all_paid
FROM t1.m_core_bill_match where m_version = %(data_version)s and length(seller_name)=32;''')

RAWPAY = ('''SELECT core_name,chain_name,coreerpchainuid,
pay_uuid,order_no,pay_time,amount
FROM t1.m_pay_row where m_version = %(data_version)s and length(chain_name)=32;''')

cnx = mysql.connector.connect(user='t1', password='t1@DCFOffLine#8932%',
                              host='offlinecentre.cigru3mivzpd.rds.cn-north-1.amazonaws.com.cn',
                              database='t1',
                              use_unicode=True)

data_version = '20170105200000'
rbill = pd.DataFrame(pd.read_sql(RAWBILL,con=cnx,params={'data_version':data_version}))
rpay = pd.DataFrame(pd.read_sql(RAWPAY,con=cnx,params={'data_version':data_version}))

##############
rbill.rename(columns={'seller_name':'chain_name','buyer_name':'core_name',
                      'buyer_id':'chain_id','bill_time':'time','bill_amount':'amount'},inplace=True)
Bill = rbill[['core_name','chain_name','order_no','time','amount']]
rpay.rename(columns={'coreerpchainuid':'chain_id','pay_time':'time'},inplace=True)
Pay = rpay[['core_name','chain_name','order_no','time','amount']]
Trade = Bill.append(Pay)

CaseMaxDate = Trade['time'].max()
CaseMinDate = Trade['time'].min()

###################Get Time Series###############
'''
########################read_data#######################
RAW = ('SELECT buyer_group_name,seller_name,seller_id,order_no,bill_time,
bill_amount,proof_no,proj_no,match_pay_time,paid_amount,due_time,is_all_paid
FROM t1.m_core_bill_match where m_version = %(data_version)s;')

cnx = mysql.connector.connect(user='t1', password='t1@DCFOffLine#8932%',
                              host='offlinecentre.cigru3mivzpd.rds.cn-north-1.amazonaws.com.cn',
                              database='t1',
                              use_unicode=True)

data_version = '20170105200000'
rdata = pd.DataFrame(pd.read_sql(RAW,con=cnx,params={'data_version':data_version}))

#####################data_prepare#########################
rdata.rename(columns={'buyer_group_name':'core_name','seller_name':'chain_name','seller_id':'chain_id'},inplace=True)
Bill = rdata[['core_name','chain_name','order_no','bill_time','bill_amount']]

BillChain = Bill.groupby(['chain_name'])['order_no'].count().reset_index()
BillChain = BillChain.drop(['order_no'],1)
#BillCore = Bill.groupby(['core_name'])['order_no'].count().reset_index()
#BillCore = BillCore.drop(['order_no'],1)

CaseMaxDate = Bill['bill_time'].max() + DateOffset(months=1)

#Time Series #Bill.bill_time.dt.to_period("M")
CoreBillTimeSeries = pd.DataFrame(Bill.loc[:,['bill_time','bill_amount']].resample('M',on='bill_time').sum().reset_index())
CoreSeries = list(CoreBillTimeSeries['bill_amount'].fillna(0)[:])

BillTime_Series = Bill.loc[:,['chain_name','bill_time','bill_amount']].groupby(['chain_name']).resample('M',on='bill_time').sum().reset_index
BillTime_Series = pd.DataFrame(BillTime_Series())

BillTimeSeries = pd.DataFrame(columns=BillTime_Series.columns)
for i in BillChain.index:
    chain = BillChain['chain_name'].iloc[i]
    ChainBillTime_Series = pd.Series(list(BillTime_Series[BillTime_Series['chain_name'] == chain]['bill_amount']),index=list(BillTime_Series[BillTime_Series['chain_name'] == chain]['bill_time']))
    ChainMinDate = BillTime_Series[BillTime_Series['chain_name'] == chain]['bill_time'].min()
    Chain_BillTimeSeries = pd.Series(ChainBillTime_Series,index=pd.date_range(ChainMinDate,CaseMaxDate,freq='M'))
    ChainBillTimeSeries = pd.DataFrame({'chain_name':[chain]*len(Chain_BillTimeSeries), 'bill_time':Chain_BillTimeSeries.index.tolist(), 'bill_amount':Chain_BillTimeSeries.values.tolist()})
    BillTimeSeries = BillTimeSeries.append(ChainBillTimeSeries,ignore_index=True)
'''
ChainTimeSeries = pd.read_csv('ChainTimeSeries.csv',encoding='gbk',index_col=0)
CoreTimeSeries = pd.read_csv('CoreTimeSeries.csv',encoding='gbk',index_col=0)

ChainTimeSeries.drop('level_1',inplace=True,axis=1)
ChainTimeSeries.rename(columns={'level_0':'characteristics'},inplace=True)
Chain = ChainTimeSeries['chain_name'].unique()
Characteristics = ChainTimeSeries['characteristics'].unique()
CoreTimeSeries.drop('level_1',inplace=True,axis=1)
CoreTimeSeries.rename(columns={'level_0':'characteristics'},inplace=True)
CaseMonthDuration = len(CoreTimeSeries[CoreTimeSeries['characteristics']=='BillAmtMonthTs'])
CaseQuaterDuration = len(CoreTimeSeries[CoreTimeSeries['characteristics']=='BillAmtQuaterTs'])
CaseWeekDuration = len(CoreTimeSeries[CoreTimeSeries['characteristics']=='BillAmtWeekTs'])

#################TREND################
def Ztrend(ts,alpha):
    """计算Z值反映趋势显著性检验"""
    #计算Z值
    ts['value'] = ts['value'].fillna(0)
    if len(ts)<=1:
        return
#    Ndot = ts[ts[feature]!=0][feature].count()
#    Nlen = len(ts)
#    LenPct = float(Ndot)/float(Nlen)
#    if (LenPct < 0.8) & (Nlen < 24):
#        significant = None
#    else:
    r = []
    n = len(ts)
    for i in range(n - 1):
        k = 0
        j = i+1
        for s in range(j,n,1):
            if ts['value'].iloc[i] < ts['value'].iloc[s]:
                k = k + 1
        r.append(k)
    Z = 4*math.fsum(r)/(n*(n-1))-1

    #显著性水平
    if alpha == 0.05:
        ConZ = 1.96 * math.sqrt((4 * n + 10.0) / (9.0 * n * (n - 1)))
    elif alpha == 0.01:
        ConZ = 2.33 * math.sqrt((4 * n + 10.0) / (9.0 * n * (n - 1)))
    elif alpha == 0.1:
        ConZ = 1.65 * math.sqrt((4 * n + 10.0) / (9.0 * n * (n - 1)))

    if math.fabs(Z) > ConZ:
        significant = 1
    else:
        significant = 0
    return significant

def MKtrend(ts,alpha):
    ts['value'] = ts['value'].fillna(0)
#    Ndot = ts[ts[feature] != 0][feature].count()
#    Nlen = len(ts)
#    LenPct = float(Ndot)/float(Nlen)
#    if (LenPct < 0.8) & (Nlen < 24):
#        trend = None
#    else:
    n = len(ts)
    if n <= 1:
        return None,None
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign((ts['value'].iloc[j] - ts['value'].iloc[k]))
    unique_ts = np.unique(ts['value'])
    g = len(unique_ts)
    if n==g:
        var_s = (n*(n-1)*(2*n+5))/18
    else:
        tp = np.zeros(unique_ts.shape)
        for i in range(len(unique_ts)):
            tp[i] = sum(unique_ts[i] == ts['value'])
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s-1)/np.sqrt(var_s)
    elif s==0:
        z = 0
    elif s<0:
        z = (s+1)/np.sqrt(var_s)

    if(z<0):
        trend = 'Decrease'
    elif(z>0):
        trend = 'Incease'
    else:
        trend = 'No_trend'

#   test
    p = 2 * (1 - norm.cdf(abs(z)))  # 双尾检验
    h = abs(z) > norm.ppf(1 - alpha / 2)
    if (z < 0) and h:
        trendtest = 'decreasing'
    elif (z > 0) and h:
        trendtest = 'increasing'
    else:
        trendtest = 'no trend'

    return trend, trendtest

def Ltrend(ts,alpha):
    ts['value'] = ts['value'].fillna(0)
    if len(ts) <= 1:
        return
    slope,intercept,rvalue,pvalue,stderr = linregress(ts['value'],range(len(ts)))
    if (rvalue > 0) & (pvalue > alpha):
        trend = 'increase'
    elif (rvalue < 0) & (pvalue > alpha):
        trend = 'decrease'
    else:
        trend = 'no_trend'
    return trend

def MTT(ts,SL,alpha):
    ts['value'] = ts['value'].fillna(0)
    n = len(ts)
    if 2*SL > n:
        abrupt = None
    else:
        t_test = []
        p_value = []
        for i in range(n - (2*SL-1)):
            SubDatF = ts[i:i+SL]
            SubDatL = ts[i+SL:(i+2*SL)]
            Subt_test,Subp_value = scipy.stats.ttest_ind(SubDatF['value'],SubDatL['value'])
            t_test.append(Subt_test)
            p_value.append(Subp_value)
        MMTgt = any(i < alpha for i in p_value)
        if MMTgt:
            abrupt = 1
        else:
            abrupt = 0
    return abrupt

def Trend(TimeSeries,feature,SL=10,alpha=0.05):
    mkTrend = []
    liTrend = []
    testTrend = []
    Significant = []
    Abrupt = []
    ChainName = []
    for i in Chain:
        ChainTimeSeries = TimeSeries[TimeSeries['chain_name'] == i]
        significant = Ztrend(ChainTimeSeries,alpha)
        mktrend,trendtest = MKtrend(ChainTimeSeries,alpha)
        ltrend = Ltrend(ChainTimeSeries,alpha)
        abrupt = MTT(ChainTimeSeries,SL,alpha)
        mkTrend.append(mktrend)
        testTrend.append(trendtest)
        liTrend.append(ltrend)
        Significant.append(significant)
        Abrupt.append(abrupt)
        ChainName.append(i)

    trend = pd.DataFrame({'chain_name':ChainName,'feather':[feature] * len(ChainName),'ltrend':liTrend,
                          'mktrend':mkTrend,'mktrendtest':testTrend,'significant':Significant,'abrupt':Abrupt})
#    trend = trend.set_index('chain_name')
    return trend

Characteristics = ['BillAmtMonthTs','PayAmtMonthTs','BillAmtQuaterTs','PayAmtQuaterTs']
ResultTrend = pd.DataFrame(columns=['chain_name','feature','ltrend','mktrend','mktrendtest','significant',
                                    'abrupt'])
for feature in Characteristics:
    FeatureTimeSeries = ChainTimeSeries[ChainTimeSeries.characteristics==feature]
    trend = Trend(FeatureTimeSeries,feature)
    ResultTrend = ResultTrend.append(trend)

###############SEASON#####################
def MonthIndex(monthts,low=0.25,high=0.75):
    #####Data prepare#####
    monthts['value'] = monthts['value'].fillna(0)
    monthts['time'] = pd.to_datetime(monthts['time'])
    monthts['month'] = monthts['time'].map(lambda x: x.strftime('%m'))
    #    ts['year'] = ts['time'].map(lambda x: x.strftime('%Y'))
    ####Data clean######
#    MonInts = ts.groupby('year')['month'].count().reset_index()
#    MonInts['pct'] = MonInts['month'] / 12.0
#    Monts = list(MonInts[MonInts['pct']==1]['year']) #data all year
#    tsts = ts[ts[feature]!=0]
#    MonInYr = ts.groupby('year')['month'].count().reset_index()
#    MonInYr['pct'] = MonInYr['month']/12.0
#    tsYr = list(MonInYr[MonInYr['pct']>=0.5]['year']) #data in year
#    ts = ts[(ts['year'].isin(MonDat)) & (ts['year'].isin(DatYr))]
    ####Data process#####
#    Ndot = Dat[Dat[feature]!=0][feature].count()
#    Nlen = len(Dat)
#    LenPct = float(Ndot)/float(Nlen)
#    if (LenPct < 0.8) & (Nlen < 24):
#        season = np.nan
    AvgAmt = monthts['value'].mean()
    MonAmt = monthts.groupby('month')['value'].mean().reset_index()
    MonAmt['Index'] = MonAmt['value'] / AvgAmt
    LowIndex, HighIndex = MonAmt['Index'].quantile([low, high])
    MonAmt['type'] = np.where(MonAmt['Index']>=HighIndex,1,np.where(MonAmt['Index']<=LowIndex,-1,0))
    return MonAmt

def QuanterIndex(quaterts,low=0.5,high=0.5):
    quaterts['value'] = quaterts['value'].fillna(0)
    quaterts['time'] = pd.to_datetime(quaterts['time'])
    quaterts['month'] = quaterts['time'].map(lambda x: x.strftime('%m'))
    AvgAmt = quaterts['value'].mean()
    QuaAmt = quaterts.groupby('month')['value'].mean().reset_index()
    QuaAmt['Index'] = QuaAmt['value'] / AvgAmt
    LowIndex, HighIndex = QuaAmt['Index'].quantile([low, high])
    QuaAmt['type'] = np.where(QuaAmt['Index']>HighIndex,1,np.where(QuaAmt['Index']<LowIndex,-1,0))
    return QuaAmt

def Season(TimeSeries,CoreTimeSeries,feature,tm='quater',low=0.5,high=0.5):
    Index = pd.DataFrame(columns=['chain_name','value','index','type'])
    if tm == 'quater':
        IndexComputer = QuanterIndex
    elif tm == 'month':
        IndexComputer = MonthIndex
    else:
        print "error"

    ###Core Season####
    CoreIndex = IndexComputer(CoreTimeSeries,low,high)
    ###Chain Season###
    SeasonLow = []
    SeasonHigh = []
    chain_name = []
    Feature = []
    for i in Chain:
        chainTimeSeries = TimeSeries[TimeSeries['chain_name'] == i]
        ChainIndex = IndexComputer(chainTimeSeries,low,high)
        ChainIndex['chain_name'] = [i]*len(ChainIndex)
        ChainIndex['feather'] = [feature]*len(ChainIndex)
        Index = Index.append(ChainIndex)

        if tm == 'quater':
            if (ChainIndex['Index'][ChainIndex['type']==1].min() / ChainIndex['Index'][ChainIndex['type']==1].max()) >= 0.8:
                seasonhigh = ChainIndex['month'][ChainIndex['type']==1].values
            else:
                seasonhigh = ChainIndex['month'][(ChainIndex['Index']==(ChainIndex['Index'].max()))].values
            if (ChainIndex['Index'][ChainIndex['type']==0].min() / ChainIndex['Index'][ChainIndex['type']==0].max()) >= 0.8:
                seasonlow = ChainIndex['month'][ChainIndex['type']==0].values
            else:
                seasonlow = ChainIndex['month'][(ChainIndex['Index']==(ChainIndex['Index'].min()))].values
        elif tm == 'month':
            seasonhigh = ChainIndex['month'][ChainIndex['type']==1].values
            seasonlow = ChainIndex['month'][ChainIndex['type']==0].values
        SeasonHigh.append(seasonhigh)
        SeasonLow.append(seasonlow)
        Feature.append(feature)
        chain_name.append(i)
    season = pd.DataFrame({'chain_name': chain_name, 'season_high': SeasonHigh, 'season_low': SeasonLow,'feather':Feature})
#    season = season.set_index('chain_name')
    return season,CoreIndex,Index

Characteristics = ['BillAmtQuaterTs','PayAmtQuaterTs']
ResultSeason = pd.DataFrame(columns=['chain_name','season_high','season_low','feature'])
ResultSeasonIndex = pd.DataFrame(columns=['chain_name','feature','month','value','Index','type'])
CoreResultSeasonIndex = pd.DataFrame(columns=['feature','month','value','Index','type'])
for feature in Characteristics:
    FeatureTimeSeries = ChainTimeSeries[ChainTimeSeries.characteristics==feature]
    CoreFeatureTimeSeries = CoreTimeSeries[CoreTimeSeries.characteristics==feature]
    season,seasonindex,coreseasonindex = Season(FeatureTimeSeries,CoreFeatureTimeSeries,feature)
    ResultSeason = ResultSeason.append(season)
    ResultSeasonIndex = ResultSeasonIndex.append(seasonindex)
    CoreResultSeasonIndex = CoreResultSeasonIndex.append(coreseasonindex)

#################SPECTRUM POWER########################
def PSD(TimeSeries,CoreTimeSeries,feature):
    PP = pd.DataFrame(columns=['chain_name','cycle','cycle_spectrum'])
    for i in Chain:
        chainTimeSeries = TimeSeries[TimeSeries['chain_name'] == i]
        ChainSeries = list(chainTimeSeries['value'].fillna(0)[:])
        ChainFreq,ChainPowDen = scipy.signal.welch(ChainSeries,fs=1,detrend='linear',scaling='spectrum')
        ChainT = 1/ChainFreq
        ChainPP = pd.DataFrame({'chain_name':[i]*len(ChainT),'cycle':ChainT,'cycle_spectrum':ChainPowDen,'feature':[feature]* len(ChainT)})
        PP = PP.append(ChainPP)
    CoreSeries = list(CoreTimeSeries['value'].fillna(0)[:])
    CoreFreq,CorePowDen = scipy.signal.welch(CoreSeries,fs=1,detrend='linear',scaling='spectrum')
    CoreT = 1/CoreFreq
    CorePP = pd.DataFrame({'cycle':CoreT,'cycle_spectrum':CorePowDen,'feature':feature})
    return PP,CorePP

Characteristics = ['BillAmtMonthTs','PayAmtMonthTs']
ResultPP = pd.DataFrame(columns=['chain_name','feature','cycle','cycle_spectrum'])
CoreResultPP = pd.DataFrame(columns=['feature','cycle','cycle_spectrum'])
for feature in Characteristics:
    FeatureTimeSeries = ChainTimeSeries[ChainTimeSeries.characteristics==feature]
    CoreFeatureTimeSeries = CoreTimeSeries[CoreTimeSeries.characteristics==feature]
    pp,corepp = PSD(FeatureTimeSeries,CoreFeatureTimeSeries,feature)
    ResultPP = ResultPP.append(pp)
    CoreResultPP = CoreResultPP.append(corepp)

def Tdiscretion(psd,name='core'):
    cycle = []
    spectrum = []
    company = []
    for i in range(0,13):
        j = i + 0.5
        k = i - 0.5
        cycle.append(i)
        spectrum.append(psd[(psd['cycle']>k) & (psd['cycle']<j)]['cycle_spectrum'].sum())
        company.append(name)
    Discretion = pd.DataFrame({'chain_name':company,'cycle':cycle,'cycle_spectrum':spectrum})
    Discretion['cycle_spectrum_pct'] = Discretion['cycle_spectrum'] / Discretion['cycle_spectrum'].sum()
    return Discretion

def Ttype(tpsd,name='core'):
    type = ['intraseasonal','seasonal','intrahalfyear','halfyear','intrayear','year']
    spectrum = []
    spectrum_pct = []
    company = []
    for i in type:
        if i == 'intraseasonal':
            spectrum.append(tpsd[(tpsd['cycle']<3)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']<3)]['cycle_spectrum_pct'].sum())
            company.append(name)
        elif i == 'seasonal':
            spectrum.append(tpsd[(tpsd['cycle']==3)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']==3)]['cycle_spectrum_pct'].sum())
            company.append(name)
        elif i == 'intrahalfyear':
            spectrum.append(tpsd[(tpsd['cycle']>3) & (tpsd['cycle']<6)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']>3) & (tpsd['cycle']<6)]['cycle_spectrum_pct'].sum())
            company.append(name)
        elif i == 'halfyear':
            spectrum.append(tpsd[(tpsd['cycle']==6)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']==6)]['cycle_spectrum_pct'].sum())
            company.append(name)
        elif i == 'intrahalfyear':
            spectrum.append(tpsd[(tpsd['cycle']>6) & (tpsd['cycle']<12)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']>6) & (tpsd['cycle']<12)]['cycle_spectrum_pct'].sum())
            company.append(name)
        else:
            spectrum.append(tpsd[(tpsd['cycle']==12)]['cycle_spectrum'].sum())
            spectrum_pct.append(tpsd[(tpsd['cycle']==12)]['cycle_spectrum_pct'].sum())
            company.append(name)
    tType = pd.DataFrame({'chain_name':company,'type':type,'cycle_spectrum':spectrum,'cycle_spectrum_pct':spectrum_pct})
    return tType

def Cycle(Psd,feature):
    TPSD = pd.DataFrame(columns=['chain_name','cycle','cycle_spectrum','cycle_spectrum_pct'])
    TYPE = pd.DataFrame(columns=['chain_name','type','cycle_spectrum','cycle_spectrum_pct'])
    type1 = []
    type2 = []
    type3 = []
    chain_name = []
    Feature =[]
    for i in Chain:
        ChainPsd = Psd[Psd['chain_name'] == i]
        Chaintpsd = Tdiscretion(ChainPsd,i)
        ChainType = Ttype(Chaintpsd,i)
        TPSD.append(Chaintpsd)
        TYPE.append(ChainType)
        TPSD['feature'] = feature
        TYPE['feature'] = feature
#        Chaintpsd.sort_values('cycle_spectrum_pct', inplace=True, ascending=False)
        ChainType.sort_values('cycle_spectrum_pct', inplace=True, ascending=False)
        chain_name.append(i)
        type1.append((ChainType['type'].iloc[0],ChainType['cycle_spectrum_pct'].iloc[0]))
        type2.append((ChainType['type'].iloc[1],ChainType['cycle_spectrum_pct'].iloc[1]))
        type3.append((ChainType['type'].iloc[2],ChainType['cycle_spectrum_pct'].iloc[2]))
    cycle = pd.DataFrame({'chain_name':chain_name,'feature':[feature]*len(chain_name),'Type1':type1,'Type2':type2,'Type3':type3})
#    cycle = cycle.set_index('chain_name')
    return cycle,TPSD,TYPE

ResultCycle = pd.DataFrame(columns=['chain_name','Type1','Type2','Type3','feature'])
ResultTPSD = pd.DataFrame(columns=['chain_name','feature','cycle_spectrum','cycle_spectrum_pct','cycle'])
ResultTYPE = pd.DataFrame(columns=['chain_name','feature','cycle_spectrum','cycle_spectrum_pct','type'])
for feature in Characteristics:
    FeaturePP = ResultPP[ResultPP.feature==feature]
    cycle,tpsd,type = Cycle(FeaturePP,feature)
    ResultCycle = ResultCycle.append(cycle)
    ResultTPSD = ResultTPSD.append(tpsd)
    ResultTYPE = ResultTYPE.append(type)

################Distance##############
def Distance(Psd,CorePsd,SEASON,CoreSeason,feature):
    ###Euclidean distance####
    Coretpsd = Tdiscretion(CorePsd)
    chain_name = []
    psd_distance = []
#    season_distance = []
    for i in Chain:
        ChainPsd = Psd[Psd['chain_name'] == i]
#        ChainSeason = SEASON[SEASON['chain_name']==i]
        Chaintpsd = Tdiscretion(ChainPsd,i)
        chain_name.append(i)
        psd_distance.append(scipy.spatial.distance.euclidean(Coretpsd['cycle_spectrum_pct'],Chaintpsd['cycle_spectrum_pct']))
#        season_distance.append(scipy.spatial.distance.euclidean(CoreSeason['Index'], ChainSeason['Index']))
    distance = pd.DataFrame({'chain_name':chain_name,'feature':[feature]*len(chain_name),'psd_distance':psd_distance})
#    distance = distance.set_index('chain_name')
    return distance

ResultDistance = pd.DataFrame(columns=['chain_name','feature','psd_distance'])
for feature in Characteristics:
    FeaturePSD = ResultPP[ResultPP.feature==feature]
    FeatureSeasonIndex = ResultSeasonIndex[ResultSeasonIndex.feature==feature]
    distance = Distance(FeaturePSD,CoreResultPP,FeatureSeasonIndex,CoreResultSeasonIndex,feature)
    ResultDistance = ResultDistance.append(distance)

#############Feature Index############
def CaculateTsFeather(TimeSeries,tm='month'):
    ###Case duration###
    if tm == 'month':
        CaseDur = CaseMonthDuration
    elif tm == 'quater':
        CaseDur = CaseQuaterDuration
    #####Len#####
    TimeSeries['value'] = TimeSeries['value'].fillna(0)
    tslen = len(TimeSeries)
    TsWithData = TimeSeries[TimeSeries['value']!=0]
    tsdot = len(TsWithData)
    tslenpct = tslen*1.0 / CaseDur*1.0
    tsdotpct = tsdot*1.0 / CaseDur*1.0
    if tslen == 0:
        tsdotINts = None
    else:
        tsdotINts = tsdot*1.0 / tslen*1.0
    ####Season Integrit####
    TimeSeries['time'] = pd.to_datetime(TimeSeries['time'])
    TimeSeries['year'] = TimeSeries['time'].map(lambda x: x.strftime('%Y'))
    TimeSeries['month'] = TimeSeries['time'].map(lambda x: x.strftime('%m'))
    LenInTS = TimeSeries.groupby('year')['month'].count().reset_index()
    LenInTS['pct'] = LenInTS['month'] / 12.0
    tslenpctYavg = LenInTS['pct'].mean()
    TsWithData = TimeSeries[TimeSeries['value'] != 0]
    DotInTS = TsWithData.groupby('year')['month'].count().reset_index()
    DotInTS['pct'] = DotInTS['month'] / 12.0
    tsdotpctYavg = DotInTS['pct'].mean()
    return tslen,tsdot,tslenpct,tsdotpct,tsdotINts,tslenpctYavg,tsdotpctYavg

def CaculateRMFFeature(Dat):
    Dat['YM'] = Dat.time.map(lambda x: x.strftime('%Y-%m'))
    DatMinDate = Dat['time'].min()
    DatMaxDate = Dat['time'].max()
    spanmonth = (pd.to_datetime(DatMaxDate) - pd.to_datetime(DatMinDate)).days / 30
    spanquater = (pd.to_datetime(DatMaxDate) - pd.to_datetime(DatMinDate)).days / 90
    tsspanmonth = (pd.to_datetime(CaseMaxDate) - pd.to_datetime(DatMinDate)).days / 30
    wspanlastmonth = (pd.to_datetime(CaseMaxDate) - pd.to_datetime(DatMaxDate)).days / 30
    return spanmonth,spanquater,tsspanmonth,wspanlastmonth

def Feature(MonthTimeSeries,QuaterTimeSeries,Dat):
    feature = pd.DataFrame(columns=['mon_tslen','mon_tsdot','mon_tslenpct','mon_tsdotpct','mon_tsdotints',
                                    'mon_tslenpctYavg','mon_tsdotpctYavg','qua_tslen','qua_tsdot',
                                    'qua_tslenpct','qua_tsdotpct','qua_tsdotints','qua_tslenpctYavg',
                                    'qua_tsdotpctYavg','spanmonth','spanquater','tsspanmonth',
                                    'wspanlastmonth','chain_name'])
    for i in Chain:
        ChainMonthTimeSeries = MonthTimeSeries[MonthTimeSeries['chain_name'] == i]
        ChainQuaterTimeSeries = QuaterTimeSeries[QuaterTimeSeries['chain_name'] == i]
        ChainDat = Dat[Dat['chain_name'] == i]
        MonTsFeature = CaculateTsFeather(ChainMonthTimeSeries,'month')
        QuaTsFeature = CaculateTsFeather(ChainQuaterTimeSeries,'quater')
        RMFFeature = CaculateRMFFeature(ChainDat)
        chain_feature = list(MonTsFeature) + list(QuaTsFeature) + list(RMFFeature) + [i]
        feature = feature.append(
            pd.Series(chain_feature,
                      index=['mon_tslen','mon_tsdot','mon_tslenpct','mon_tsdotpct','mon_tsdotints',
                             'mon_tslenpctYavg','mon_tsdotpctYavg','qua_tslen','qua_tsdot','qua_tslenpct',
                             'qua_tsdotpct','qua_tsdotints','qua_tslenpctYavg','qua_tsdotpctYavg',
                             'spanmonth','spanquater','tsspanmonth','wspanlastmonth','chain_name'])
            ,ignore_index=True)
#    feature = feature.set_index('chain_name')
    return feature

BillMonthTs = ChainTimeSeries[ChainTimeSeries.characteristics=='BillAmtMonthTs']
BillQuaterTs = ChainTimeSeries[ChainTimeSeries.characteristics=='BillAmtQuaterTs']
ResultBillFeature = Feature(BillMonthTs,BillQuaterTs,Bill)
PayMonthTs = ChainTimeSeries[ChainTimeSeries.characteristics=='PayAmtMonthTs']
PayQuaterTs = ChainTimeSeries[ChainTimeSeries.characteristics=='PayAmtQuaterTs']
ResultPayFeature = Feature(PayMonthTs,PayQuaterTs,Pay)