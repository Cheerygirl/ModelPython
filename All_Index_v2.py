#!/usr/bin/python
# -*- coding: gbk -*
__author__ = 'Cheery'

import mysql.connector
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import *

########################read_data#######################
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

#####################rawdata_prepare#########################
rbill.rename(columns={'seller_name':'chain_name','buyer_name':'core_name',
                      'buyer_id':'chain_id','bill_time':'time','bill_amount':'amount'},inplace=True)
Bill = rbill[['core_name','chain_name','order_no','time','amount']]
ValidBill = Bill[Bill['amount']>0]
RefundBill = Bill[Bill['amount']<0]
rpay.rename(columns={'coreerpchainuid':'chain_id','pay_time':'time'},inplace=True)
Pay = rpay[['core_name','chain_name','order_no','time','amount']]
ValidPay = Pay[Pay['amount']>0]
RefundPay = Pay[Pay['amount']<0]
BillPay = rbill[['core_name','chain_name','order_no','time','amount','due_time','match_pay_time','paid_amount','is_all_paid']]
BillPay['pay_day'] = (BillPay['match_pay_time'] - BillPay['time']).dt.days
BillPay['due_day'] = (BillPay['due_time'] - BillPay['time']).dt.days
BillPay['pay_due_day'] = (BillPay['due_time'] - BillPay['match_pay_time']).dt.days
ValidBillPay = BillPay[BillPay['amount']>0]
Trade = Bill.append(Pay)

CaseMaxDate = Trade['time'].max()
CaseMinDate = Trade['time'].min()
Chain = Trade.groupby(['chain_name'])['order_no'].count().reset_index()
Chain = Chain.drop(['order_no'],1)

##########Gap##########
def Gap(Dat,characteristic,duplicate=True):
    DatChain = np.unique(Dat['chain_name'])
    GapDat = pd.DataFrame(columns=['chain_name',characteristic,'Gap'])
    if duplicate:
        for i in DatChain:
            ChainDat = Dat[Dat['chain_name'] == i]
            GapChainDat = ChainDat.sort_values([characteristic]).reset_index()
            GapChainDat[characteristic] = pd.to_datetime(GapChainDat[characteristic])
            GapChainDat['Gap'] = (GapChainDat[characteristic] - GapChainDat[characteristic].shift(1)).dt.days
            GapDat = GapDat.append(GapChainDat[['chain_name', characteristic, 'Gap']], ignore_index=True)
    else:
        for i in DatChain:
            ChainDat = Dat[Dat['chain_name'] == i]
            GapChainDat = ChainDat.drop_duplicates([characteristic]).sort_values([characteristic]).reset_index()
            GapChainDat[characteristic] = pd.to_datetime(GapChainDat[characteristic])
            GapChainDat['Gap'] = (pd.to_datetime(GapChainDat[characteristic]) - pd.to_datetime(GapChainDat[characteristic]).shift(1)).dt.days
            GapDat = GapDat.append(GapChainDat[['chain_name', characteristic, 'Gap']], ignore_index=True)
    return GapDat

BillGapDatU = Gap(Bill,'time',duplicate=False)
BillGapDat = Gap(Bill,'time',duplicate=True)
PayGapDatU = Gap(Pay,'time',duplicate=False)
PayGapDat = Gap(Pay,'time',duplicate=True)

##Transform to Time Series       #/////Bill.bill_time.dt.to_period("M")/////
def Trans_ts(Dat,act='avg',feature='amount',freq='M',tm='time',maxdate=CaseMaxDate):
    Dat = Dat[~pd.isnull(Dat[tm])]
    if act == 'sum':
        ts_df_o = Dat.loc[:, [tm, feature]].resample(freq,on=tm).sum().reset_index
    elif act == 'avg':
        ts_df_o = Dat.loc[:, [tm, feature]].resample(freq, on=tm).mean().reset_index
    elif act == 'max':
        ts_df_o = Dat.loc[:, [tm, feature]].resample(freq, on=tm).max()[feature].reset_index
    elif act == 'min':
        ts_df_o = Dat.loc[:, [tm, feature]].resample(freq, on=tm).min()[feature].reset_index

    ts_df_o = pd.DataFrame(ts_df_o())
    ts_s_o = pd.Series(list(ts_df_o[feature]),index=list(ts_df_o[tm]))
    MaxDate = maxdate
    MinDate = Dat[tm].min()
    ts_s = pd.Series(ts_s_o, index=pd.date_range(MinDate, MaxDate, freq=freq))
    return ts_s

def Get_ts(Dat,act='avg',feature='amount',freq='M',tm='time',maxdate=CaseMaxDate):
    DatChain = np.unique(Dat['chain_name'])
    ChainTs = pd.DataFrame(columns=['chain_name','time','value'])
    for i in DatChain:
        chain_dat = Dat[Dat['chain_name'] == i]
        chain_dat = chain_dat[~pd.isnull(chain_dat[tm])]
        if len(chain_dat) < 1:
            continue
        else:
            chain_ts_s = Trans_ts(chain_dat,act,feature,freq,tm,maxdate)
            chain_ts = pd.DataFrame({'chain_name': [i] * len(chain_ts_s),
                                    'time': chain_ts_s.index.tolist(), 'value': chain_ts_s.values.tolist()})
            ChainTs = ChainTs.append(chain_ts)
    return ChainTs

CaseBillMaxDate = Bill['time'].max() + DateOffset(months=1)
CasePayMaxDate = Pay['time'].max() + DateOffset(months=1)
CaseMatchPayMaxDate = BillPay['match_pay_time'].max() + DateOffset(months=1)
BillAmtMonthTs = Get_ts(Bill,'sum','amount','M','time',CaseBillMaxDate)
BillAmtAvgMonthTs = Get_ts(Bill,'avg','amount','M','time',CaseBillMaxDate)
BillAmtMaxMonthTs = Get_ts(Bill,'max','amount','M','time',CaseBillMaxDate)
BillAmtMinMonthTs = Get_ts(Bill,'min','amount','M','time',CaseBillMaxDate)

BillGapAvgMonthTs = Get_ts(BillGapDatU,'avg','Gap','M','time',CaseBillMaxDate)
BillGapMaxMonthTs = Get_ts(BillGapDatU,'max','Gap','M','time',CaseBillMaxDate)
BillGapMinMonthTs = Get_ts(BillGapDatU,'min','Gap','M','time',CaseBillMaxDate)

PayAmtMonthTs = Get_ts(Pay,'sum','amount','M','time',CasePayMaxDate)
PayAmtAvgMonthTs = Get_ts(Pay,'avg','amount','M','time',CasePayMaxDate)
PayAmtMaxMonthTs = Get_ts(Pay,'max','amount','M','time',CasePayMaxDate)
PayAmtMinMonthTs = Get_ts(Pay,'min','amount','M','time',CasePayMaxDate)

PayGapAvgMonthTs = Get_ts(PayGapDatU,'avg','Gap','M','time',CasePayMaxDate)
PayGapMaxMonthTs = Get_ts(PayGapDatU,'max','Gap','M','time',CasePayMaxDate)
PayGapMinMonthTs = Get_ts(PayGapDatU,'min','Gap','M','time',CasePayMaxDate)

PayDayAvgMonthTs = Get_ts(BillPay,'avg','pay_day','M','match_pay_time',CaseMatchPayMaxDate)
PayDayMaxMonthTs = Get_ts(BillPay,'max','pay_day','M','match_pay_time',CaseMatchPayMaxDate)
PayDayMinMonthTs = Get_ts(BillPay,'min','pay_day','M','match_pay_time',CaseMatchPayMaxDate)

BillAmtQuaterTs = Get_ts(Bill,'sum','amount','Q','time',CaseBillMaxDate)
BillAmtAvgQuaterTs = Get_ts(Bill,'avg','amount','Q','time',CaseBillMaxDate)
BillAmtMaxQuaterTs = Get_ts(Bill,'max','amount','Q','time',CaseBillMaxDate)
BillAmtMinQuaterTs = Get_ts(Bill,'min','amount','Q','time',CaseBillMaxDate)

BillGapAvgQuaterTs = Get_ts(BillGapDatU,'avg','Gap','Q','time',CaseBillMaxDate)
BillGapMaxQuaterTs = Get_ts(BillGapDatU,'max','Gap','Q','time',CaseBillMaxDate)
BillGapMinQuaterTs = Get_ts(BillGapDatU,'min','Gap','Q','time',CaseBillMaxDate)

PayAmtQuaterTs = Get_ts(Pay,'sum','amount','Q','time',CasePayMaxDate)
PayAmtAvgQuaterTs = Get_ts(Pay,'avg','amount','Q','time',CasePayMaxDate)
PayAmtMaxQuaterTs = Get_ts(Pay,'max','amount','Q','time',CasePayMaxDate)
PayAmtMinQuaterTs = Get_ts(Pay,'min','amount','Q','time',CasePayMaxDate)

PayGapAvgQuaterTs = Get_ts(PayGapDatU,'avg','Gap','Q','time',CasePayMaxDate)
PayGapMaxQuaterTs = Get_ts(PayGapDatU,'max','Gap','Q','time',CasePayMaxDate)
PayGapMinQuaterTs = Get_ts(PayGapDatU,'min','Gap','Q','time',CasePayMaxDate)

PayDayAvgQuaterTs = Get_ts(BillPay,'avg','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)
PayDayMaxQuaterTs = Get_ts(BillPay,'max','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)
PayDayMinQuaterTs = Get_ts(BillPay,'min','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)

BillAmtWeekTs = Get_ts(Bill,'sum','amount','W','time',CaseBillMaxDate)
BillAmtAvgWeekTs = Get_ts(Bill,'avg','amount','W','time',CaseBillMaxDate)
BillAmtMaxWeekTs = Get_ts(Bill,'max','amount','W','time',CaseBillMaxDate)
BillAmtMinWeekTs = Get_ts(Bill,'min','amount','W','time',CaseBillMaxDate)

BillGapAvgWeekTs = Get_ts(BillGapDatU,'avg','Gap','W','time',CaseBillMaxDate)
BillGapMaxWeekTs = Get_ts(BillGapDatU,'max','Gap','W','time',CaseBillMaxDate)
BillGapMinWeekTs = Get_ts(BillGapDatU,'min','Gap','W','time',CaseBillMaxDate)

PayAmtWeekTs = Get_ts(Pay,'sum','amount','W','time',CasePayMaxDate)
PayAmtAvgWeekTs = Get_ts(Pay,'avg','amount','W','time',CasePayMaxDate)
PayAmtMaxWeekTs = Get_ts(Pay,'max','amount','W','time',CasePayMaxDate)
PayAmtMinWeekTs = Get_ts(Pay,'min','amount','W','time',CasePayMaxDate)

PayGapAvgWeekTs = Get_ts(PayGapDatU,'avg','Gap','W','time',CasePayMaxDate)
PayGapMaxWeekTs = Get_ts(PayGapDatU,'max','Gap','W','time',CasePayMaxDate)
PayGapMinWeekTs = Get_ts(PayGapDatU,'min','Gap','W','time',CasePayMaxDate)

PayDayAvgWeekTs = Get_ts(BillPay,'avg','pay_day','W','match_pay_time',CaseMatchPayMaxDate)
PayDayMaxWeekTs = Get_ts(BillPay,'max','pay_day','W','match_pay_time',CaseMatchPayMaxDate)
PayDayMinWeekTs = Get_ts(BillPay,'min','pay_day','W','match_pay_time',CaseMatchPayMaxDate)

BillAmtYrTs = Get_ts(Bill,'sum','amount','A','time',CaseBillMaxDate)
BillAmtAvgYrTs = Get_ts(Bill,'avg','amount','A','time',CaseBillMaxDate)
BillAmtMaxYrTs = Get_ts(Bill,'max','amount','A','time',CaseBillMaxDate)
BillAmtMinYrTs = Get_ts(Bill,'min','amount','A','time',CaseBillMaxDate)

BillGapAvgYrTs = Get_ts(BillGapDatU,'avg','Gap','A','time',CaseBillMaxDate)
BillGapMaxYrTs = Get_ts(BillGapDatU,'max','Gap','A','time',CaseBillMaxDate)
BillGapMinYrTs = Get_ts(BillGapDatU,'min','Gap','A','time',CaseBillMaxDate)

PayAmtYrTs = Get_ts(Pay,'sum','amount','A','time',CasePayMaxDate)
PayAmtAvgYrTs = Get_ts(Pay,'avg','amount','A','time',CasePayMaxDate)
PayAmtMaxYrTs = Get_ts(Pay,'max','amount','A','time',CasePayMaxDate)
PayAmtMinYrTs = Get_ts(Pay,'min','amount','A','time',CasePayMaxDate)

PayGapAvgYrTs = Get_ts(PayGapDatU,'avg','Gap','A','time',CasePayMaxDate)
PayGapMaxYrTs = Get_ts(PayGapDatU,'max','Gap','A','time',CasePayMaxDate)
PayGapMinYrTs = Get_ts(PayGapDatU,'min','Gap','A','time',CasePayMaxDate)

PayDayAvgYrTs = Get_ts(BillPay,'avg','pay_day','A','match_pay_time',CaseMatchPayMaxDate)
PayDayMaxYrTs = Get_ts(BillPay,'max','pay_day','A','match_pay_time',CaseMatchPayMaxDate)
PayDayMinYrTs = Get_ts(BillPay,'min','pay_day','A','match_pay_time',CaseMatchPayMaxDate)

TsFrames = [BillAmtMonthTs,BillAmtAvgMonthTs,BillAmtMaxMonthTs,BillAmtMinMonthTs,BillGapAvgMonthTs,BillGapMaxMonthTs,
            BillGapMinMonthTs,PayAmtMonthTs,PayAmtAvgMonthTs,PayAmtMaxMonthTs,PayAmtMinMonthTs,PayGapAvgMonthTs,
            PayGapMaxMonthTs,PayGapMinMonthTs,PayDayAvgMonthTs,PayDayMaxMonthTs,PayDayMinMonthTs,BillAmtWeekTs,
            BillAmtAvgWeekTs, BillAmtMaxWeekTs,BillAmtMinWeekTs, BillGapAvgWeekTs, BillGapMaxWeekTs,BillGapMinWeekTs,
            PayAmtWeekTs, PayAmtAvgWeekTs,PayAmtMaxWeekTs, PayAmtMinWeekTs,PayGapAvgWeekTs, PayGapMaxWeekTs,
            PayGapMinWeekTs,PayDayAvgWeekTs,PayDayMaxWeekTs, PayDayMinWeekTs,BillAmtQuaterTs, BillAmtAvgQuaterTs,
            BillAmtMaxQuaterTs, BillAmtMinQuaterTs,BillGapAvgQuaterTs, BillGapMaxQuaterTs,BillGapMinQuaterTs,
            PayAmtQuaterTs, PayAmtAvgQuaterTs, PayAmtMaxQuaterTs,PayAmtMinQuaterTs,PayGapAvgQuaterTs, PayGapMaxQuaterTs,
            PayGapMinQuaterTs,PayDayAvgQuaterTs,PayDayMaxQuaterTs, PayDayMinQuaterTs, BillAmtYrTs,BillAmtAvgYrTs,
            BillAmtMaxYrTs, BillAmtMinYrTs, BillGapAvgYrTs, BillGapMaxYrTs,BillGapMinYrTs, PayAmtYrTs, PayAmtAvgYrTs,
            PayAmtMaxYrTs,PayAmtMinYrTs,PayGapAvgYrTs, PayGapMaxYrTs, PayGapMinYrTs,PayDayAvgYrTs, PayDayMaxYrTs,
            PayDayMinYrTs]

names = ['BillAmtMonthTs','BillAmtAvgMonthTs','BillAmtMaxMonthTs','BillAmtMinMonthTs',
         'BillGapAvgMonthTs','BillGapMaxMonthTs','BillGapMinMonthTs','PayAmtMonthTs',
         'PayAmtAvgMonthTs','PayAmtMaxMonthTs','PayAmtMinMonthTs','PayGapAvgMonthTs',
         'PayGapMaxMonthTs','PayGapMinMonthTs','PayDayAvgMonthTs','PayDayMaxMonthTs',
         'PayDayMinMonthTs','BillAmtWeekTs', 'BillAmtAvgWeekTs', 'BillAmtMaxWeekTs',
         'BillAmtMinWeekTs', 'BillGapAvgWeekTs', 'BillGapMaxWeekTs','BillGapMinWeekTs',
         'PayAmtWeekTs', 'PayAmtAvgWeekTs','PayAmtMaxWeekTs', 'PayAmtMinWeekTs','PayGapAvgWeekTs',
         'PayGapMaxWeekTs','PayGapMinWeekTs','PayDayAvgWeekTs','PayDayMaxWeekTs', 'PayDayMinWeekTs',
         'BillAmtQuaterTs', 'BillAmtAvgQuaterTs','BillAmtMaxQuaterTs', 'BillAmtMinQuaterTs',
         'BillGapAvgQuaterTs', 'BillGapMaxQuaterTs','BillGapMinQuaterTs','PayAmtQuaterTs',
         'PayAmtAvgQuaterTs', 'PayAmtMaxQuaterTs','PayAmtMinQuaterTs','PayGapAvgQuaterTs',
         'PayGapMaxQuaterTs', 'PayGapMinQuaterTs','PayDayAvgQuaterTs','PayDayMaxQuaterTs',
         'PayDayMinQuaterTs', 'BillAmtYrTs','BillAmtAvgYrTs','BillAmtMaxYrTs', 'BillAmtMinYrTs',
         'BillGapAvgYrTs', 'BillGapMaxYrTs','BillGapMinYrTs', 'PayAmtYrTs', 'PayAmtAvgYrTs',
         'PayAmtMaxYrTs','PayAmtMinYrTs','PayGapAvgYrTs', 'PayGapMaxYrTs', 'PayGapMinYrTs',
         'PayDayAvgYrTs', 'PayDayMaxYrTs','PayDayMinYrTs']

TimeSeries = pd.concat(TsFrames,keys=names)
Ts = TimeSeries.reset_index()
Ts.to_csv('TimeSeries.csv',encoding='gbk')

#######Ts Index############
def Ts_feature(TsDat):
    ChainAvg = TsDat.groupby(['chain_name'])['value'].mean()
    ChainStd = TsDat.groupby(['chain_name'])['value'].std()
    ChainMax = TsDat.groupby(['chain_name'])['value'].max()
    ChainMin = TsDat.groupby(['chain_name'])['value'].min()
    ChainSum = TsDat.groupby(['chain_name'])['value'].sum()
    TsIndex = pd.DataFrame({'Avg': ChainAvg, 'Std': ChainStd,'Max': ChainMax, 'Min': ChainMin,
                            'Sum': ChainSum})
    return TsIndex

BillAmtMonthIndex = Ts_feature(BillAmtMonthTs)
BillAmtAvgMonthIndex = Ts_feature(BillAmtAvgMonthTs)
BillAmtMaxMonthIndex = Ts_feature(BillAmtMaxMonthTs)
BillAmtMinMonthIndex = Ts_feature(BillAmtMinMonthTs)

BillGapAvgMonthIndex = Ts_feature(BillGapAvgMonthTs)
BillGapMaxMonthIndex = Ts_feature(BillGapMaxMonthTs)
BillGapMinMonthIndex = Ts_feature(BillGapMinMonthTs)

PayAmtMonthIndex = Ts_feature(PayAmtMonthTs)
PayAmtAvgMonthIndex = Ts_feature(PayAmtAvgMonthTs)
PayAmtMaxMonthIndex = Ts_feature(PayAmtMaxMonthTs)
PayAmtMinMonthIndex = Ts_feature(PayAmtMinMonthTs)

PayGapAvgMonthIndex = Ts_feature(PayGapAvgMonthTs)
PayGapMaxMonthIndex = Ts_feature(PayGapMaxMonthTs)
PayGapMinMonthIndex = Ts_feature(PayGapMinMonthTs)

PayDayAvgMonthIndex = Ts_feature(PayDayAvgMonthTs)
PayDayMaxMonthIndex = Ts_feature(PayDayMaxMonthTs)
PayDayMinMonthIndex = Ts_feature(PayDayMinMonthTs)

BillAmtQuaterIndex = Ts_feature(BillAmtQuaterTs)
BillAmtAvgQuaterIndex = Ts_feature(BillAmtAvgQuaterTs)
BillAmtMaxQuaterIndex = Ts_feature(BillAmtMaxQuaterTs)
BillAmtMinQuaterIndex = Ts_feature(BillAmtMinQuaterTs)

BillGapAvgQuaterIndex = Ts_feature(BillGapAvgQuaterTs)
BillGapMaxQuaterIndex = Ts_feature(BillGapMaxQuaterTs)
BillGapMinQuaterIndex = Ts_feature(BillGapMinQuaterTs)

PayAmtQuaterIndex = Ts_feature(PayAmtQuaterTs)
PayAmtAvgQuaterIndex = Ts_feature(PayAmtAvgQuaterTs)
PayAmtMaxQuaterIndex = Ts_feature(PayAmtMaxQuaterTs)
PayAmtMinQuaterIndex = Ts_feature(PayAmtMinQuaterTs)

PayGapAvgQuaterIndex = Ts_feature(PayGapAvgQuaterTs)
PayGapMaxQuaterIndex = Ts_feature(PayGapMaxQuaterTs)
PayGapMinQuaterIndex = Ts_feature(PayGapMinQuaterTs)

PayDayAvgQuaterIndex = Ts_feature(PayDayAvgQuaterTs)
PayDayMaxQuaterIndex = Ts_feature(PayDayMaxQuaterTs)
PayDayMinQuaterIndex = Ts_feature(PayDayMinQuaterTs)

BillAmtWeekIndex = Ts_feature(BillAmtWeekTs)
BillAmtAvgWeekIndex = Ts_feature(BillAmtAvgWeekTs)
BillAmtMaxWeekIndex = Ts_feature(BillAmtMaxWeekTs)
BillAmtMinWeekIndex = Ts_feature(BillAmtMinWeekTs)

BillGapAvgWeekIndex = Ts_feature(BillGapAvgWeekTs)
BillGapMaxWeekIndex = Ts_feature(BillGapMaxWeekTs)
BillGapMinWeekIndex = Ts_feature(BillGapMinWeekTs)

PayAmtWeekIndex = Ts_feature(PayAmtWeekTs)
PayAmtAvgWeekIndex = Ts_feature(PayAmtAvgWeekTs)
PayAmtMaxWeekIndex = Ts_feature(PayAmtMaxWeekTs)
PayAmtMinWeekIndex = Ts_feature(PayAmtMinWeekTs)

PayGapAvgWeekIndex = Ts_feature(PayGapAvgWeekTs)
PayGapMaxWeekIndex = Ts_feature(PayGapMaxWeekTs)
PayGapMinWeekIndex = Ts_feature(PayGapMinWeekTs)

PayDayAvgWeekIndex = Ts_feature(PayDayAvgWeekTs)
PayDayMaxWeekIndex = Ts_feature(PayDayMaxWeekTs)
PayDayMinWeekIndex = Ts_feature(PayDayMinWeekTs)

BillAmtYrIndex = Ts_feature(BillAmtYrTs)
BillAmtAvgYrIndex = Ts_feature(BillAmtAvgYrTs)
BillAmtMaxYrIndex = Ts_feature(BillAmtMaxYrTs)
BillAmtMinYrIndex = Ts_feature(BillAmtMinYrTs)

BillGapAvgYrIndex = Ts_feature(BillGapAvgYrTs)
BillGapMaxYrIndex = Ts_feature(BillGapMaxYrTs)
BillGapMinYrIndex = Ts_feature(BillGapMinYrTs)

PayAmtYrIndex = Ts_feature(PayAmtYrTs)
PayAmtAvgYrIndex = Ts_feature(PayAmtAvgYrTs)
PayAmtMaxYrIndex = Ts_feature(PayAmtMaxYrTs)
PayAmtMinYrIndex = Ts_feature(PayAmtMinYrTs)

PayGapAvgYrIndex = Ts_feature(PayGapAvgYrTs)
PayGapMaxYrIndex = Ts_feature(PayGapMaxYrTs)
PayGapMinYrIndex = Ts_feature(PayGapMinYrTs)

PayDayAvgYrIndex = Ts_feature(PayDayAvgYrTs)
PayDayMaxYrIndex = Ts_feature(PayDayMaxYrTs)
PayDayMinYrIndex = Ts_feature(PayDayMinYrTs)

TsIndexFrame = [BillAmtMonthIndex,BillAmtAvgMonthIndex,BillAmtMaxMonthIndex,BillAmtMinMonthIndex,BillGapAvgMonthIndex,
                BillGapMaxMonthIndex,BillGapMinMonthIndex,PayAmtMonthIndex,PayAmtAvgMonthIndex,PayAmtMaxMonthIndex,
                PayAmtMinMonthIndex,PayGapAvgMonthIndex,PayGapMaxMonthIndex,PayGapMinMonthIndex,PayDayAvgMonthIndex,
                PayDayMaxMonthIndex,PayDayMinMonthIndex,BillAmtWeekIndex, BillAmtAvgWeekIndex, BillAmtMaxWeekIndex,
                BillAmtMinWeekIndex, BillGapAvgWeekIndex, BillGapMaxWeekIndex,BillGapMinWeekIndex,PayAmtWeekIndex,
                PayAmtAvgWeekIndex,PayAmtMaxWeekIndex, PayAmtMinWeekIndex,PayGapAvgWeekIndex, PayGapMaxWeekIndex,
                PayGapMinWeekIndex,PayDayAvgWeekIndex,PayDayMaxWeekIndex, PayDayMinWeekIndex,BillAmtQuaterIndex,
                BillAmtAvgQuaterIndex,BillAmtMaxQuaterIndex, BillAmtMinQuaterIndex,BillGapAvgQuaterIndex,
                BillGapMaxQuaterIndex,BillGapMinQuaterIndex,PayAmtQuaterIndex, PayAmtAvgQuaterIndex,
                PayAmtMaxQuaterIndex,PayAmtMinQuaterIndex,PayGapAvgQuaterIndex, PayGapMaxQuaterIndex,
                PayGapMinQuaterIndex,PayDayAvgQuaterIndex,PayDayMaxQuaterIndex, PayDayMinQuaterIndex, BillAmtYrIndex,
                BillAmtAvgYrIndex,BillAmtMaxYrIndex, BillAmtMinYrIndex, BillGapAvgYrIndex, BillGapMaxYrIndex,
                BillGapMinYrIndex, PayAmtYrIndex, PayAmtAvgYrIndex, PayAmtMaxYrIndex,PayAmtMinYrIndex,PayGapAvgYrIndex,
                PayGapMaxYrIndex, PayGapMinYrIndex,PayDayAvgYrIndex, PayDayMaxYrIndex,PayDayMinYrIndex]

TsIndex = pd.concat(TsIndexFrame,keys=names,axis=1)
TsIndex.columns = ['_'.join(col) for col in TsIndex.columns]

###########Duration#########
def Cal_Dr(TimeSeries,freq=30):
    ###Case duration###
    CaseDur = (CaseMaxDate-CaseMinDate).days / freq
    #####All Len#####
    TimeSeries['value'] = TimeSeries['value'].fillna(0)
    tslen = len(TimeSeries)
    TsWithData = TimeSeries[TimeSeries['value']!=0]
    tsdlen = len(TsWithData)
    tspct = tslen*1.0 / CaseDur*1.0
    tsdpct = tsdlen*1.0 / CaseDur*1.0
    tsdtspct = tsdlen*1.0 / tslen*1.0
    return CaseDur,tslen,tsdlen,tspct,tsdpct,tsdtspct

def Get_Dr(TimeSeries,freq=30):
    TsChain = np.unique(TimeSeries['chain_name'])
    Dur = pd.DataFrame(columns=['case_dur','ts_len','tsd_len','ts_pct','tsd_pct','tsdts_pct','chain_name'])
    for i in TsChain:
        ChainTimeSeries = TimeSeries[TimeSeries['chain_name'] == i]
        ChainDur = list(Cal_Dr(ChainTimeSeries,freq))
        ChainDur.append(i)
        Dur = Dur.append(pd.Series(ChainDur,index=['case_dur','ts_len','tsd_len','ts_pct','tsd_pct','tsdts_pct','chain_name']),ignore_index=True)
    Dur = Dur.set_index('chain_name')
    return Dur

BillAmtMonthDr = Get_Dr(BillAmtMonthTs,30)
BillGapMonthDr = Get_Dr(BillGapAvgMonthTs,30)
PayAmtMonthDr = Get_Dr(PayAmtMonthTs,30)
PayGapMonthDr = Get_Dr(PayGapAvgMonthTs,30)
PayDayMonthDr = Get_Dr(PayDayAvgMonthTs,30)

BillAmtQuaterDr = Get_Dr(BillAmtQuaterTs,90)
BillGapAvgQuaterDr = Get_Dr(BillGapAvgQuaterTs,90)
PayAmtQuaterDr = Get_Dr(PayAmtQuaterTs,90)
PayGapAvgQuaterDr = Get_Dr(PayGapAvgQuaterTs,90)
PayDayAvgQuaterDr = Get_Dr(PayDayAvgQuaterTs,90)

BillAmtWeekDr = Get_Dr(BillAmtWeekTs,7)
BillGapAvgWeekDr = Get_Dr(BillGapAvgWeekTs,7)
PayAmtWeekDr = Get_Dr(PayAmtWeekTs,7)
PayGapAvgWeekDr = Get_Dr(PayGapAvgWeekTs,7)
PayDayAvgWeekDr = Get_Dr(PayDayAvgWeekTs,7)

BillAmtYrDr = Get_Dr(BillAmtYrTs,360)
BillGapAvgYrDr = Get_Dr(BillGapAvgYrTs,360)
PayAmtYrDr = Get_Dr(PayAmtYrTs,360)
PayGapAvgYrDr = Get_Dr(PayGapAvgYrTs,360)
PayDayAvgYrDr = Get_Dr(PayDayAvgYrTs,360)

DrFrames = [BillAmtMonthDr,BillGapMonthDr,PayAmtMonthDr,PayGapMonthDr,PayDayMonthDr,BillAmtQuaterDr,BillGapAvgQuaterDr,
            PayAmtQuaterDr,PayGapAvgQuaterDr,PayDayAvgQuaterDr, BillAmtWeekDr,BillGapAvgWeekDr,PayAmtWeekDr,
            PayGapAvgWeekDr,PayDayAvgWeekDr,BillAmtYrDr,BillGapAvgYrDr,PayAmtYrDr,PayGapAvgYrDr,PayDayAvgYrDr]

Drnames = ['BillAmtMonth','BillGapAvgMonth','PayAmtMonth','PayGapAvgMonth','PayDayAvgMonth','BillAmtWeek',
           'BillGapAvgWeek','PayAmtWeek', 'PayGapAvgWeek','PayDayAvgWeek','BillAmtQuater',
           'BillGapAvgQuater','PayAmtQuater',  'PayGapAvgQuater','PayDayAvgQuater',
           'BillAmtYr','BillGapAvgYr','PayAmtYr','PayGapAvgYr', 'PayDayAvgYr']

DrIndex = pd.concat(DrFrames,keys=Drnames,axis=1)
DrIndex.columns = ['_'.join(col) for col in DrIndex.columns]

#############Index Prepare################
def RecentDat(Dat, data_end_date,Tm):
    RecentPeriod = pd.to_datetime(data_end_date) - datetime.timedelta(Tm)
    cod = Dat.time > RecentPeriod
    DatRct = Dat[cod]
    return DatRct

def sigma(Dat,sig,characteristic):
    """caculate the percentage of bill amount with the characteristic in N*sig """
    Chain = np.unique(Dat['chain_name'])
    SigPer = []
    ChainName = []
    for i in Chain:
        ChainDat = Dat[Dat['chain_name']==i]
        Amtavg = np.nanmean(ChainDat.loc[:,characteristic])
        Amtstd = np.nanstd(ChainDat.loc[:,characteristic])
        cod = (ChainDat[characteristic] >= Amtavg - sig * Amtstd) & (ChainDat[characteristic] <= Amtavg + sig * Amtstd)
        sigChainDat = ChainDat[cod]
        sigChainSum = sigChainDat['amount'].sum()
        if ChainDat['amount'].sum() == 0:
            sigChainPer = 0
        else:
            sigChainPer = sigChainSum / ChainDat['amount'].sum()
            SigPer.append(sigChainPer)
            ChainName.append(i)
    DatSigper = pd.Series(SigPer, index=ChainName)
#    DatSigper = pd.Series({'ChainName' : ChainName, characteristic+'SigPer' : SigPer})
    return DatSigper

def RMF_RM(Dat,data_end_date,fs):
    CoreAmt = Dat['amount'].sum()
    CoreNum = Dat['amount'].count()
    ChainAmt = Dat.groupby(['chain_name'])['amount'].sum()
    ChainNum = Dat.groupby(['chain_name'])['amount'].count()

    Rct1yr = RecentDat(Dat,data_end_date,360)
    ChainAmt1Yr = Rct1yr.groupby(['chain_name'])['amount'].sum()
    RctHfyr = RecentDat(Dat,data_end_date,180)
    ChainAmtHfyr = RctHfyr.groupby(['chain_name'])['amount'].sum()
    RctQt = RecentDat(Dat,data_end_date,90)
    ChainAmtQt = RctQt.groupby(['chain_name'])['amount'].sum()

    ChainBillAvgamt = Dat['amount'].groupby(Dat['chain_name']).mean()
    ChainBillStdamt = Dat['amount'].groupby(Dat['chain_name']).std()
    ChainBillMaxamt = Dat['amount'].groupby(Dat['chain_name']).max()
    ChainBillMinamt = Dat['amount'].groupby(Dat['chain_name']).min()
    ChainBillCvamt = ChainBillStdamt / ChainBillAvgamt
    ChainBill75amt = Dat['amount'].groupby(Dat['chain_name']).quantile(0.75)
    ChainBill25amt = Dat['amount'].groupby(Dat['chain_name']).quantile(0.25)
    ChainBill50amt = Dat['amount'].groupby(Dat['chain_name']).quantile(0.5)
    ChainBillSkewamt = Dat['amount'].groupby(Dat['chain_name']).skew()
    ChainBillKurtamt = Dat['amount'].groupby(Dat['chain_name']).apply(pd.Series.kurt)
    ChainBillIqramt = ChainBill75amt - ChainBill25amt
    ChainAmt1sig = sigma(Dat,1,'amount')
    ChainAmt2sig = sigma(Dat,2,'amount')
    ChainAmt3sig = sigma(Dat,3,'amount')

    RmIndex = pd.DataFrame({fs+'Num': ChainNum, fs+'Amt': ChainAmt,
                             fs+'Avgamt': ChainBillAvgamt, fs+'Stdamt': ChainBillStdamt,
                             fs+'Maxamt': ChainBillMaxamt, fs+'Minamt': ChainBillMinamt,
                             fs+'Sig1': ChainAmt1sig, fs+'Sig2': ChainAmt2sig,
                             fs+'Sig3': ChainAmt3sig, fs+'RctAmt1yr': ChainAmt1Yr,
                             fs+'RctAmtHfyr': ChainAmtHfyr, fs+'RctAmtQt': ChainAmtQt,
                            fs+'Cvamt': ChainBillCvamt, fs+'75amt': ChainBill75amt, fs+'25amt': ChainBill25amt,
                            fs+'Kurtamt':ChainBillKurtamt, fs+'Skewamt':ChainBillSkewamt,fs+'IQRamt':ChainBillIqramt,
                            fs+'Medianamt':ChainBill50amt})
    return RmIndex

def RMF_RF(Dat,data_end_date,fs):
    ChainMinDate = Dat.groupby(['chain_name'])['time'].min()
    ChainMaxDate = Dat.groupby(['chain_name'])['time'].max()
    ChainSpan = (pd.to_datetime(ChainMaxDate) - pd.to_datetime(ChainMinDate)).dt.days
    DatMinDate = Dat['time'].min()
    DatMaxDate = Dat['time'].max()
    ChainWoSpanL = (pd.to_datetime(DatMaxDate) - pd.to_datetime(ChainMaxDate)).dt.days
    ChainWoSpanF = (pd.to_datetime(ChainMinDate) - pd.to_datetime(DatMinDate)).dt.days
    ChainSpanP = (ChainWoSpanF + ChainWoSpanL) *1.0 / ChainSpan
    ChainSpanW = ChainWoSpanL * 1.0 / (ChainWoSpanF+1.0)

    Dat['YMD'] = Dat.time.map(lambda x: x.strftime('%Y-%m-%d'))
    Dat['YM'] = Dat.time.map(lambda x: x.strftime('%Y-%m'))
    ChainDay = Dat.groupby(['chain_name'])['YMD'].nunique()
    ChainMonth = Dat.groupby(['chain_name'])['YM'].nunique()

    RFIndex = pd.DataFrame({fs+'ChainFirstDate':ChainMinDate,
                             fs+'ChainLastDate':ChainMaxDate,fs+'ChainSpan':ChainSpan,
                             fs+'ChainDay':ChainDay,fs+'ChainMonth':ChainMonth,
                             fs+'ChainWoSpanL':ChainWoSpanL,fs+'ChainWoSpanF':ChainWoSpanF,
                             fs+'ChainSpanRio': ChainSpanP, fs+'ChainSpanWRio': ChainSpanW})

    return RFIndex

def RMF_Gap(Gap,GapU,fs):

    ChainGapAvg = Gap.groupby(['chain_name'])['Gap'].mean()
    ChainGapMax = Gap.groupby(['chain_name'])['Gap'].max()
    ChainGapMin = Gap.groupby(['chain_name'])['Gap'].min()
    ChainGapStd = Gap.groupby(['chain_name'])['Gap'].std()

    ChainGapAvgU = GapU.groupby(['chain_name'])['Gap'].mean()
    ChainGapMaxU = GapU.groupby(['chain_name'])['Gap'].max()
    ChainGapMinU = GapU.groupby(['chain_name'])['Gap'].min()
    ChainGapStdU = GapU.groupby(['chain_name'])['Gap'].std()

    GapIndex = pd.DataFrame({fs+'ChainAvgGap':ChainGapAvg, fs+'ChainMaxGap':ChainGapMax,
                             fs+'ChainMinGap':ChainGapMin, fs+'ChainStdGap':ChainGapStd,
                             fs+'ChainAvgGapU':ChainGapAvgU, fs+'ChainMaxGapU':ChainGapMaxU,
                             fs+'ChainMinGapU':ChainGapMinU, fs+'ChainStdGapU':ChainGapStdU})
    return GapIndex

def RMF_Pay(Dat,is_credit=1):
    if is_credit == 1:
        Dat = Dat[(Dat['due_day'] >= 0) & ~pd.isnull(Dat['due_day'])]
        ChainCrAmt = Dat.groupby(['chain_name'])['amount'].sum()
        ChainCrNum = Dat.groupby(['chain_name'])['amount'].count()
    else:
        ChainCrAmt = pd.isnull()
        ChainCrNum = pd.isnull()
    ChainAvgDueDay = Dat.groupby(['chain_name'])['due_day'].mean()
    ChainMaxDueDay = Dat.groupby(['chain_name'])['due_day'].max()
    ChainMinDueDay = Dat.groupby(['chain_name'])['due_day'].min()

    PaidDat = Dat[~pd.isnull(Dat['pay_day'])]
    ChainAvgPayDay = PaidDat.groupby(['chain_name'])['pay_day'].mean()
    ChainAvgPayPeriod = ChainAvgPayDay / ChainAvgDueDay
    ChainMaxPayDay = PaidDat.groupby(['chain_name'])['pay_day'].max()
    ChainMaxPayPeriod = ChainMaxPayDay / ChainAvgDueDay
    ChainMinPayDay = PaidDat.groupby(['chain_name'])['pay_day'].min()
    ChainStdPayDay = PaidDat.groupby(['chain_name'])['pay_day'].std()
    ChainNumPayDay = PaidDat.groupby(['chain_name'])['amount'].count()
    ChainAmtPayDay = PaidDat.groupby(['chain_name'])['amount'].sum()
    ChainCvPayDay = ChainStdPayDay / ChainAvgPayDay
    Chain75PayDay = PaidDat['pay_day'].groupby(PaidDat['chain_name']).quantile(0.75)
    Chain25PayDay = PaidDat['pay_day'].groupby(PaidDat['chain_name']).quantile(0.25)
    Chain50PayDay = PaidDat['pay_day'].groupby(PaidDat['chain_name']).quantile(0.5)
    ChainSkewPayDay = PaidDat['pay_day'].groupby(PaidDat['chain_name']).skew()
    ChainKurtPayDay = PaidDat['pay_day'].groupby(PaidDat['chain_name']).apply(pd.Series.kurt)
    ChainIqrPayDay = Chain75PayDay - Chain25PayDay
    ChainPayDay1sig = sigma(PaidDat,1,'pay_day')
    ChainPayDay2sig = sigma(PaidDat,2,'pay_day')
    ChainPayDay3sig = sigma(PaidDat,3,'pay_day')

    PrePayPaidDat = PaidDat[PaidDat['pay_due_day'] >= 0]
    OverduePaidDat = PaidDat[PaidDat['pay_due_day'] < 0]
    ChainPrePayAmt = PrePayPaidDat.groupby(['chain_name'])['amount'].sum()
    ChainOverdueAmt = OverduePaidDat.groupby(['chain_name'])['amount'].sum()
    ChainOverdueAmtpct = ChainOverdueAmt / ChainCrAmt
    ChainPrePayNum = PrePayPaidDat.groupby(['chain_name'])['amount'].count()
    ChainOverdueNum = OverduePaidDat.groupby(['chain_name'])['amount'].count()
    ChainOverdueNumpct = ChainOverdueNum / ChainCrNum

    ChainAvgPrePayDay = PrePayPaidDat.groupby(['chain_name'])['pay_due_day'].mean()
    ChainAvgPrePayPeriod = ChainAvgPrePayDay / ChainAvgDueDay
    ChainMaxPrePayDay = PrePayPaidDat.groupby(['chain_name'])['pay_due_day'].max()
    ChainMaxPrePayPeriod = ChainMaxPrePayDay / ChainAvgDueDay
    ChainMinPrePayDay = PrePayPaidDat.groupby(['chain_name'])['pay_due_day'].min()
    ChainStdPrePayDay = PrePayPaidDat.groupby(['chain_name'])['pay_due_day'].std()
    ChainCvPrePayDay = ChainStdPrePayDay / ChainAvgPrePayDay
    Chain75PrePayDay = PrePayPaidDat['pay_due_day'].groupby(PrePayPaidDat['chain_name']).quantile(0.75)
    Chain25PrePayDay = PrePayPaidDat['pay_due_day'].groupby(PrePayPaidDat['chain_name']).quantile(0.25)
    Chain50PrePayDay = PrePayPaidDat['pay_due_day'].groupby(PrePayPaidDat['chain_name']).quantile(0.5)
    ChainSkewPrePayDay = PrePayPaidDat['pay_due_day'].groupby(PrePayPaidDat['chain_name']).skew()
    ChainKurtPrePayDay = PrePayPaidDat['pay_due_day'].groupby(PrePayPaidDat['chain_name']).apply(pd.Series.kurt)
    ChainIqrPrePayDay = Chain75PrePayDay - Chain25PrePayDay
    ChainPrePayDay1sig = sigma(PrePayPaidDat,1,'pay_due_day')
    ChainPrePayDay2sig = sigma(PrePayPaidDat,2,'pay_due_day')
    ChainPrePayDay3sig = sigma(PrePayPaidDat,3,'pay_due_day')

    ChainAvgOverduePayDay = OverduePaidDat.groupby(['chain_name'])['pay_due_day'].mean()
    ChainAvgOverduePayPeriod = ChainAvgOverduePayDay / ChainAvgDueDay
    ChainMaxOverduePayDay = OverduePaidDat.groupby(['chain_name'])['pay_due_day'].max()
    ChainMaxOverduePayPeriod = ChainMaxOverduePayDay / ChainAvgDueDay
    ChainMinOverduePayDay = OverduePaidDat.groupby(['chain_name'])['pay_due_day'].min()
    ChainStdOverduePayDay = OverduePaidDat.groupby(['chain_name'])['pay_due_day'].std()
    ChainCvOverduePayDay = ChainStdOverduePayDay / ChainAvgOverduePayDay
    Chain75OverduePayDay = OverduePaidDat['pay_due_day'].groupby(OverduePaidDat['chain_name']).quantile(0.75)
    Chain25OverduePayDay = OverduePaidDat['pay_due_day'].groupby(OverduePaidDat['chain_name']).quantile(0.25)
    Chain50OverduePayDay = OverduePaidDat['pay_due_day'].groupby(OverduePaidDat['chain_name']).quantile(0.5)
    ChainSkewOverduePayDay = OverduePaidDat['pay_due_day'].groupby(OverduePaidDat['chain_name']).skew()
    ChainKurtOverduePayDay = OverduePaidDat['pay_due_day'].groupby(OverduePaidDat['chain_name']).apply(pd.Series.kurt)
    ChainIqrOverduePayDay = Chain75PayDay - Chain25PayDay
    ChainPayOverdueDay1sig = sigma(OverduePaidDat,1,'pay_due_day')
    ChainPayOverdueDay2sig = sigma(OverduePaidDat,2,'pay_due_day')
    ChainPayOverdueDay3sig = sigma(OverduePaidDat,3,'pay_due_day')

    UnpaidDat = Dat[pd.isnull(Dat['pay_day'])]
    UnpaidDat['pay_day'] = (CaseMaxDate - UnpaidDat['time']).dt.days
    ChainAvgPayAge = UnpaidDat.groupby(['chain_name'])['pay_day'].mean()
    ChainAvgPayAvgPeriod = ChainAvgPayAge / ChainAvgDueDay
    ChainMaxPayAge = UnpaidDat.groupby(['chain_name'])['pay_day'].max()
    ChainMaxPayAgePeriod = ChainMaxPayAge / ChainAvgDueDay
    ChainMinPayAge = UnpaidDat.groupby(['chain_name'])['pay_day'].min()
    ChainStdPayAge = UnpaidDat.groupby(['chain_name'])['pay_day'].std()
    ChainNumPayAge = UnpaidDat.groupby(['chain_name'])['amount'].count()
    ChainAmtPayAge = UnpaidDat.groupby(['chain_name'])['amount'].sum()
    ChainCvPayAge = ChainStdPayAge / ChainAvgPayAge
    Chain75PayAge = UnpaidDat['pay_day'].groupby(UnpaidDat['chain_name']).quantile(0.75)
    Chain25PayAge = UnpaidDat['pay_day'].groupby(UnpaidDat['chain_name']).quantile(0.25)
    Chain50PayAge = UnpaidDat['pay_day'].groupby(UnpaidDat['chain_name']).quantile(0.5)
    ChainSkewPayAge = UnpaidDat['pay_day'].groupby(UnpaidDat['chain_name']).skew()
    ChainKurtPayAge = UnpaidDat['pay_day'].groupby(UnpaidDat['chain_name']).apply(pd.Series.kurt)
    ChainIqrPayAge = Chain75PayAge - Chain25PayAge
    ChainPayAge1sig = sigma(UnpaidDat,1,'pay_day')
    ChainPayAge2sig = sigma(UnpaidDat,2,'pay_day')
    ChainPayAge3sig = sigma(UnpaidDat,3,'pay_day')

    UnpaidDat['pay_due_day'] = (UnpaidDat['due_time'] - CaseMaxDate).dt.days
    OverdueUnpaidDat = UnpaidDat[UnpaidDat['pay_due_day'] < 0]
    ChainOverdueUnpaidAmt = OverdueUnpaidDat.groupby(['chain_name'])['amount'].sum()
    ChainOverdueUnpaidNum = OverdueUnpaidDat.groupby(['chain_name'])['amount'].count()
    ChainOverdueUnpaidAmtpct = ChainOverdueUnpaidAmt / ChainAmtPayAge
    ChainOverdueUnpaidNumpct = ChainOverdueUnpaidNum / ChainNumPayAge

    PayIndex = pd.DataFrame({'ChainAvgPayDay':ChainAvgPayDay,'ChainNumPayDay':ChainNumPayDay,
                             'ChainAvgDueDay':ChainAvgDueDay,'ChainMaxDueDay':ChainMaxDueDay,
                             'ChainMinDueDay':ChainMinDueDay,'ChainCrAmt':ChainCrAmt,'ChainCrNum':ChainCrNum,
                             'ChainAmtPayDay':ChainAmtPayDay,'ChainAvgPayPeriod':ChainAvgPayPeriod,
                             'ChainMaxPayPeriod':ChainMaxPayPeriod,'ChainMaxPayDay':ChainMaxPayDay,
                             'ChainMinPayDay':ChainMinPayDay,'ChainCvPayDay':ChainCvPayDay,
                             'Chain75PayDay':Chain75PayDay,'Chain25PayDay':Chain25PayDay,'Chain50PayDay':Chain50PayDay,
                             'ChainSkewPayDay':ChainSkewPayDay,'ChainKurtPayDay':ChainKurtPayDay,
                             'ChainIQRPayDay':ChainIqrPayDay,'ChainPayDay1sig':ChainPayDay1sig,
                             'ChainPayDay2sig':ChainPayDay2sig,'ChainPayDay3sig':ChainPayDay3sig,
                             'ChainStdPayDay':ChainStdPayDay,'ChainPrePayAmt':ChainPrePayAmt,
                             'ChainOverdueAmt': ChainOverdueAmt, 'ChainPrePayNum': ChainPrePayNum,
                             'ChainOverdueNum':ChainOverdueNum,'ChainAvgPrepayDay':ChainAvgPrePayDay,
                             'ChainAvgPrePayPeriod':ChainAvgPrePayPeriod,'ChainMaxPrePayPeriod':ChainMaxPrePayPeriod,
                             'ChainMinPrepayDay':ChainMinPrePayDay,'ChainStdPrepayDay':ChainStdPrePayDay,
                             'ChainCvPrePayDay': ChainCvPrePayDay, 'Chain75PrePayDay': Chain75PrePayDay,
                             'Chain25PrePayDay': Chain25PrePayDay, 'Chain50PrePayDay': Chain50PrePayDay,
                             'ChainSkewPrePayDay': ChainSkewPrePayDay, 'ChainKurtPrePayDay': ChainKurtPrePayDay,
                             'ChainIQRPrePayDay': ChainIqrPrePayDay, 'ChainPrePayDay1sig': ChainPrePayDay1sig,
                             'ChainPrePayDay2sig': ChainPrePayDay2sig, 'ChainPrePayDay3sig': ChainPrePayDay3sig,
                             'ChainMaxPrepayDay':ChainMaxPrePayDay,'ChainMaxOverdueDay':ChainMaxOverduePayDay,
                             'ChainAvgOverduePeriod':ChainAvgOverduePayPeriod,'ChainMaxOverduePeriod':ChainMaxOverduePayPeriod,
                             'ChainMinOverdueDay':ChainMinOverduePayDay,'ChainStdOverdueDay':ChainStdOverduePayDay,
                             'ChainCvOverduePayDay': ChainCvOverduePayDay, 'Chain75OverduePayDay': Chain75OverduePayDay,
                             'Chain25OverduePayDay': Chain25OverduePayDay, 'Chain50OverduePayDay': Chain50OverduePayDay,
                             'ChainSkewOverduePayDay': ChainSkewOverduePayDay, 'ChainKurtOverduePayDay': ChainKurtOverduePayDay,
                             'ChainIQROverduePayDay': ChainIqrOverduePayDay, 'ChainOverduePayDay1sig': ChainPayOverdueDay1sig,
                             'ChainOverduePayDay2sig': ChainPayOverdueDay2sig, 'ChainOverduePayDay3sig': ChainPayOverdueDay3sig,
                             'ChainAvgOverdueDay':ChainAvgOverduePayDay,'ChainAvgPayAge':ChainAvgPayAge,
                             'ChainMaxPayAge':ChainMaxPayAge,'ChainMinPayAge':ChainMinPayAge,'ChainStdPayAge':ChainStdPayAge,
                             'ChainCvPayAge':ChainCvPayAge,'ChainAvgPayAgePeriod':ChainAvgPayAvgPeriod,
                             'ChainMaxPayAgePeriod':ChainMaxPayAgePeriod,'ChainNumPayAge':ChainNumPayAge,'ChainAmtPayAge':ChainAmtPayAge,
                             'Chain75PayAge':Chain75PayAge,'Chain25PayAge':Chain25PayAge,'Chain50PayAge':Chain50PayAge,
                             'ChainSkewPayAge':ChainSkewPayAge,'ChainkurtPayAge':ChainKurtPayAge,'ChainIqrPayAge':ChainIqrPayAge,
                             'Chain1sigPayAge':ChainPayAge1sig,'Chain2sigPayAge':ChainPayAge2sig,'Chain3sigPayAge':ChainPayAge3sig,
                             'ChainOverdueUnpaidAmt':ChainOverdueUnpaidAmt,'ChainOverdueUnpaidNum':ChainOverdueUnpaidNum,
                             'ChainOverdueUnpaidAmtpct':ChainOverdueUnpaidAmtpct,'ChainOverdueUnpaidNumpct':ChainOverdueUnpaidNumpct,
                             'ChainOverdueAmtpct':ChainOverdueAmtpct,'ChainOverdueNumpct':ChainOverdueNumpct})
    return PayIndex

Bill_RM = RMF_RM(Bill,CaseMaxDate,'Bill')
Bill_RF = RMF_RF(Bill,CaseMaxDate,'Bill')
Bill_Gap = RMF_Gap(BillGapDat,BillGapDatU,'Bill')
Pay_RM = RMF_RM(Pay,CaseMaxDate,'Pay')
Pay_RF = RMF_RF(Pay,CaseMaxDate,'Pay')
Pay_Gap = RMF_Gap(PayGapDat,PayGapDatU,'Pay')
Bill_Pay = RMF_Pay(BillPay,is_credit=1)

Index = pd.concat([Bill_RM, Bill_RF,Bill_Gap,Pay_RM,Pay_RF,Pay_Gap,Bill_Pay,DrIndex,TsIndex], axis=1)
Index.to_csv('Index.csv',encoding='gbk')

##########Index Prepare###########
ChainNum = len(Index)
NanNum = Index.isnull().sum()
Nanpct = NanNum / ChainNum
NanIndex = pd.DataFrame({'NanNum':NanNum,'Nanpct':Nanpct})
##check Nanpct > 0 ###
print NanIndex[NanIndex.Nanpct>0]

###################### 金额和笔数相关为Nan代表无交易金额，fillna(0) ######################
###################### 标准差为Nan代表数据<=1 ,过滤######################
###################### 订单相关为Nan代表间隔<=1，过滤 ######################
###################### 天数为Nan代表付款<1,过滤 ######################
Amtlist = ['BillRctAmt1yr','BillRctAmtHfyr','BillRctAmtQt','PayRctAmt1yr','PayRctAmtHfyr','PayRctAmtQt',
       'ChainOverdueAmt','ChainOverdueNum','ChainPrePayAmt','ChainPrePayNum']
Index[Amtlist] = Index[Amtlist].fillna(0)

###数据过滤###
Index.drop(['BillChainFirstDate','BillChainLastDate','PayChainFirstDate','PayChainLastDate'],axis=1,inplace=True)

####只下过一次单的企业，只下过两次单的企业####
Index = Index[Index['BillNum']>1]
#Index = Index[Index['BillNum']>2]
Index = Index[Index['BillChainDay']>1]
#Index = Index[Index['BillChainDay']>2]

####只付过一次款的企业，只下过两次单的企业####
Index = Index[Index['PayNum']>1]
#Index = Index[Index['PayNum']>2]
Index = Index[Index['PayChainDay']>1]
#Index = Index[Index['PayChainDay']>2]

bp2list =['BillChainStdGap','BillChainStdGapU','PayChainStdGap','PayChainStdGapU']
Index[bp2list] = Index[bp2list].fillna(0)

###没有完整付款的企业###
Index = Index[Index['ChainNumPayDay']>1]

###无逾期或无提前付款的企业###
#Index = Index[Index['ChainOverdueNum']>1]
#Index = Index[Index['ChainPrePayNum']>1]
oplist = ['ChainAvgOverdueDay','ChainAvgPrepayDay', 'ChainMaxOverdueDay','ChainMaxPrepayDay',
                     'ChainMinOverdueDay','ChainMinPrepayDay',]
Index[oplist] = Index[oplist].fillna(0)

###只下过一次付款或逾期或提前的企业#######
#Index = Index[Index['ChainOverdueNum']>2]
#Index = Index[Index['ChainPrePayNum']>2]
op2list = ['ChainStdOverdueDay','ChainStdPrepayDay','ChainStdPayDay']
Index[op2list] = Index[op2list].fillna(0)

