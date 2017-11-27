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
rpay.rename(columns={'coreerpchainuid':'chain_id','pay_time':'time'},inplace=True)
Pay = rpay[['core_name','chain_name','order_no','time','amount']]
BillPay = rbill[['core_name','chain_name','order_no','time','amount','due_time','match_pay_time','paid_amount','is_all_paid']]
BillPay['pay_day'] = (BillPay['match_pay_time'] - BillPay['time']).dt.days
BillPay['due_day'] = (BillPay['due_time'] - BillPay['time']).dt.days
BillPay['pay_due_day'] = (BillPay['due_time'] - BillPay['match_pay_time']).dt.days
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
    core_dat = Dat[~pd.isnull(Dat[tm])]
    core_ts = Trans_ts(core_dat,act,feature,freq,tm,maxdate)
    CoreTs = pd.DataFrame({'core_name':[u'核心企业'] * len(core_ts), 'time':core_ts.index.tolist(),
                           'value':core_ts.values.tolist()})
    return ChainTs,CoreTs

CaseBillMaxDate = Bill['time'].max() + DateOffset(months=1)
CasePayMaxDate = Pay['time'].max() + DateOffset(months=1)
CaseMatchPayMaxDate = BillPay['match_pay_time'].max() + DateOffset(months=1)
ChainBillAmtMonthTs,CoreBillAmtMonthTs = Get_ts(Bill,'sum','amount','M','time',CaseBillMaxDate)
ChainBillAmtAvgMonthTs,CoreBillAmtAvgMonthTs = Get_ts(Bill,'avg','amount','M','time',CaseBillMaxDate)
ChainBillAmtMaxMonthTs,CoreBillAmtMaxMonthTs = Get_ts(Bill,'max','amount','M','time',CaseBillMaxDate)
ChainBillAmtMinMonthTs,CoreBillAmtMinMonthTs = Get_ts(Bill,'min','amount','M','time',CaseBillMaxDate)

ChainBillGapAvgMonthTs,CoreBillGapAvgMonthTs = Get_ts(BillGapDatU,'avg','Gap','M','time',CaseBillMaxDate)
ChainBillGapMaxMonthTs,CoreBillGapMaxMonthTs = Get_ts(BillGapDatU,'max','Gap','M','time',CaseBillMaxDate)
ChainBillGapMinMonthTs,CoreBillGapMinMonthTs = Get_ts(BillGapDatU,'min','Gap','M','time',CaseBillMaxDate)

ChainPayAmtMonthTs,CorePayAmtMonthTs = Get_ts(Pay,'sum','amount','M','time',CasePayMaxDate)
ChainPayAmtAvgMonthTs,CorePayAmtAvgMonthTs = Get_ts(Pay,'avg','amount','M','time',CasePayMaxDate)
ChainPayAmtMaxMonthTs,CorePayAmtMaxMonthTs = Get_ts(Pay,'max','amount','M','time',CasePayMaxDate)
ChainPayAmtMinMonthTs,CorePayAmtMinMonthTs = Get_ts(Pay,'min','amount','M','time',CasePayMaxDate)

ChainPayGapAvgMonthTs,CorePayGapAvgMonthTs = Get_ts(PayGapDatU,'avg','Gap','M','time',CasePayMaxDate)
ChainPayGapMaxMonthTs,CorePayGapMaxMonthTs = Get_ts(PayGapDatU,'max','Gap','M','time',CasePayMaxDate)
ChainPayGapMinMonthTs,CorePayGapMinMonthTs = Get_ts(PayGapDatU,'min','Gap','M','time',CasePayMaxDate)

ChainPayDayAvgMonthTs,CorePayDayAvgMonthTs = Get_ts(BillPay,'avg','pay_day','M','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMaxMonthTs,CorePayDayMaxMonthTs = Get_ts(BillPay,'max','pay_day','M','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMinMonthTs,CorePayDayMinMonthTs = Get_ts(BillPay,'min','pay_day','M','match_pay_time',CaseMatchPayMaxDate)

ChainBillAmtQuaterTs,CoreBillAmtQuaterTs = Get_ts(Bill,'sum','amount','Q','time',CaseBillMaxDate)
ChainBillAmtAvgQuaterTs,CoreBillAmtAvgQuaterTs = Get_ts(Bill,'avg','amount','Q','time',CaseBillMaxDate)
ChainBillAmtMaxQuaterTs,CoreBillAmtMaxQuaterTs = Get_ts(Bill,'max','amount','Q','time',CaseBillMaxDate)
ChainBillAmtMinQuaterTs,CoreBillAmtMinQuaterTs = Get_ts(Bill,'min','amount','Q','time',CaseBillMaxDate)

ChainBillGapAvgQuaterTs,CoreBillGapAvgQuaterTs = Get_ts(BillGapDatU,'avg','Gap','Q','time',CaseBillMaxDate)
ChainBillGapMaxQuaterTs,CoreBillGapMaxQuaterTs = Get_ts(BillGapDatU,'max','Gap','Q','time',CaseBillMaxDate)
ChainBillGapMinQuaterTs,CoreBillGapMinQuaterTs = Get_ts(BillGapDatU,'min','Gap','Q','time',CaseBillMaxDate)

ChainPayAmtQuaterTs,CorePayAmtQuaterTs = Get_ts(Pay,'sum','amount','Q','time',CasePayMaxDate)
ChainPayAmtAvgQuaterTs,CorePayAmtAvgQuaterTs = Get_ts(Pay,'avg','amount','Q','time',CasePayMaxDate)
ChainPayAmtMaxQuaterTs,CorePayAmtMaxQuaterTs = Get_ts(Pay,'max','amount','Q','time',CasePayMaxDate)
ChainPayAmtMinQuaterTs,CorePayAmtMinQuaterTs = Get_ts(Pay,'min','amount','Q','time',CasePayMaxDate)

ChainPayGapAvgQuaterTs,CorePayGapAvgQuaterTs = Get_ts(PayGapDatU,'avg','Gap','Q','time',CasePayMaxDate)
ChainPayGapMaxQuaterTs,CorePayGapMaxQuaterTs = Get_ts(PayGapDatU,'max','Gap','Q','time',CasePayMaxDate)
ChainPayGapMinQuaterTs,CorePayGapMinQuaterTs = Get_ts(PayGapDatU,'min','Gap','Q','time',CasePayMaxDate)

ChainPayDayAvgQuaterTs,CorePayDayAvgQuaterTs = Get_ts(BillPay,'avg','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMaxQuaterTs,CorePayDayMaxQuaterTs = Get_ts(BillPay,'max','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMinQuaterTs,CorePayDayMinQuaterTs = Get_ts(BillPay,'min','pay_day','Q','match_pay_time',CaseMatchPayMaxDate)

ChainBillAmtWeekTs,CoreBillAmtWeekTs = Get_ts(Bill,'sum','amount','W','time',CaseBillMaxDate)
ChainBillAmtAvgWeekTs,CoreBillAmtAvgWeekTs = Get_ts(Bill,'avg','amount','W','time',CaseBillMaxDate)
ChainBillAmtMaxWeekTs,CoreBillAmtMaxWeekTs = Get_ts(Bill,'max','amount','W','time',CaseBillMaxDate)
ChainBillAmtMinWeekTs,CoreBillAmtMinWeekTs = Get_ts(Bill,'min','amount','W','time',CaseBillMaxDate)

ChainBillGapAvgWeekTs,CoreBillGapAvgWeekTs = Get_ts(BillGapDatU,'avg','Gap','W','time',CaseBillMaxDate)
ChainBillGapMaxWeekTs,CoreBillGapMaxWeekTs = Get_ts(BillGapDatU,'max','Gap','W','time',CaseBillMaxDate)
ChainBillGapMinWeekTs,CoreBillGapMinWeekTs = Get_ts(BillGapDatU,'min','Gap','W','time',CaseBillMaxDate)

ChainPayAmtWeekTs,CorePayAmtWeekTs = Get_ts(Pay,'sum','amount','W','time',CasePayMaxDate)
ChainPayAmtAvgWeekTs,CorePayAmtAvgWeekTs = Get_ts(Pay,'avg','amount','W','time',CasePayMaxDate)
ChainPayAmtMaxWeekTs,CorePayAmtMaxWeekTs = Get_ts(Pay,'max','amount','W','time',CasePayMaxDate)
ChainPayAmtMinWeekTs,CorePayAmtMinWeekTs = Get_ts(Pay,'min','amount','W','time',CasePayMaxDate)

ChainPayGapAvgWeekTs,CorePayGapAvgWeekTs = Get_ts(PayGapDatU,'avg','Gap','W','time',CasePayMaxDate)
ChainPayGapMaxWeekTs,CorePayGapMaxWeekTs = Get_ts(PayGapDatU,'max','Gap','W','time',CasePayMaxDate)
ChainPayGapMinWeekTs,CorePayGapMinWeekTs = Get_ts(PayGapDatU,'min','Gap','W','time',CasePayMaxDate)

ChainPayDayAvgWeekTs,CorePayDayAvgWeekTs = Get_ts(BillPay,'avg','pay_day','W','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMaxWeekTs,CorePayDayMaxWeekTs = Get_ts(BillPay,'max','pay_day','W','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMinWeekTs,CorePayDayMinWeekTs = Get_ts(BillPay,'min','pay_day','W','match_pay_time',CaseMatchPayMaxDate)

ChainBillAmtYrTs,CoreBillAmtYrTs = Get_ts(Bill,'sum','amount','A','time',CaseBillMaxDate)
ChainBillAmtAvgYrTs,CoreBillAmtAvgYrTs = Get_ts(Bill,'avg','amount','A','time',CaseBillMaxDate)
ChainBillAmtMaxYrTs,CoreBillAmtMaxYrTs = Get_ts(Bill,'max','amount','A','time',CaseBillMaxDate)
ChainBillAmtMinYrTs,CoreBillAmtMinYrTs = Get_ts(Bill,'min','amount','A','time',CaseBillMaxDate)

ChainBillGapAvgYrTs,CoreBillGapAvgYrTs = Get_ts(BillGapDatU,'avg','Gap','A','time',CaseBillMaxDate)
ChainBillGapMaxYrTs,CoreBillGapMaxYrTs = Get_ts(BillGapDatU,'max','Gap','A','time',CaseBillMaxDate)
ChainBillGapMinYrTs,CoreBillGapMinYrTs = Get_ts(BillGapDatU,'min','Gap','A','time',CaseBillMaxDate)

ChainPayAmtYrTs,CorePayAmtYrTs = Get_ts(Pay,'sum','amount','A','time',CasePayMaxDate)
ChainPayAmtAvgYrTs,CorePayAmtAvgYrTs = Get_ts(Pay,'avg','amount','A','time',CasePayMaxDate)
ChainPayAmtMaxYrTs,CorePayAmtMaxYrTs = Get_ts(Pay,'max','amount','A','time',CasePayMaxDate)
ChainPayAmtMinYrTs,CorePayAmtMinYrTs = Get_ts(Pay,'min','amount','A','time',CasePayMaxDate)

ChainPayGapAvgYrTs,CorePayGapAvgYrTs = Get_ts(PayGapDatU,'avg','Gap','A','time',CasePayMaxDate)
ChainPayGapMaxYrTs,CorePayGapMaxYrTs = Get_ts(PayGapDatU,'max','Gap','A','time',CasePayMaxDate)
ChainPayGapMinYrTs,CorePayGapMinYrTs = Get_ts(PayGapDatU,'min','Gap','A','time',CasePayMaxDate)

ChainPayDayAvgYrTs,CorePayDayAvgYrTs = Get_ts(BillPay,'avg','pay_day','A','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMaxYrTs,CorePayDayMaxYrTs = Get_ts(BillPay,'max','pay_day','A','match_pay_time',CaseMatchPayMaxDate)
ChainPayDayMinYrTs,CorePayDayMinYrTs = Get_ts(BillPay,'min','pay_day','A','match_pay_time',CaseMatchPayMaxDate)

ChainTsFrames = [ChainBillAmtMonthTs,ChainBillAmtAvgMonthTs,ChainBillAmtMaxMonthTs,ChainBillAmtMinMonthTs,
                 ChainBillGapAvgMonthTs,ChainBillGapMaxMonthTs,ChainBillGapMinMonthTs,ChainPayAmtMonthTs,
                 ChainPayAmtAvgMonthTs,ChainPayAmtMaxMonthTs,ChainPayAmtMinMonthTs,ChainPayGapAvgMonthTs,
                 ChainPayGapMaxMonthTs,ChainPayGapMinMonthTs,ChainPayDayAvgMonthTs,ChainPayDayMaxMonthTs,
                 ChainPayDayMinMonthTs,ChainBillAmtWeekTs, ChainBillAmtAvgWeekTs, ChainBillAmtMaxWeekTs,
                 ChainBillAmtMinWeekTs, ChainBillGapAvgWeekTs, ChainBillGapMaxWeekTs,ChainBillGapMinWeekTs,
                 ChainPayAmtWeekTs, ChainPayAmtAvgWeekTs,ChainPayAmtMaxWeekTs, ChainPayAmtMinWeekTs,
                 ChainPayGapAvgWeekTs, ChainPayGapMaxWeekTs,ChainPayGapMinWeekTs,ChainPayDayAvgWeekTs,
                 ChainPayDayMaxWeekTs, ChainPayDayMinWeekTs,ChainBillAmtQuaterTs, ChainBillAmtAvgQuaterTs,
                 ChainBillAmtMaxQuaterTs, ChainBillAmtMinQuaterTs,ChainBillGapAvgQuaterTs, ChainBillGapMaxQuaterTs,
                 ChainBillGapMinQuaterTs,ChainPayAmtQuaterTs, ChainPayAmtAvgQuaterTs, ChainPayAmtMaxQuaterTs,
                 ChainPayAmtMinQuaterTs,ChainPayGapAvgQuaterTs, ChainPayGapMaxQuaterTs, ChainPayGapMinQuaterTs,
                 ChainPayDayAvgQuaterTs,ChainPayDayMaxQuaterTs, ChainPayDayMinQuaterTs, ChainBillAmtYrTs,
                 ChainBillAmtAvgYrTs,ChainBillAmtMaxYrTs, ChainBillAmtMinYrTs, ChainBillGapAvgYrTs, ChainBillGapMaxYrTs,
                 ChainBillGapMinYrTs, ChainPayAmtYrTs, ChainPayAmtAvgYrTs, ChainPayAmtMaxYrTs,ChainPayAmtMinYrTs,
                 ChainPayGapAvgYrTs, ChainPayGapMaxYrTs, ChainPayGapMinYrTs,ChainPayDayAvgYrTs, ChainPayDayMaxYrTs,
                 ChainPayDayMinYrTs]

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

CoreTsFrames = [CoreBillAmtMonthTs,CoreBillAmtAvgMonthTs,CoreBillAmtMaxMonthTs,CoreBillAmtMinMonthTs,
                CoreBillGapAvgMonthTs,CoreBillGapMaxMonthTs,CoreBillGapMinMonthTs,CorePayAmtMonthTs,
                CorePayAmtAvgMonthTs,CorePayAmtMaxMonthTs,CorePayAmtMinMonthTs,CorePayGapAvgMonthTs,
                CorePayGapMaxMonthTs,CorePayGapMinMonthTs,CorePayDayAvgMonthTs,CorePayDayMaxMonthTs,
                CorePayDayMinMonthTs,CoreBillAmtWeekTs, CoreBillAmtAvgWeekTs, CoreBillAmtMaxWeekTs,
                CoreBillAmtMinWeekTs, CoreBillGapAvgWeekTs, CoreBillGapMaxWeekTs,CoreBillGapMinWeekTs,
                CorePayAmtWeekTs, CorePayAmtAvgWeekTs,CorePayAmtMaxWeekTs, CorePayAmtMinWeekTs,
                CorePayGapAvgWeekTs, CorePayGapMaxWeekTs,CorePayGapMinWeekTs,CorePayDayAvgWeekTs,
                CorePayDayMaxWeekTs, CorePayDayMinWeekTs,CoreBillAmtQuaterTs, CoreBillAmtAvgQuaterTs,
                CoreBillAmtMaxQuaterTs, CoreBillAmtMinQuaterTs,CoreBillGapAvgQuaterTs, CoreBillGapMaxQuaterTs,
                CoreBillGapMinQuaterTs,CorePayAmtQuaterTs, CorePayAmtAvgQuaterTs, CorePayAmtMaxQuaterTs,
                CorePayAmtMinQuaterTs,CorePayGapAvgQuaterTs, CorePayGapMaxQuaterTs, CorePayGapMinQuaterTs,
                CorePayDayAvgQuaterTs,CorePayDayMaxQuaterTs, CorePayDayMinQuaterTs, CoreBillAmtYrTs,
                CoreBillAmtAvgYrTs,CoreBillAmtMaxYrTs, CoreBillAmtMinYrTs, CoreBillGapAvgYrTs, CoreBillGapMaxYrTs,
                CoreBillGapMinYrTs, CorePayAmtYrTs, CorePayAmtAvgYrTs, CorePayAmtMaxYrTs,CorePayAmtMinYrTs,
                CorePayGapAvgYrTs, CorePayGapMaxYrTs, CorePayGapMinYrTs,CorePayDayAvgYrTs, CorePayDayMaxYrTs,
                CorePayDayMinYrTs]

ChainTimeSeries = pd.concat(ChainTsFrames,keys=names)
ChainTs = ChainTimeSeries.reset_index()
ChainTs.to_csv('ChainTimeSeries.csv',encoding='gbk')
CoreTimeSeries = pd.concat(CoreTsFrames,keys=names)
CoreTs = CoreTimeSeries.reset_index()
CoreTs.to_csv('CoreTimeSeries.csv',encoding='gbk')
