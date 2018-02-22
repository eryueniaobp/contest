# encoding=utf-8
"""
@author : pengalg
"""


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging,datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='prop.log',
                    filemode='a')



import pandas as pd
import numpy as np
from fbprophet import Prophet

import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation

def make_holidays():
    """
    将所有的空缺都算成holidays,避免重复规避.


    future中的也要考虑.
    :return:
    """
    traindf = pd.read_csv('train_20171215.txt', sep='\t', header=0)
    #make sample
    traindf = traindf[['date', 'day_of_week', 'cnt']].groupby(['date', 'day_of_week']).count().reset_index()


    testdf = pd.read_csv('test_A_20171225.txt',sep='\t',header=0)
    testdf['cnt'] = 0

    df = pd.concat([traindf, testdf.iloc[1:, :]])


    dvals = df.values
    prev = 2

    date = 1

    p = 0

    buf = []
    while p < len(dvals):
        _, day_of_week, cnt = dvals[p]
        target = prev % 7 + 1
        if day_of_week == target:
            p += 1
        else:
            buf.append(date)  #no register , so it 's  holiday
        prev = target
        date +=1
    holiday_df = pd.DataFrame({'holiday': 'off', 'ds': buf})

    holiday_df['ds'] = holiday_df['ds'].apply(lambda x: pd.DateOffset(days=x) + pd.to_datetime('2017-02-07', format='%Y-%m-%d'))
    return holiday_df

def prepare_data():
    df = pd.read_csv('train_20171215.txt', sep='\t', header=0)
    medvals = np.ravel(df[['day_of_week', 'cnt']].groupby('day_of_week').median().values)

    df = df[['date', 'day_of_week', 'cnt']].groupby(['date', 'day_of_week']).sum().reset_index()
    with open('train_20171215.full.txt', 'w') as f:
        dvals = df.values
        prev = 2

        date = 1


        p = 0
        while  p < len(dvals):
            _ ,day_of_week, cnt = dvals[p]


            target = prev%7 +1
            if day_of_week == target:
                f.write("{0}\t{1}\t{2}\n".format(date,target,cnt))
                p +=1
            else:
                cnt = medvals[target-1]
                cnt = 0  # todo : check if 0 is better.
                logging.info('dayofweek = {0}  fillmiss = {1}'.format(target, cnt))
                f.write("{0}\t{1}\t{2}\n".format(date,target,cnt))
            prev = target
            date += 1

    return  'train_20171215.full.txt'

pd.set_option('display.precision', 1)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
"""
span = 250:
cutoff
2019-05-05   363945.3
2019-09-07   863657.8
Name: error, dtype: float64

提交后的答案为 87万 ；这个说鸣  越往后，越不好预测? ???  观察下trend的情况?

"""
def post_data():
    pass
def main():
    logging.info("begin")

    path = prepare_data()

    holidays = make_holidays()
    df = pd.read_csv(path,sep='\t',header=None)


    df.columns = ['ds','day_of_week', 'y']


    df['ds']  =  df['ds'].apply(lambda  x : pd.DateOffset(days = x ) + pd.to_datetime('2017-02-07', format='%Y-%m-%d'))

    m = Prophet(changepoint_prior_scale=0.05, holidays=holidays)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    print df.columns
    # m.add_regressor('day_of_week')
    m.fit(df)

    # span = 250
    # cv = cross_validation(m , horizon='{0} days'.format(span))
    #
    #
    # print cv['cutoff'].unique()
    #
    # cv['error'] = 1./span * (cv['yhat'] - cv['y'] ) ** 2
    #
    # print cv.groupby('cutoff')['error'].sum()
    #
    #
    #
    #
    # return

    future = m.make_future_dataframe(freq='D', periods=500)
    # future['day_of_week']  = future['ds'].apply(lambda  x :  x.dayofweek)
    forecast = m.predict(future)

    # m.plot(forecast)
    m.plot_components(forecast)
    m.plot(forecast)

    yhat = forecast['yhat'].tail(501).values

    print len(yhat)

    day = datetime.datetime.now().strftime('%Y%m%d')

    # pd.DataFrame ({'ds': [ i%7 +1 for i in range(3, 3+501 ) ]  , 'yhat':  yhat}).to_csv('{day}.csv'.format(day=day),sep='\t', header=False,index=False)


    test = pd.read_csv('test_A_20171225.txt', sep='\t',header=0)
    dvals = test['day_of_week'].values

    p = 0
    ds = [ i%7 +1 for i in range(3, 3+501 ) ]

    buf = []
    for dw in dvals:
        while ds[p] != dw:
            # print 'miss hole'
            p +=1

        buf.append(yhat[p])

    pd.DataFrame({'ds': test['date'] , 'yhat': buf }).to_csv('{day}.monthly.csv'.format(day=day),sep='\t', header=False,index=False)



    plt.show()




if __name__ =='__main__':
    #build()
    # predict()
    main()
