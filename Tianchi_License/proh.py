#encoding=utf-8

import pandas as pd

from fbprophet import Prophet
import sys
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation
import datetime
import numpy as np

def shiftdw(cur, step):
    t = ( cur + step  )%7 
    if t == 0:
        return 7
    if t < 0:
        return t + 7 
    else:
        return t

    
def read_train_df():
    df = pd.read_csv('train_20171215.txt', sep='\t', header=0)
    print df['cnt'].dtype

    traindfsum = df.groupby(['date','day_of_week'])['cnt'].sum().reset_index()

    medvals = np.ravel(df.groupby('day_of_week')['cnt'].median().values)


    print medvals

    testdf = pd.read_csv('test_A_20171225.txt' ,sep='\t',header=0)

    testdf['cnt'] = 0
    testdf = testdf.iloc[1: , : ]

    dfsum =  pd.concat([traindfsum , testdf]).reset_index()

    hashdate = []
    rldate = []
    cnt = []
    day_of_week  = []
    
    prdate = -1 
    prdw = -1 
    for row in dfsum.itertuples():
        if prdate == -1:
            hashdate.append(row.date)
            rldate.append(row.date)
            cnt.append( row.cnt )
            day_of_week.append(row.day_of_week)
        else:
            prdw = shiftdw(prdw, 1)
            while prdw != row.day_of_week:
                rldate.append(rldate[-1] + 1)
                hashdate.append(-1)


                cnt.append(medvals[prdw - 1 ] )


                day_of_week.append(prdw)
                prdw = shiftdw(prdw,1)
            rldate.append(rldate[-1] + 1) 
            hashdate.append(row.date)
            cnt.append(row.cnt)
            day_of_week.append(row.day_of_week)

        prdate = row.date
        prdw = row.day_of_week


    dfsum = pd.DataFrame({'date': hashdate, 'rldate': rldate, 'day_of_week': day_of_week, 'cnt':cnt})

    dfsum.to_csv('train.csv', index=False)
    return dfsum, traindfsum['date'].max()
def amplify(df):
    dis2lastwork = []
    dis2lastholiday = []

    dis2nextwork = []
    dis2nextholiday = []

    prdate = 0 
    prholiday = 0 

    for row in df.itertuples():
        dis2lastwork += [ row.rldate  - prdate ]
        dis2lastholiday  += [row.rldate - prholiday ] 
        if row.date > -1:
            prdate = row.rldate
        else:
            prholiday = row.rldate

    
    nrdate = df['rldate'].max()
    nrholiday =df['rldate'].max() 
    rdf = df.sort_values(by='rldate', ascending=False)
    for row in rdf.itertuples():
        dis2nextwork += [ nrdate - row.rldate  ]
        dis2nextholiday += [nrholiday -  row.rldate ] 
        if row.date > -1:
            nrdate = row.rldate
        else:
            nrholiday = row.rldate
    
    dis2nextwork.reverse()
    dis2nextholiday.reverse()
    df['lastwork'] = dis2lastwork
    df['lastholiday'] = dis2lastholiday
    df['nextwork'] = dis2nextwork 
    df['nextholiday'] = dis2nextholiday
    return df
if __name__  == '__main__':
    zero_day = '2013-01-01'
    df, traindate =read_train_df()
    df = amplify(df)
    lentrain = df[df['date'] == traindate].index[0] + 1

    print df.iloc[lentrain-1 , :]

    # raw_input('\t\tPress')

    print 'lentrain = ' , lentrain
    # raw_input('\t\tPress any ')
    df['ds']  =  df['rldate'].apply(lambda  x : pd.DateOffset(days = x ) + pd.to_datetime(zero_day, format='%Y-%m-%d'))
    df['y' ] = df['cnt']

    holidays = df[df['date'] == -1][['ds']]
    holidays['holiday'] = 'off'
    holidays['lower_window'] = -1
    holidays['upper_window'] = 1
    print df.columns


    m = Prophet(changepoint_prior_scale=0.05, holidays=holidays)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    m.add_regressor('lastwork')
    m.add_regressor('lastholiday')
    m.add_regressor('nextwork')
    m.add_regressor('nextholiday')

    m.fit(df.iloc[:lentrain, :]) # train
    future = m.make_future_dataframe(freq='D', periods=500)
    future  =  pd.merge( future, df , on ='ds')
    print future.head() ,future.tail()

    forecast =  m.predict(future)  # type:pd.DataFrame  ds ,yhat

    m.plot_components(forecast)
    m.plot(forecast)

    testdf = df.iloc[lentrain-1:, :]
    testdf = pd.merge(testdf,  forecast, on='ds')

    day = datetime.datetime.now().strftime('%Y%m%d')

    print testdf[testdf['date'] != -1][['yhat']].describe()
    testdf[testdf['date'] != -1][['date', 'yhat']].to_csv('{day}.reg.more.csv'.format(day=day),sep='\t', header=False,index=False)
    plt.show()
