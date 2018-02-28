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


def hand_holidays(df):
    fd = 'fd'
    spring = 'spring'
    qingming = 'qingming'
    laodong = 'laodong'
    duanwu = 'duanwu'
    zhongqiu = 'zhongqiu'

    guoqing = 'guoqing'

    other = 'other'



    hdays = [
        ('2013-01-01', 3, fd) , # 元旦
        ('2013-02-09', 7, spring ) ,
        ('2013-04-04', 3, qingming) ,
        ('2013-04-29', 3,  laodong) ,
        ('2013-06-10', 3, duanwu),
        ('2013-09-19', 3, zhongqiu),
        ('2013-10-01', 7, guoqing),

        ('2014-01-01', 1, fd ),
        ('2014-01-31', 7, spring),
        ('2014-04-05', 3, qingming),
        ('2014-05-01', 3, laodong),
        ('2014-05-31', 3, duanwu),
        ('2014-09-06', 3, zhongqiu),
        ('2014-10-01', 7, guoqing),

        ('2015-01-01', 3, fd),
        ('2015-02-18', 7, spring),
        ('2015-04-04', 3, qingming),
        ('2015-05-01', 3, laodong),
        ('2015-06-20', 3, duanwu),
        ('2015-09-03', 3, other),
        ('2015-09-26', 2, zhongqiu),
        ('2015-10-01', 7, guoqing),


        ('2016-01-01', 3, fd ),
        ('2016-02-07', 7,spring),
        ('2016-04-02', 3, qingming),
        ('2016-04-30', 3, laodong),
        ('2016-06-09', 3, duanwu),
        ('2016-09-15', 3, zhongqiu),
        ('2016-10-01', 7,guoqing),
        ('2016-12-31', 3, fd),

        ('2017-01-27', 7,spring),
        ('2017-04-02', 3, qingming),
        ('2017-04-29', 3,laodong),
        ('2017-05-28', 3, duanwu),
        ('2017-10-01', 8, guoqing),
        ('2017-12-30', 3, fd),
        ('2018-02-15', 7, spring),


    ]
    work_day = [
        ('2013-01-05', 2, ''  ) ,
        ('2013-02-16', 2, '' ) ,
        ('2013-04-07', 1, ''),
        ('2013-04-27', 2, '' ),
        ('2013-06-08', 2, '' ),
        ('2013-09-22', 1, '' ),
        ('2013-09-29', 1, ''),
        ('2013-10-12', 1, '' ),
        ('2014-01-26', 1, '' ),
        ('2014-02-08', 1, ''),
        ('2014-05-04', 1, ''),
        ('2014-09-28', 1, ''),
        ('2014-10-11', 1, ''),
        ('2015-01-04', 1, ''),
        ('2015-02-15', 1, ''),
        ('2015-02-28', 1, ''),
        ('2015-09-06', 1, ''),
        ('2015-10-10', 1, ''),
        ('2016-02-06', 1, ''),
        ('2016-02-14', 1, ''),
        ('2016-06-12', 1, ''),
        ('2016-09-18', 1, ''),
        ('2016-10-08', 2, ''),
        ('2017-01-22', 1, ''),
        ('2017-02-04', 1, ''),
        ('2017-04-01', 1, ''),
        ('2017-05-27', 1, ''),
        ('2017-09-30', 1, ''),
        ('2018-02-11', 1, ''),
        ('2018-02-24', 1, ''),

    ]
    def expand(buf):
        nb = {}
        for day , span, tag in buf:
            if span == 1:
                # nb += [day]
                nb[day] = tag
            else:
                while span > 0:
                    nday = datetime.datetime.strptime(day, '%Y-%m-%d') + datetime.timedelta(days = span -1 )
                    nday = nday.strftime('%Y-%m-%d')
                    # nb += [nday]
                    nb[nday] = tag
                    span -= 1

        return nb
    hdays = expand(hdays)
    wdays = expand(work_day)

    """
    df['ds'] is ok
    """
    def check_holiday(d):
        dw = d.dayofweek +1
        ds =  d.strftime('%Y-%m-%d')


        if ds in hdays:
            return 1
        elif ds in wdays:
            return 0
        else:
            if dw >= 6:
                return 1
            else:
                return 0

    def check_holiday_name(d):
        dw = d.dayofweek +1
        ds =  d.strftime('%Y-%m-%d')


        if ds in hdays:
            return hdays[ds]
        elif ds in wdays:
            return 'work'
        else:
            if dw == 6:
                return 'saturday'
            elif dw == 7:
                return 'sunday'
            else:
                return 'normal'



    df['holiday'] = df['ds'].apply(lambda d:  check_holiday(d) )
    df['holiday_name'] = df['ds'].apply(lambda d : check_holiday_name(d))
    return df
def read_testA_df():
    testdf = pd.read_csv('./data/test_A_20171225.txt', sep='\t', header=0)
    ans = pd.read_csv('./data/answer_A_20180225.txt',sep='\t',header=None)
    ans.columns = ['date', 'cnt']

    testdf = pd.merge(testdf, ans, on='date')

    print testdf.head()

    return testdf.iloc[1:, :]

def read_train_df():
    df = pd.read_csv('./data/train_20171215.txt', sep='\t', header=0)
    print df['cnt'].dtype

    traindfsum = df.groupby(['date','day_of_week'])['cnt'].sum().reset_index()

    medvals = np.ravel(df.groupby('day_of_week')['cnt'].median().values)


    print medvals

    medvals = [0] * 7

    # todo: 国庆，春节，需要特殊处理 ，直接给量为 10 左右

    # sys.exit(1)


    dfA = read_testA_df()

    testdf = pd.read_csv('./data/test_B_20171225.txt' ,sep='\t',header=0)

    testdf['cnt'] = 0
    # testdf = testdf.iloc[1: , : ]

    dfsum =  pd.concat([traindfsum , dfA,  testdf]).reset_index()

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
    return dfsum, dfA['date'].max() # traindfsum['date'].max()
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
        if row.holiday == 0 :
            prdate = row.rldate
        else:
            prholiday = row.rldate

    
    nrdate = df['rldate'].max()
    nrholiday =df['rldate'].max() 
    rdf = df.sort_values(by='rldate', ascending=False)
    for row in rdf.itertuples():
        dis2nextwork += [ nrdate - row.rldate  ]
        dis2nextholiday += [nrholiday -  row.rldate ] 
        if row.holiday == 0:
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
    df['ds'] = df['rldate'].apply(lambda x: pd.DateOffset(days=x) + pd.to_datetime(zero_day, format='%Y-%m-%d'))

    df = hand_holidays(df)

    df.to_csv('train.csv',index=False)
    df = amplify(df)
    lentrain = df[df['date'] == traindate].index[0] + 1

    print df.iloc[lentrain-1 , :]

    # raw_input('\t\tPress')

    print 'lentrain = ' , lentrain
    # raw_input('\t\tPress any ')

    df['y' ] = df['cnt']




    holidays = df[df['holiday'] == 1][['ds', 'holiday_name']]
    holidays.columns = ['ds','holiday']
    # holidays['holiday'] = 'off'
    holidays['lower_window'] = -1
    holidays['upper_window'] = 1
    print df.columns


    m = Prophet(changepoint_prior_scale=0.05, holidays=holidays)

    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    add_reg = False
    if add_reg == True:
        m.add_regressor('lastwork')
        m.add_regressor('lastholiday')
        m.add_regressor('nextwork')
        m.add_regressor('nextholiday')

    m.fit(df.iloc[:lentrain, :]) # train

    # span = 250
    #
    #
    # cv = cross_validation(m , horizon='{0} days'.format(span))
    #
    #
    # print cv['cutoff'].unique()
    #
    # cv['error'] = 1./span * (cv['yhat'] - cv['y'] ) ** 2
    #
    # print cv.groupby('cutoff')['error'].sum()
    #
    # sys.exit(0)

    future = m.make_future_dataframe(freq='D', periods=500)
    future  =  pd.merge( future, df , on ='ds')
    print future.head() ,future.tail()

    forecast =  m.predict(future)  # type:pd.DataFrame  ds ,yhat

    #m.plot_components(forecast)
    #m.plot(forecast)

    testdf = df.iloc[lentrain:, :] # testB lentrain: is ok .!
    testdf = pd.merge(testdf,  forecast, on='ds')

    testdf['yhat'] =  testdf['yhat'].clip_lower(0)

    #day = datetime.datetime.now().strftime('%Y%m%d')
    day = 20180226

    print testdf[testdf['date'] != -1][['yhat']].describe()
    testdf[testdf['date'] != -1][['date', 'yhat']].to_csv('./submit/{day}.reg.{add}.csv'.format(day=day,add=add_reg),sep='\t', header=False,index=False)
    testdf[testdf['date'] != -1][['date', 'ds', 'day_of_week', 'yhat', 'holiday','holiday_name']].to_csv('./submit/{day}.withdate.{add}.csv'.format(day=day,add=add_reg), sep='\t', header=False,
                                                          index=False)
    #plt.show()
