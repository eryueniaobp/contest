# encoding=utf-8


import pandas as pd
import numpy as np
import datetime



import  matplotlib.pyplot as plt

def fore_holidays():
    """
    节后第二天 标注出来；方便直接处理.
    :param df:
    :return:
    """
    fd = 'fd'
    spring = 'spring'
    qingming = 'qingming'
    laodong = 'laodong'
    duanwu = 'duanwu'
    zhongqiu = 'zhongqiu'

    guoqing = 'guoqing'

    other = 'other'

    hdays = [

        ('2016-06-09', 3, duanwu),
        ('2016-09-15', 3, zhongqiu),
        ('2016-10-01', 7, guoqing),
        ('2016-12-31', 3, fd),

        ('2017-01-27', 7, spring),
        ('2017-04-02', 3, qingming),
        ('2017-04-29', 3, laodong),
        ('2017-05-28', 3, duanwu),
        ('2017-10-01', 8, guoqing),
        ('2017-12-30', 3, fd),
        ('2018-02-15', 7, spring),

    ]
    def expandfore(buf,fore):
        nb = {}

        fb = {}
        for day, span, tag in buf:
            nday = datetime.datetime.strptime(day, '%Y-%m-%d') + datetime.timedelta(days=span + fore - 1)
            nday = nday.strftime('%Y-%m-%d')
            nb[nday] = tag


        return nb

    foredays = expandfore(hdays,1)

    return foredays


def leak_check():
    # day = datetime.datetime.now().strftime('%Y%m%d')
    day = 20180304


    df = pd.read_csv('./submit/{}.csv'.format(day), header=0, sep='\t')
    aux = pd.read_csv('./train.csv',header=0)

    df = pd.merge( df ,  aux[['date', 'ds' , 'holiday', 'holiday_name']] , on ='date', how='left')
    df.to_csv('./submit/{}.train.all.csv'.format(day),index=False)

    pred_sum_df =  df.groupby('ds')['cnt'].sum().reset_index()

    real_sum_df = pd.read_csv('./data/real_sum.csv',header=0)

    mdf = pd.merge(pred_sum_df, real_sum_df, on = 'ds')

    mdf['rate'] = mdf['cnt_y']/(mdf['cnt_x']+1) # real/pred.

    mdf[['ds','rate']].to_csv('leak.rate.csv',header=True,index=False)


def hand_check2():
    # day = datetime.datetime.now().strftime('%Y%m%d')
    day = 20180303

    df = pd.read_csv('./submit/{}.csv'.format(day), header=0, sep='\t')
    aux = pd.read_csv('./train.csv',header=0)


    df = pd.merge( df ,  aux[['date', 'ds' , 'holiday', 'holiday_name']] , on ='date', how='left')






    df.to_csv('./submit/{}.train.all.csv'.format(day),index=False)

    rate = pd.read_csv('leak.rate.csv',header=0)

    df =  pd.merge(df, rate, on ='ds')

    plt.plot(df['cnt'])
    print df['cnt'].describe()
    print '--' * 8
    df['cnt'] = df['cnt'] * df['rate']

    plt.plot(df['cnt'], label='leak')
    plt.legend()
    plt.show()
    print df['cnt'].describe()


    """
       fd = 'fd'
    spring = 'spring'
    qingming = 'qingming'
    laodong = 'laodong'
    duanwu = 'duanwu'
    zhongqiu = 'zhongqiu'

    guoqing = 'guoqing'

    """


    df['cnt'] = df[['holiday_name', 'cnt']].apply(
        lambda row: min(5,  row['cnt']) if row['holiday_name'] in ['fd', 'spring', 'qingming', 'laodong','duanwu','zhongqiu','guoqing'] else row['cnt'], axis=1)


    #########fore holidays
    foredays = fore_holidays()
    print foredays
    print '\n\n Before foredays'
    print df['cnt'].describe()

    print df[ (df['brand'] == 5) & (df['ds'].isin( foredays.keys()))]
    #df['cnt'] = df[['ds' , 'brand', 'cnt']].apply(lambda  row : 1000  if  row['brand'] == 5 and row['ds'] in foredays else row['cnt'] ,axis=1 )


    print '\n\n'
    print df['cnt'].describe()


    df[['date','brand', 'cnt']].to_csv('./submit/{}.hand.reg.txt'.format(day) ,index=False, header=False, sep='\t')
    df[['date','brand', 'cnt', 'ds', 'holiday_name']].to_csv('./submit/{}.withdate.csv'.format(day) ,index=False, header=False, sep='\t')


def hand_check():
    df = pd.read_csv('20180226.withdate.False.csv',header=None , sep='\t')
    # df =df.iloc[1: ,:]
    df.columns = ['date', 'ds', 'dayofweek', 'score', 'holiday', 'holiday_name']


    print df.shape

    print df['score'].describe()

    df['score'] =df[['holiday_name','score']].apply(lambda row: row['score'] if row['holiday_name'] in ['normal', 'saturday','sunday','work'] else 30, axis=1)

    print df['score'].describe()

    print df[~df['holiday_name'].isin(['normal', 'saturday','sunday','work'])].shape



    print df.shape

    df[['date', 'score']].to_csv('hand.reg.csv', index=False, header=False, sep='\t')



if __name__ =='__main__':

    # hand_check()
    hand_check2()

    # leak_check()
    # main()
    # infer()
