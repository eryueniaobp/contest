import pandas as pd
import numpy as np
import datetime





def hand_check2():
    df = pd.read_csv('./submit/20180301.csv', header=0, sep='\t')
    aux = pd.read_csv('./train.csv',header=0)

    day = datetime.datetime.now().strftime('%Y%m%d')
    df = pd.merge( df ,  aux[['date', 'ds' , 'holiday', 'holiday_name']] , on ='date', how='left')

    df.to_csv('./submit/{}.train.all.csv'.format(day),index=False)
    print df['cnt'].describe()


    df['cnt'] = df[['holiday_name', 'cnt']].apply(
        lambda row: row['cnt'] if row['holiday_name'] in ['normal', 'saturday', 'sunday', 'work'] else min(5,  row['cnt']), axis=1)

    print '\n\n'
    print df['cnt'].describe()


    df[['date','brand', 'cnt']].to_csv('./submit/{}.hand.reg.txt'.format(day) ,index=False, header=False, sep='\t')
    df[['date','brand', 'cnt', 'ds', 'holiday_name']].to_csv('./submit/{}.withdate.csv'.format(day) ,index=False, header=False, sep='\t')
