# encoding=utf-8




import pandas as pd
import numpy as np



def hand_check():
    df = pd.read_csv('./submit/20180226.withdate.False.csv',header=None , sep='\t')
    # df =df.iloc[1: ,:]
    df.columns = ['date', 'ds', 'dayofweek', 'score', 'holiday', 'holiday_name']


    print df.shape

    print df['score'].describe()

    df['score'] =df[['holiday_name','score']].apply(lambda row: row['score'] if row['holiday_name'] in ['normal', 'saturday','sunday','work'] else 30, axis=1)

    print df['score'].describe()

    print df[~df['holiday_name'].isin(['normal', 'saturday','sunday','work'])].shape



    print df.shape

    df[['date', 'score']].to_csv('./submit/hand.reg.csv', index=False, header=False, sep='\t')



if __name__ =='__main__':
    hand_check()
