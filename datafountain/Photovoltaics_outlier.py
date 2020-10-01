from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import tqdm
pd.set_option('display.max_columns',None)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
# clf = IsolationForest(n_estimators=10, warm_start=True)
# clf.fit(X)  # fit 10 trees
# clf.set_params(n_estimators=20)  # add 10 more trees
# clf.fit(X)  # fit the added trees
#
#
#
# clf.predict([-1,-1])
def build_label(df, ys):
    """
    异常数据是由光伏逆变器运行过程与设计运行工况出现较大偏离时产生，此处异常数据可分为3类：
    1）非正常0值数据，标记为“-1”；  =>
    2）超量程数据，标记为“-2”；      =>
    3）偏离正常数据，且非0值非超量程数据，标记为“-3”。  => 比较小的

    :param df:
    :param y:
    :return:
    """
    buf = []
    for idx, row in df[['Power', 'SuperHigh', 'BHigh', 'LHigh', 'y']].iterrows():
        power, super_high, bhigh, lhigh, y = row['Power'], row['SuperHigh'], row['BHigh'], row['LHigh'], row['y']
        if y == 1:
            buf += [0]
        else:
            # y == -1

            if power == 0.0:
                buf += [-1]
            else:
                if power > super_high:
                    buf += [-2]
                else:
                    buf += [-3]

    return buf

def build_block(df, row):
    p1 = df[(df['termNum'] == row['termNum']) & (df['distNum'] == row['distNum']) & (df['blockNum'] == row['blockNum']) & (
            df['powerNum'] == 1)].sort_values(by='nsTime').reset_index()
    p2 = df[(df['termNum'] == row['termNum']) & (df['distNum'] == row['distNum']) & (df['blockNum'] == row['blockNum']) & (
                df['powerNum'] == 2)].sort_values(by='nsTime').reset_index()
    p3 = df[(df['termNum'] == row['termNum']) & (df['distNum'] == row['distNum']) & (df['blockNum'] == row['blockNum']) & (
                df['powerNum'] == 3)].sort_values(by='nsTime').reset_index()
    import itertools
    # assert p1.shape[0] == p2.shape[0] and p1.shape[0] == p3.shape[0], '{} {} {} '.format(p1.shape, p2.shape, p3.shape)
    # bdf = pd.concat([p1, p2, p3], axis=1, join='inner')
    if p1.shape[0] == p2.shape[0] and p1.shape[0] == p3.shape[0]:
        bdf = pd.concat([p1, p2, p3], axis=1, join='inner')[['powerNum', 'termNum', 'distNum', 'blockNum', 'Power']]

        bdf.columns = list(itertools.chain.from_iterable([['{}0'.format(i), '{}1'.format(i), '{}2'.format(i)] for i in
                                                          ['powerNum', 'termNum', 'distNum', 'blockNum', 'Power']]))

        bdf['PowerAvg'] = (bdf['Power0'] + bdf['Power1'] + bdf['Power2']) / 3.0

        return 1, bdf[['termNum0', 'distNum0', 'blockNum0', 'PowerAvg']]

    else:
        p12 = pd.merge(p1, p2, suffixes=('_1', '_2'), how='outer', on='Time')
        p123 = pd.merge(p12, p3, how='outer', on='Time')

        cols = list(itertools.chain.from_iterable([['{}'.format(i), '{}_1'.format(i), '{}_2'.format(i)] for i in
                                                      ['powerNum', 'termNum', 'distNum', 'blockNum', 'Power']]))
        bdf = p123[cols + ['Time']]

        # print(bdf.shape, p1.shape, p2.shape, p3.shape, bdf.columns)
        # input("press any key ... ")



        bdf.columns = list(itertools.chain.from_iterable([['{}0'.format(i), '{}1'.format(i), '{}2'.format(i)] for i in
                                                      ['powerNum', 'termNum', 'distNum', 'blockNum', 'Power']])) + ['Time']

        bdf['PowerAvg'] = (bdf['Power0'] + bdf['Power1'] + bdf['Power2'])/3.0

        return 0, bdf[['termNum0', 'distNum0', 'blockNum0', 'Time', 'PowerAvg']]




df = pd.read_csv('data/dataset.csv',header=0)
print(df.dtypes)
df['nsTime'] = pd.to_datetime(df['Time'])
print(df.dtypes)
# input("Press any key to continue..")
units = df.groupby(['termNum', 'distNum', 'blockNum', 'powerNum']).count().reset_index()
na_value = 0
for idx, row in tqdm.tqdm(units.iterrows()):
    subdf = df[(df['termNum'] == row['termNum']) & (df['distNum'] == row['distNum']) & (df['blockNum'] == row['blockNum']) & (
                df['powerNum'] == row['powerNum'])].sort_values(by='nsTime').reset_index()
    # print(row[['termNum', 'distNum', 'blockNum', 'powerNum']], '\n' , subdf.shape, subdf.dtypes)
    # subdf = subdf

    same, block_df = build_block(df, row)

    block_df['block_ma7'] = block_df['PowerAvg'].rolling(window=7).mean().fillna(na_value)
    block_df['block_ma21'] = block_df['PowerAvg'].rolling(window=21).mean().fillna(na_value)


    subdf['ma7'] = subdf['Power'].rolling(window=7).mean().fillna(na_value)
    subdf['ma21'] = subdf['Power'].rolling(window=21).mean().fillna(na_value)

    subdf['26_ema'] = subdf['Power'].ewm(span=26).mean().fillna(na_value)
    subdf['12_ema'] = subdf['Power'].ewm(span=12).mean().fillna(na_value)

    subdf['macd'] = subdf['12_ema'] - subdf['26_ema']

    window = 21
    no_of_std = 2

    rolling_mean = subdf['Power'].rolling(window).mean()
    rolling_std = subdf['Power'].rolling(window).std()

    subdf['BHigh'] = (rolling_mean + (rolling_std * no_of_std)).fillna(na_value)
    subdf['SuperHigh'] = (rolling_mean + (rolling_std * 3)).fillna(na_value)
    subdf['LHigh'] = (rolling_mean - (rolling_std * no_of_std)).fillna(na_value)

    subdf['ema'] = subdf['Power'].ewm(com=0.5).mean().fillna(na_value)

    print(idx, ' is na = ' , subdf['ema'].isna().sum())
    # input("Press any ..")

    subdf['momentum'] = subdf['Power'] - 1
    subdf['power_by_ma7'] = subdf['Power']/(subdf['ma7'] + 0.001)
    subdf['power_by_ma21'] = subdf['Power']/(subdf['ma21'] + 0.001)
    subdf['power_by_ema'] = subdf['Power']/(subdf['ema'] + 0.001)

    # print(subdf[['power_by_ma7', 'power_by_ma21', 'power_by_ema']].describe())
    # input("Press nay ")

    if same == 1:

        subdf = pd.concat([subdf.reset_index(), block_df.reset_index()], axis=1).fillna(0)
    else:
        subdf2 = pd.merge(subdf, block_df, how='left', on='Time').fillna(0)
        # assert
        # print('subdf2 = ', subdf2.shape)
        # print('subdf = ', subdf.shape)
        # print('blockdf = ', block_df.shape)
        # input("Press any key.. ")
        subdf =subdf2
    # print(subdf.shape)

    # print(subdf.head())

    # input("Press any key..")

    subdf['power_by_avg'] = subdf['Power']/(subdf['PowerAvg'] + 0.001)
    subdf['power_by_block_ma7'] = subdf['Power']/(subdf['block_ma7'] + 0.001)
    subdf['power_by_block_ma21'] = subdf['Power']/(subdf['block_ma21'] + 0.001)


    # train_dataset = subdf[['Power', 'ma7', 'ma21', 'macd', 'ema', 'BHigh', 'LHigh',
    #                        'power_by_ma7', 'power_by_ma21', 'power_by_ema',
    #                        'power_by_avg', 'power_by_block_ma7', 'power_by_block_ma21'
    #                        ]]

    train_dataset = subdf[[
                            # 'Power', 'ma7', 'ma21', 'macd', 'ema', 'BHigh', 'LHigh',
                           'power_by_ma7',
                           'power_by_ma21',
                           'power_by_ema',
                           'power_by_avg',
                           'power_by_block_ma7',
                           'power_by_block_ma21'
                           ]]

    clf = IsolationForest(n_estimators=10, warm_start=True)
    clf.fit(train_dataset)  # fit 10 trees

    y = clf.predict(train_dataset)

    # print(y)
    subdf['y'] = y
    subdf['Label'] = build_label(subdf, y)
    import os
    os.makedirs('./submit', exist_ok=True)
    subdf[['termNum', 'distNum', 'blockNum', 'Time', 'powerNum', 'Label']].to_csv('submit/{}.csv'.format(idx), header=False, index=False)
    # input("Press nay key ..")


    # input("Press ")
    """ make sure sorted by Time"""



