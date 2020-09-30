from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import tqdm
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



df = pd.read_csv('data/dataset.csv',header=0)
print(df.dtypes)
df['nsTime'] = pd.to_datetime(df['Time'])
print(df.dtypes)
# input("Press any key to continue..")
units = df.groupby(['termNum', 'distNum', 'blockNum', 'powerNum']).count().reset_index()

for idx, row in tqdm.tqdm(units.iterrows()):
    subdf = df[(df['termNum'] == row['termNum']) & (df['distNum'] == row['distNum']) & (df['blockNum'] == row['blockNum']) & (
                df['powerNum'] == row['powerNum'])]
    # print(row[['termNum', 'distNum', 'blockNum', 'powerNum']], '\n' , subdf.shape, subdf.dtypes)
    subdf = subdf.sort_values(by='nsTime')
    subdf['ma7'] = subdf['Power'].rolling(window=7).mean().fillna(0)
    subdf['ma21'] = subdf['Power'].rolling(window=21).mean().fillna(0)

    subdf['26_ema'] = subdf['Power'].ewm(span=26).mean().fillna(0)
    subdf['12_ema'] = subdf['Power'].ewm(span=12).mean().fillna(0)

    subdf['macd'] = subdf['12_ema'] - subdf['26_ema']

    window = 21
    no_of_std = 2

    rolling_mean = subdf['Power'].rolling(window).mean()
    rolling_std = subdf['Power'].rolling(window).std()

    subdf['BHigh'] = (rolling_mean + (rolling_std * no_of_std)).fillna(0)
    subdf['SuperHigh'] = (rolling_mean + (rolling_std * 3)).fillna(0)
    subdf['LHigh'] = (rolling_mean - (rolling_std * no_of_std)).fillna(0)

    subdf['ema'] = subdf['Power'].ewm(com=0.5).mean()

    subdf['momentum'] = subdf['Power'] - 1

    # print(subdf.head())


    train_dataset = subdf[['Power', 'ma7', 'ma21', 'macd', 'ema', 'BHigh', 'LHigh']]
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



