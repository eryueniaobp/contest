# encoding=utf-8



import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv('train.chusai.csv',header=0)
df1 = pd.read_csv('train.csv',header=0)


mdf = pd.merge(df, df1,  on='ds', how='left')

print mdf[mdf['ds'] == '2016-05-01'].index[0]

mdf['ratio'] = mdf['cnt_y']/mdf['cnt_x']  #fusai/chusai  = 1.44

mdf['cnt_y'] = mdf['cnt_x'] * 1.44


mdf[['ds', 'cnt_y']].to_csv('data/real_sum.csv',index=False)






pdf = mdf[1215:]

plt.plot(pdf['ds'], pdf['cnt_x'])
plt.plot(pdf['ds'], pdf['cnt_y'])
print mdf['ratio'].describe()

"""
count    1258.000000
mean             inf
std              NaN
min         0.000000
25%         1.213120
50%         1.441673
75%         1.749564
max              inf
Name: ratio, dtype: float64

fusai /chusai  = 1.44 附近

chusai:  answer_a  ->2017-02-17 about .

fusai:  a -> 2016-10-09 about .


"""

mdf.plot('ds', 'ratio')

mdf.plot('ds', ['cnt_x','cnt_y'])



plt.show()


