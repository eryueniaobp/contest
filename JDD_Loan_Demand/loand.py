# encoding=utf-8
"""
@author : pengalg

信贷需求预测
"""

import numpy as np, traceback
import pandas as pd
import os,pickle,math
import re,datetime,time,sys,logging
from feature import Feature,StatFeature,SuperStatFeature


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='loan.log',
                    filemode='a')


# t_order ='/home/mi/zeroplan/t_order_rec.csv'
# t_loan = '/home/mi/zeroplan/t_loan_rec.csv'
# t_loan_sum = '/home/mi/zeroplan/t_loan_sum2_rec.csv'

def getFullSet():
    """
    这里恰好有序，比较奇怪
    :return:
    """
    full =set()
    if sys.argv[1] == 'train':
        df = pd.read_csv(t_loan_sum,header= 0)

        for uid in df['uid'].unique():
            full.add(int(uid))

    else:
        df = pd.read_csv(t_user, header=0)

        for uid in df['uid'].unique():
            full.add(int(uid))

    l = list(full)
    l.sort()
    return l #统一都排序

def hitPrefix(buf, fealist):
    hit = False
    for prefix in buf:
        for fea in fealist:
            if fea.prefix == prefix and fea.drop == False:
                hit = True
    logging.info('hit {0} : {1}'.format(buf, hit))
    return hit

if len(sys.argv) == 1:
    sys.argv = ['', 'train']

print '-'*10 , sys.argv[1]

logging.info('begin')

t_user ='/home/mi/zeroplan/t_user.csv'
t_order ='/home/mi/zeroplan/t_order.csv'
t_order_svd = '/home/mi/zeroplan/t_order.svd.csv'


t_click = '/home/mi/zeroplan/t_click.csv'
t_click_svd = '/home/mi/zeroplan/t_click.svd.csv'

t_loan = '/home/mi/zeroplan/t_loan.csv'

t_loan = '/home/mi/PycharmProjects/redplan/loan-demand/snippet/t_loan_consecutive.csv'  #加入了consecutive信息

t_loan_svd = '/home/mi/zeroplan/t_loan.svd.csv'
t_loan_ts = '/home/mi/zeroplan/t_loan_ts_train.csv'
#t_loan = '/home/mi/zeroplan/t_loan_withlimit.csv'

t_loan_sum = '/home/mi/zeroplan/t_loan_sum.csv'
t_loan_sum = '/home/mi/zeroplan/t_loan_sum2.csv' #11.csv

t_loan_cnt = './11.csv.cnt'

t_loan_sum_10 = '10.csv'

train_month = ['08','09','10']
train_prev = '10'
train_prev2 = '09'
train_next = '11'
train_next_day = '2016-11-01'

test_month = ['08','09','10','11']
test_prev = '11'
test_prev2= '10'
test_next = '12'
test_next_day = '2016-12-01'

trainfile = '/home/mi/zeroplan/sample.txt'
testfile= '/home/mi/zeroplan/test.txt'
testidfile= '/home/mi/zeroplan/test.id.txt'


if sys.argv[1] == 'test':
    #addFeature的时候需要区分
    train_month = test_month
    train_prev = test_prev
    train_prev2 = test_prev2
    train_next = test_next
    train_next_day = test_next_day
    t_loan_ts = '/home/mi/zeroplan/t_loan_ts_test.csv'

elif sys.argv[1] == 'test1':
    #addFeature的时候需要区分
    train_month = ['09','10','11']
    train_prev = '11'
    train_prev2 = '10'
    train_next = '12'
    testfile = '/home/mi/zeroplan/test.txt.1'
elif sys.argv[1] == 'test2':
    #addFeature的时候需要区分
    train_month = ['10','11']
    train_prev = '11'
    train_prev2 = '10'
    train_next = '12'
    testfile = '/home/mi/zeroplan/test.txt.2'

elif sys.argv[1] == 'train10':
    train_month = ['08','09']
    train_prev = '09'
    train_prev2= '08'
    train_next = '10'

    t_loan_sum = t_loan_sum_10
    trainfile = '/home/mi/zeroplan/sample.10.txt'
    sys.argv[1] = 'train' #保证后续逻辑正确



fullset = getFullSet()
print 'fullset size = ' ,len(fullset)

feature_list, stat_list , superstat_list = [], [], []

prefix_set = []
def initFeatureList():

    feature_list.extend([

    # Feature(prefix='uid', startid=1, name='uid',drop=False), #magic user . 最后阶段可以用这个提升一下.
    Feature(prefix='age', startid=1, name='age',drop=True),

    Feature(prefix='sex', startid=1, name='sex', drop=True),
    Feature(prefix='age_sex', startid=1, name='age_sex',drop=True),


    #active_date:激活日期
    #加上drop=False 反而下降了
    Feature(prefix='active_date', startid=1, name='active_date',drop=True),

    #实测有效
    Feature(prefix='days_to_now', startid=1, name='days_to_now', drop=False),
    Feature(prefix='limit', startid=1, name='limit',drop=False),

    # rank feature

    # large consume count 
    Feature(prefix='buy_large1_count', startid= 1, name = 'buy_large1_count') , #sum
    Feature(prefix='buy_large2_count', startid= 1, name = 'buy_large2_count') , #sum
    Feature(prefix='buy_large3_count', startid= 1, name = 'buy_large3_count') , #sum
    Feature(prefix='buy_large4_count', startid= 1, name = 'buy_large4_count') , #sum

    #品类购买金额
    Feature(prefix='buy_cate', startid= 1, name = 'buy_cate') , #sum
    Feature(prefix='buy_cate_count', startid= 1, name = 'buy_cate_count') ,
    # Feature(prefix='buy_cate_min', startid= 1, name = 'buy_cate_min') ,
    # Feature(prefix='buy_cate_max', startid= 1, name = 'buy_cate_max') ,
    # Feature(prefix='buy_cate_mad', startid= 1, name = 'buy_cate_mad') ,
    # Feature(prefix='buy_cate_mean', startid= 1, name = 'buy_cate_mean') ,
    # Feature(prefix='buy_cate_std', startid= 1, name = 'buy_cate_std') ,
    # Feature(prefix='buy_cate_skew', startid= 1, name = 'buy_cate_skew') ,

    Feature(prefix='buy_sum', startid=1, name='buy_sum'),
    Feature(prefix='buy_count', startid=1, name='buy_count'),
    # Feature(prefix='buy_mean', startid=1, name='buy_mean'),
    # Feature(prefix='buy_min', startid=1, name='buy_min'),
    # Feature(prefix='buy_max', startid=1, name='buy_max'),
    # Feature(prefix='buy_mad', startid=1, name='buy_mad'),
    # Feature(prefix='buy_skew', startid=1, name='buy_skew'),


    Feature(prefix='buy_cate_discount', startid=1, name='buy_cate_discount'),
    Feature(prefix='buy_discount_sum', startid=1, name='buy_discount_sum'),

    # Feature(prefix='svd_order_param1', startid=1, name='svd_order_param1'),
    # Feature(prefix='svd_order_param2', startid=1, name='svd_order_param2'),
    # Feature(prefix='svd_order_param3', startid=1, name='svd_order_param3'),
    # Feature(prefix='svd_order_param4', startid=1, name='svd_order_param4'),
    # Feature(prefix='svd_order_param5', startid=1, name='svd_order_param5'),

    Feature(prefix='click', startid=1,name='click', drop=False) , #历史点击某个商品的次数
    Feature(prefix='click_param', startid=1, name='click_param', drop=False),  # 历史点击某个商品+param的次数

    ##针对click的svd处理
    # Feature(prefix='svd_click_param1',startid=1, name='svd_click_param1'),
    # Feature(prefix='svd_click_param2',startid=1, name='svd_click_param2'),
    # Feature(prefix='svd_click_param3',startid=1, name='svd_click_param3'),
    # Feature(prefix='svd_click_param4',startid=1, name='svd_click_param4'),
    # Feature(prefix='svd_click_param5',startid=1, name='svd_click_param5'),

    # Feature(prefix='svd_loan_param1', startid=1, name='svd_loan_param1'),
    # Feature(prefix='svd_loan_param2', startid=1, name='svd_loan_param2'),
    # Feature(prefix='svd_loan_param3', startid=1, name='svd_loan_param3'),
    # Feature(prefix='svd_loan_param4', startid=1, name='svd_loan_param4'),
    # Feature(prefix='svd_loan_param5', startid=1, name='svd_loan_param5'),

    Feature(prefix='ts_loan_mean',startid=1, name='ts_loan_mean') ,
    Feature(prefix='ts_loan_mad', startid=1, name='ts_loan_mad'),
    Feature(prefix='ts_loan_skew', startid=1, name='ts_loan_skew'),
    Feature(prefix='ts_loan_kurt', startid=1, name='ts_loan_kurt'),



    # Feature(prefix='order_click', startid=1,name='order_click'),




    ])
def initStatFeatureList():
    """
    最近一个月的统计贷款数
    :return:
    """
    stat_list.extend([
        #上一个月的借贷均值
        StatFeature(prefix='age',startid=1,  expand=True,name='age' , idfile='user.id', drop=True),
        StatFeature(prefix='sex',startid=1,  expand=True, name='sex', idfile='user.id', drop=True),
        StatFeature(prefix='active_date', startid=1,  expand=True, name ='active_date',  idfile='user.id' ,drop=True),

        #limit stat特征 重要度低  这个特征也基本没用，后续废弃
        StatFeature(prefix='limit', startid=1, expand=True, name='limit', idfile='user.id',drop=True),
        StatFeature(prefix='age_sex',startid=1, expand=True , name= 'age_sex', idfile='user.id', drop=True),
        # 类别购买的平均贷款额度
        StatFeature(prefix='buy_cate_avg_loan', startid=1,  expand=True, name='buy_cate_avg_loan', idfile='buy_cate.id', drop=True),
        #lookback stack
        StatFeature(prefix='lookback_stack_sum1',startid=1,name='lookback_stack_sum1', expand=False),
        StatFeature(prefix='lookback_stack_sum2',startid=1,name='lookback_stack_sum2',expand=False),
        StatFeature(prefix='lookback_stack_sum3',startid=1,name='lookback_stack_sum3',expand=False),
        StatFeature(prefix='lookback_stack_sum4',startid=1,name='lookback_stack_sum4',expand=False),
        StatFeature(prefix='lookback_stack_sum5',startid=1,name='lookback_stack_sum5',expand=False),
        StatFeature(prefix='lookback_stack_sum6',startid=1,name='lookback_stack_sum6',expand=False),
        StatFeature(prefix='lookback_stack_sum7',startid=1,name='lookback_stack_sum7',expand=False),
        StatFeature(prefix='lookback_stack_sum8',startid=1,name='lookback_stack_sum8',expand=False),
        StatFeature(prefix='lookback_stack_sum9',startid=1,name='lookback_stack_sum9',expand=False),
        StatFeature(prefix='lookback_stack_sum10',startid=1,name='lookback_stack_sum10',expand=False),
        StatFeature(prefix='lookback_stack_sum11',startid=1,name='lookback_stack_sum11',expand=False),
        StatFeature(prefix='lookback_stack_sum12',startid=1,name='lookback_stack_sum12',expand=False),
        StatFeature(prefix='lookback_stack_sum13',startid=1,name='lookback_stack_sum13',expand=False),
        StatFeature(prefix='lookback_stack_sum14',startid=1,name='lookback_stack_sum14',expand=False),
        StatFeature(prefix='lookback_stack_sum15',startid=1,name='lookback_stack_sum15',expand=False),
        StatFeature(prefix='lookback_stack_sum16',startid=1,name='lookback_stack_sum16',expand=False),
        StatFeature(prefix='lookback_stack_sum17',startid=1,name='lookback_stack_sum17',expand=False),
        StatFeature(prefix='lookback_stack_sum18',startid=1,name='lookback_stack_sum18',expand=False),
        StatFeature(prefix='lookback_stack_sum19',startid=1,name='lookback_stack_sum19',expand=False),
        #lookback 
        StatFeature(prefix='lookback1_loan_sum', startid=1, name='lookback1_loan_sum', expand=False),
        StatFeature(prefix='lookback2_loan_sum', startid=1, name='lookback2_loan_sum', expand=False),
        StatFeature(prefix='lookback3_loan_sum', startid=1, name='lookback3_loan_sum', expand=False),
        StatFeature(prefix='lookback4_loan_sum', startid=1, name='lookback4_loan_sum', expand=False),
        StatFeature(prefix='lookback5_loan_sum', startid=1, name='lookback5_loan_sum', expand=False),
        StatFeature(prefix='lookback6_loan_sum', startid=1, name='lookback6_loan_sum', expand=False),
        StatFeature(prefix='lookback7_loan_sum', startid=1, name='lookback7_loan_sum', expand=False),
        StatFeature(prefix='lookback8_loan_sum', startid=1, name='lookback8_loan_sum', expand=False),
        StatFeature(prefix='lookback9_loan_sum', startid=1, name='lookback9_loan_sum', expand=False),
        StatFeature(prefix='lookback10_loan_sum', startid=1, name='lookback10_loan_sum', expand=False),
        StatFeature(prefix='lookback11_loan_sum', startid=1, name='lookback11_loan_sum', expand=False),
        StatFeature(prefix='lookback12_loan_sum', startid=1, name='lookback12_loan_sum', expand=False),
        StatFeature(prefix='lookback13_loan_sum', startid=1, name='lookback13_loan_sum', expand=False),
        StatFeature(prefix='lookback14_loan_sum', startid=1, name='lookback14_loan_sum', expand=False),
        StatFeature(prefix='lookback15_loan_sum', startid=1, name='lookback15_loan_sum', expand=False),
        StatFeature(prefix='lookback16_loan_sum', startid=1, name='lookback16_loan_sum', expand=False),
        StatFeature(prefix='lookback17_loan_sum', startid=1, name='lookback17_loan_sum', expand=False),
        StatFeature(prefix='lookback18_loan_sum', startid=1, name='lookback18_loan_sum', expand=False),
        StatFeature(prefix='lookback19_loan_sum', startid=1, name='lookback19_loan_sum', expand=False),
        StatFeature(prefix='lookback20_loan_sum', startid=1, name='lookback20_loan_sum', expand=False),
        StatFeature(prefix='lookback21_loan_sum', startid=1, name='lookback21_loan_sum', expand=False),
        StatFeature(prefix='lookback22_loan_sum', startid=1, name='lookback22_loan_sum', expand=False),
        StatFeature(prefix='lookback23_loan_sum', startid=1, name='lookback23_loan_sum', expand=False),
        StatFeature(prefix='lookback24_loan_sum', startid=1, name='lookback24_loan_sum', expand=False),
        StatFeature(prefix='lookback25_loan_sum', startid=1, name='lookback25_loan_sum', expand=False),
        StatFeature(prefix='lookback26_loan_sum', startid=1, name='lookback26_loan_sum', expand=False),
        StatFeature(prefix='lookback27_loan_sum', startid=1, name='lookback27_loan_sum', expand=False),
        StatFeature(prefix='lookback28_loan_sum', startid=1, name='lookback28_loan_sum', expand=False),
        StatFeature(prefix='lookback29_loan_sum', startid=1, name='lookback29_loan_sum', expand=False),
        StatFeature(prefix='lookback30_loan_sum', startid=1, name='lookback30_loan_sum', expand=False),
        StatFeature(prefix='lookback31_loan_sum', startid=1, name='lookback31_loan_sum', expand=False),
        StatFeature(prefix='lookback32_loan_sum', startid=1, name='lookback32_loan_sum', expand=False),
        StatFeature(prefix='lookback33_loan_sum', startid=1, name='lookback33_loan_sum', expand=False),
        StatFeature(prefix='lookback34_loan_sum', startid=1, name='lookback34_loan_sum', expand=False),
        StatFeature(prefix='lookback35_loan_sum', startid=1, name='lookback35_loan_sum', expand=False),
        StatFeature(prefix='lookback36_loan_sum', startid=1, name='lookback36_loan_sum', expand=False),
        StatFeature(prefix='lookback37_loan_sum', startid=1, name='lookback37_loan_sum', expand=False),
        StatFeature(prefix='lookback38_loan_sum', startid=1, name='lookback38_loan_sum', expand=False),
        StatFeature(prefix='lookback39_loan_sum', startid=1, name='lookback39_loan_sum', expand=False),
        StatFeature(prefix='lookback40_loan_sum', startid=1, name='lookback40_loan_sum', expand=False),
        StatFeature(prefix='lookback41_loan_sum', startid=1, name='lookback41_loan_sum', expand=False),
        StatFeature(prefix='lookback42_loan_sum', startid=1, name='lookback42_loan_sum', expand=False),
        StatFeature(prefix='lookback43_loan_sum', startid=1, name='lookback43_loan_sum', expand=False),
        StatFeature(prefix='lookback44_loan_sum', startid=1, name='lookback44_loan_sum', expand=False),
        StatFeature(prefix='lookback45_loan_sum',startid=1,name='lookback45_loan_sum', expand=False),
        StatFeature(prefix='lookback46_loan_sum', startid=1, name='lookback46_loan_sum', expand=False),
        StatFeature(prefix='lookback47_loan_sum', startid=1, name='lookback47_loan_sum', expand=False),
        StatFeature(prefix='lookback48_loan_sum', startid=1, name='lookback48_loan_sum', expand=False),
        StatFeature(prefix='lookback49_loan_sum', startid=1, name='lookback49_loan_sum', expand=False),
        StatFeature(prefix='lookback50_loan_sum', startid=1, name='lookback50_loan_sum', expand=False),
        StatFeature(prefix='lookback51_loan_sum', startid=1, name='lookback51_loan_sum', expand=False),
        StatFeature(prefix='lookback52_loan_sum', startid=1, name='lookback52_loan_sum', expand=False),
        StatFeature(prefix='lookback53_loan_sum', startid=1, name='lookback53_loan_sum', expand=False),
        StatFeature(prefix='lookback54_loan_sum', startid=1, name='lookback54_loan_sum', expand=False),
        StatFeature(prefix='lookback55_loan_sum', startid=1, name='lookback55_loan_sum', expand=False),
        StatFeature(prefix='lookback56_loan_sum', startid=1, name='lookback56_loan_sum', expand=False),
        StatFeature(prefix='lookback57_loan_sum', startid=1, name='lookback57_loan_sum', expand=False),
        StatFeature(prefix='lookback58_loan_sum', startid=1, name='lookback58_loan_sum', expand=False),
        StatFeature(prefix='lookback59_loan_sum', startid=1, name='lookback59_loan_sum', expand=False),


        #
        StatFeature(prefix='lookahead1_repay', startid=1, name='lookahead1_repay', expand=False),
        StatFeature(prefix='lookahead2_repay', startid=1, name='lookahead2_repay', expand=False),
        StatFeature(prefix='lookahead3_repay', startid=1, name='lookahead3_repay', expand=False),
        StatFeature(prefix='lookahead4_repay', startid=1, name='lookahead4_repay', expand=False),
        StatFeature(prefix='lookahead5_repay', startid=1, name='lookahead5_repay', expand=False),
        StatFeature(prefix='lookahead6_repay', startid=1, name='lookahead6_repay', expand=False),
        StatFeature(prefix='lookahead7_repay', startid=1, name='lookahead7_repay', expand=False),
        StatFeature(prefix='lookahead8_repay', startid=1, name='lookahead8_repay', expand=False),
        StatFeature(prefix='lookahead9_repay', startid=1, name='lookahead9_repay', expand=False),
        StatFeature(prefix='lookahead10_repay', startid=1, name='lookahead10_repay', expand=False),
        StatFeature(prefix='lookahead11_repay', startid=1, name='lookahead11_repay', expand=False),
        StatFeature(prefix='lookahead12_repay', startid=1, name='lookahead12_repay', expand=False),
        StatFeature(prefix='lookahead13_repay', startid=1, name='lookahead13_repay', expand=False),
        StatFeature(prefix='lookahead14_repay', startid=1, name='lookahead14_repay', expand=False),
        StatFeature(prefix='lookahead15_repay', startid=1, name='lookahead15_repay', expand=False),
        StatFeature(prefix='lookahead16_repay', startid=1, name='lookahead16_repay', expand=False),
        StatFeature(prefix='lookahead17_repay', startid=1, name='lookahead17_repay', expand=False),
        StatFeature(prefix='lookahead18_repay', startid=1, name='lookahead18_repay', expand=False),
        StatFeature(prefix='lookahead19_repay', startid=1, name='lookahead19_repay', expand=False),
        StatFeature(prefix='lookahead20_repay', startid=1, name='lookahead20_repay', expand=False),
        StatFeature(prefix='lookahead21_repay', startid=1, name='lookahead21_repay', expand=False),
        StatFeature(prefix='lookahead22_repay', startid=1, name='lookahead22_repay', expand=False),
        StatFeature(prefix='lookahead23_repay', startid=1, name='lookahead23_repay', expand=False),
        StatFeature(prefix='lookahead24_repay', startid=1, name='lookahead24_repay', expand=False),
        StatFeature(prefix='lookahead25_repay', startid=1, name='lookahead25_repay', expand=False),
        StatFeature(prefix='lookahead26_repay', startid=1, name='lookahead26_repay', expand=False),
        StatFeature(prefix='lookahead27_repay', startid=1, name='lookahead27_repay', expand=False),
        StatFeature(prefix='lookahead28_repay', startid=1, name='lookahead28_repay', expand=False),
        StatFeature(prefix='lookahead29_repay', startid=1, name='lookahead29_repay', expand=False),
        StatFeature(prefix='lookahead30_repay', startid=1, name='lookahead30_repay', expand=False),
        StatFeature(prefix='lookahead31_repay', startid=1, name='lookahead31_repay', expand=False),
        StatFeature(prefix='lookahead32_repay', startid=1, name='lookahead32_repay', expand=False),
        StatFeature(prefix='lookahead33_repay', startid=1, name='lookahead33_repay', expand=False),
        StatFeature(prefix='lookahead34_repay', startid=1, name='lookahead34_repay', expand=False),
        StatFeature(prefix='lookahead35_repay', startid=1, name='lookahead35_repay', expand=False),
        StatFeature(prefix='lookahead36_repay', startid=1, name='lookahead36_repay', expand=False),
        StatFeature(prefix='lookahead37_repay', startid=1, name='lookahead37_repay', expand=False),
        StatFeature(prefix='lookahead38_repay', startid=1, name='lookahead38_repay', expand=False),
        StatFeature(prefix='lookahead39_repay', startid=1, name='lookahead39_repay', expand=False),
        StatFeature(prefix='lookahead40_repay', startid=1, name='lookahead40_repay', expand=False),
        StatFeature(prefix='lookahead41_repay', startid=1, name='lookahead41_repay', expand=False),
        StatFeature(prefix='lookahead42_repay', startid=1, name='lookahead42_repay', expand=False),
        StatFeature(prefix='lookahead43_repay', startid=1, name='lookahead43_repay', expand=False),
        StatFeature(prefix='lookahead44_repay', startid=1, name='lookahead44_repay', expand=False),
        StatFeature(prefix='lookahead45_repay',startid=1,name='lookahead45_repay', expand=False),

        StatFeature(prefix='lookahead46_repay', startid=1, name='lookahead46_repay', expand=False),
        StatFeature(prefix='lookahead47_repay', startid=1, name='lookahead47_repay', expand=False),
        StatFeature(prefix='lookahead48_repay', startid=1, name='lookahead48_repay', expand=False),
        StatFeature(prefix='lookahead49_repay', startid=1, name='lookahead49_repay', expand=False),
        StatFeature(prefix='lookahead50_repay', startid=1, name='lookahead50_repay', expand=False),
        StatFeature(prefix='lookahead51_repay', startid=1, name='lookahead51_repay', expand=False),
        StatFeature(prefix='lookahead52_repay', startid=1, name='lookahead52_repay', expand=False),
        StatFeature(prefix='lookahead53_repay', startid=1, name='lookahead53_repay', expand=False),
        StatFeature(prefix='lookahead54_repay', startid=1, name='lookahead54_repay', expand=False),
        StatFeature(prefix='lookahead55_repay', startid=1, name='lookahead55_repay', expand=False),
        StatFeature(prefix='lookahead56_repay', startid=1, name='lookahead56_repay', expand=False),
        StatFeature(prefix='lookahead57_repay', startid=1, name='lookahead57_repay', expand=False),
        StatFeature(prefix='lookahead58_repay', startid=1, name='lookahead58_repay', expand=False),
        StatFeature(prefix='lookahead59_repay', startid=1, name='lookahead59_repay', expand=False),

        # rank feature
        StatFeature(prefix = 'loan_sum_rank', startid=1 , name='loan_sum_rank', expand=False),
        #历史贷款总额, 实测有效
        StatFeature(prefix='loan_consecutive_sum_before',startid=1,name='loan_consecutive_sum_before', expand=False),

        #近似的left_limit， 实测有效
        StatFeature(prefix='loan_consecutive_left_limit',startid=1,name='loan_consecutive_left_limit', expand=False),

        StatFeature(prefix='loan_sum_before',startid=1,name='loan_sum_before', expand=False) ,
        # 对数值 求和
        # StatFeature(prefix='loan_sumlog_before', startid=1, name='loan_sumlog_before', expand=False),
    #    StatFeature(prefix='loan_weight_sum_before', startid=1, name='loan_weight_sum_before', expand=False), #加权重的贷款总额

#        StatFeature(prefix='loan_weight_sum_before_0', startid=1, name='loan_weight_sum_before_0', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_1', startid=1, name='loan_weight_sum_before_1', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_2', startid=1, name='loan_weight_sum_before_2', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_3', startid=1, name='loan_weight_sum_before_3', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_4', startid=1, name='loan_weight_sum_before_4', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_5', startid=1, name='loan_weight_sum_before_5', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_6', startid=1, name='loan_weight_sum_before_6', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_7', startid=1, name='loan_weight_sum_before_7', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_8', startid=1, name='loan_weight_sum_before_8', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_9', startid=1, name='loan_weight_sum_before_9', expand=False), #加权重的贷款总额
        # 加权重的贷款总额
        #历史贷款均值
        StatFeature(prefix='loan_avg_before', startid=1, name='loan_avg_before', expand=False),
        # StatFeature(prefix='loan_avglog_before', startid=1, name='loan_avglog_before', expand=False),
        StatFeature(prefix='loan_median_before', startid=1, name='loan_median_before', expand=False),
        StatFeature(prefix='loan_avg_std_before', startid=1, name='loan_avg_std_before', expand=False),

        ##NOTE: todo  half_std  and double_std 与  avg_std 高度相关，加入特征后 并没有明显提升  ,后续可以考虑放弃 plan
        StatFeature(prefix='loan_avg_half_std_before', startid=1, name='loan_avg_half_std_before', expand=False),
        StatFeature(prefix='loan_avg_double_std_before', startid=1, name='loan_avg_double_std_before', expand=False),

        StatFeature(prefix='loan_skew_before', startid=1, name='loan_skew_before', expand=False),
        StatFeature(prefix='loan_kurt_before', startid=1, name='loan_kurt_before', expand=False),

        # StatFeature(prefix='loan_skewlog_before', startid=1, name='loan_skewlog_before', expand=False),
        # StatFeature(prefix='loan_kurtlog_before', startid=1, name='loan_kurtlog_before', expand=False),


        StatFeature(prefix='loan_max_before', startid=1, name='loan_max_before', expand=False),
        StatFeature(prefix='loan_min_before', startid=1, name='loan_min_before', expand=False),

        StatFeature(prefix='loan_mad_before', startid=1, name='loan_mad_before', expand=False),
        # StatFeature(prefix='loan_madlog_before', startid=1, name='loan_madlog_before', expand=False),

        #StatFeature(prefix='loan_diff_before', startid=1, name='loan_diff_before', expand=False), # sum 1st diff will be last - first . 
        #
        #上个月贷款总额
        StatFeature(prefix='loan_sum_previous',startid=1,name='loan_sum_previous', expand=False) ,
        StatFeature(prefix='loan_avg_previous', startid=1, name='loan_avg_previous', expand=False),
        StatFeature(prefix='loan_mad_previous', startid=1, name='loan_mad_previous', expand=False),

        StatFeature(prefix='loan_min_previous', startid=1, name='loan_min_previous', expand=False),
        StatFeature(prefix='loan_max_previous', startid=1, name='loan_max_previous', expand=False),
        StatFeature(prefix='loan_median_previous', startid=1, name='loan_median_previous', expand=False),

        StatFeature(prefix='loan_skew_previous', startid=1, name='loan_skew_previous', expand=False),
        StatFeature(prefix='loan_kurt_previous', startid=1, name='loan_kurt_previous', expand=False),



        # StatFeature(prefix='loan_sumlog_previous', startid=1, name='loan_sumlog_previous', expand=False),
        StatFeature(prefix='loan_sum_previous2', startid=1, name='loan_sum_previous2', expand=False),
        # StatFeature(prefix='loan_sumlog_previous2', startid=1, name='loan_sumlog_previous2', expand=False),


        # limit will be occupied until repayed all the loan_amount
        StatFeature(prefix='limit_in_use', startid=1, name='limit_in_use', expand=False),
        StatFeature(prefix='limit_free', startid=1, name='limit_free', expand=False),
        StatFeature(prefix='limit_use_rate', startid=1, name='limit_use_rate', expand=False),

        # StatFeature(prefix='lookahead_limit_in_use', startid=1, name='lookahead_limit_in_use', expand=False),
        # StatFeature(prefix='lookahead_limit_free', startid=1, name='lookahead_limit_free', expand=False),
        # StatFeature(prefix='lookahead_limit_use_rate', startid=1, name='lookahead_limit_use_rate', expand=False),
        #
        # StatFeature(prefix='lookahead2_limit_in_use', startid=1, name='lookahead2_limit_in_use', expand=False),
        # StatFeature(prefix='lookahead2_limit_free', startid=1, name='lookahead2_limit_free', expand=False),
        # StatFeature(prefix='lookahead2_limit_use_rate', startid=1, name='lookahead2_limit_use_rate', expand=False),
        #
        # StatFeature(prefix='lookahead3_limit_in_use', startid=1, name='lookahead3_limit_in_use', expand=False),
        # StatFeature(prefix='lookahead3_limit_free', startid=1, name='lookahead3_limit_free', expand=False),
        # StatFeature(prefix='lookahead3_limit_use_rate', startid=1, name='lookahead3_limit_use_rate', expand=False),


        StatFeature(prefix='loan_day_min1', startid=1, name = 'loan_day_min1', expand=False),
        StatFeature(prefix='loan_day_max1', startid=1, name = 'loan_day_max1', expand=False),
        StatFeature(prefix='loan_day_mad1', startid=1, name = 'loan_day_mad1', expand=False),
        StatFeature(prefix='loan_day_median1', startid=1, name = 'loan_day_median1', expand=False),

        StatFeature(prefix='loan_day_min3', startid=1, name = 'loan_day_min3', expand=False),
        StatFeature(prefix='loan_day_max3', startid=1, name = 'loan_day_max3', expand=False),
        StatFeature(prefix='loan_day_mad3', startid=1, name = 'loan_day_mad3', expand=False),
        StatFeature(prefix='loan_day_median3', startid=1, name = 'loan_day_median3', expand=False),

        StatFeature(prefix='loan_day_min6', startid=1, name = 'loan_day_min6', expand=False),
        StatFeature(prefix='loan_day_max6', startid=1, name = 'loan_day_max6', expand=False),
        StatFeature(prefix='loan_day_mad6', startid=1, name = 'loan_day_mad6', expand=False),
        StatFeature(prefix='loan_day_median6', startid=1, name = 'loan_day_median6', expand=False),
        StatFeature(prefix='loan_day_min12', startid=1, name = 'loan_day_min12', expand=False),
        StatFeature(prefix='loan_day_max12', startid=1, name = 'loan_day_max12', expand=False),
        StatFeature(prefix='loan_day_mad12', startid=1, name = 'loan_day_mad12', expand=False),
        StatFeature(prefix='loan_day_median12', startid=1, name = 'loan_day_median12', expand=False),
        #贷款时间的统计 active_date -> loan_time
        StatFeature(prefix='loan_day_min', startid=1, name = 'loan_day_min', expand=False),
        StatFeature(prefix='loan_day_max', startid=1, name = 'loan_day_max', expand=False),
        StatFeature(prefix='loan_day_mad', startid=1, name = 'loan_day_mad', expand=False),
        StatFeature(prefix='loan_day_mean', startid=1, name = 'loan_day_mean', expand=False),
        StatFeature(prefix='loan_day_median', startid=1, name = 'loan_day_median', expand=False),
        StatFeature(prefix='loan_day_skew', startid=1, name = 'loan_day_skew', expand=False),

        #  loan_time -> train_next
        StatFeature(prefix='loan_past_day_min', startid=1, name='loan_past_day_min', expand=False),
        StatFeature(prefix='loan_past_day_max', startid=1, name='loan_past_day_max', expand=False),
        StatFeature(prefix='loan_past_day_mad', startid=1, name='loan_past_day_mad', expand=False),
        StatFeature(prefix='loan_past_day_mean', startid=1, name='loan_past_day_mean', expand=False),
        StatFeature(prefix='loan_past_day_median', startid=1, name='loan_past_day_median', expand=False),
        StatFeature(prefix='loan_past_day_skew', startid=1, name='loan_past_day_skew', expand=False),

        # clear经过线上实验，clear都无效 .
        # StatFeature(prefix='clear_loan_past_day_min', startid=1, name='clear_loan_past_day_min', expand=False),
        # StatFeature(prefix='clear_loan_past_day_max', startid=1, name='clear_loan_past_day_max', expand=False),
        # StatFeature(prefix='clear_loan_past_day_mad', startid=1, name='clear_loan_past_day_mad', expand=False),
        # StatFeature(prefix='clear_loan_past_day_mean', startid=1, name='clear_loan_past_day_mean', expand=False),
        # StatFeature(prefix='clear_loan_past_day_median', startid=1, name='clear_loan_past_day_median', expand=False),
        # StatFeature(prefix='clear_loan_past_day_skew', startid=1, name='clear_loan_past_day_skew', expand=False),

        #plannum as categorical feature

        StatFeature(prefix='loan_plannum1_count', startid=1,name='loan_plannum1_count',expand=False),
        StatFeature(prefix='loan_plannum1_sum', startid=1,name='loan_plannum1_sum',expand=False),

        StatFeature(prefix='loan_plannum3_count', startid=1,name='loan_plannum3_count',expand=False),
        StatFeature(prefix='loan_plannum3_sum', startid=1,name='loan_plannum3_sum',expand=False),

        StatFeature(prefix='loan_plannum6_count', startid=1,name='loan_plannum6_count',expand=False),
        StatFeature(prefix='loan_plannum6_sum', startid=1,name='loan_plannum6_sum',expand=False),

        StatFeature(prefix='loan_plannum12_count', startid=1,name='loan_plannum12_count',expand=False),
        StatFeature(prefix='loan_plannum12_sum', startid=1,name='loan_plannum12_sum',expand=False),
        # StatFeature(prefix='loan_day_kurt', startid=1, name = 'loan_day_min', expand=False),

        # ================================================================
        # 截止当前的贷款余额 ，通过 贷款额度 + 分期数 推算
        # loan_balance_plannum是总期数
        StatFeature(prefix='loan_balance', startid=1, name='loan_balance', expand=False),
        StatFeature(prefix='repay_pressure',startid=1, name='repay_pressure', expand=False), #下一次还款总金额
        # balance vs initial limit.
        StatFeature(prefix='loan_left_limit', startid=1, name='loan_left_limit', expand=False),

        StatFeature(prefix= 'lookahead_loan_balance',startid=1, name = 'lookahead_loan_balance',expand=False),
        StatFeature(prefix= 'lookahead_repay_pressure', startid=1, name = 'lookahead_repay_pressure', expand=False),
        StatFeature(prefix= 'lookahead_loan_left_limit' ,startid=1, name='lookahead_loan_left_limit', expand=False),

        StatFeature(prefix='lookahead2_loan_balance', startid=1, name='lookahead2_loan_balance', expand=False),
        StatFeature(prefix='lookahead2_repay_pressure', startid=1, name='lookahead2_repay_pressure', expand=False),
        StatFeature(prefix='lookahead2_loan_left_limit', startid=1, name='lookahead2_loan_left_limit', expand=False),

        StatFeature(prefix='lookahead3_loan_balance', startid=1, name='lookahead3_loan_balance', expand=False),
        StatFeature(prefix='lookahead3_repay_pressure', startid=1, name='lookahead3_repay_pressure', expand=False),
        StatFeature(prefix='lookahead3_loan_left_limit', startid=1, name='lookahead3_loan_left_limit', expand=False),



        # ================================================================



        StatFeature(prefix='repay_mean_amount',startid=1, name='repay_mean_amount', expand=False), #下一次还款 平均金额(下个月可能有多笔贷款)

        # 下下次归还的次数

        # ====================================== 当月会有多少 结清的款项  和 金额 ；比如预测时，12月会有多少结清的款项和金额

        StatFeature(prefix='repay_plannum', startid=1, name='repay_plannum', expand=False),  # 下一次还款的笔数
        StatFeature(prefix='lookahead_repay_plannum', startid=1, name='lookahead_repay_plannum', expand=False),

        StatFeature(prefix='repay_marginal_clear_count', startid=1, name='repay_marginal_clear_count', expand=False),
        StatFeature(prefix='lookahead_repay_marginal_clear_count', startid=1, name='lookahead_repay_marginal_clear_count', expand=False),

        StatFeature(prefix='repay_marginal_clear_amount', startid=1, name='repay_marginal_clear_amount', expand=False),
        StatFeature(prefix='lookahead_repay_marginal_clear_amount', startid=1, name='lookahead_repay_marginal_clear_amount', expand=False),


        StatFeature(prefix='repay_clear_count', startid=1,name='repay_clear_count',expand=False),
        StatFeature(prefix='lookahead_repay_clear_count', startid=1, name='lookahead_repay_clear_count', expand=False),

        StatFeature(prefix='repay_clear_amount', startid=1,name='repay_clear_amount',expand=False),
        StatFeature(prefix='lookahead_repay_clear_amount', startid=1,name='lookahead_repay_clear_amount',expand=False),

        # ======================================

        #  step1:  用最大值近似  近似以后 效果降低了 .
        #  step2:  采用时间和贷款次数近似 .
        #  step3:  采用  left_balance/limit , 效果稳定了，挺好.
    #    StatFeature(prefix='loan_left_magic_limit', startid=1, name='loan_left_magic_limit', expand=False),

        # max/limit 额度占用情况
        StatFeature(prefix='loan_max_rate', startid=1, name='loan_max_rate', expand=False),

        # 上个月的 left_limit情况
        # StatFeature(prefix='loan_previous_left_limit',startid=1, name='loan_previous_left_limit', expand=False),




        #每期的平均额度
        # NOTE:没有提升成绩，还是1.79 ， 后期放弃
        StatFeature(prefix='loan_perplannum_avg_before', startid=1, name='loan_perplannum_avg_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_max_before', startid=1, name='loan_perplannum_max_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_min_before', startid=1, name='loan_perplannum_min_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_mad_before', startid=1, name='loan_perplannum_mad_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_skew_before', startid=1, name='loan_perplannum_skew_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_kurt_before', startid=1, name='loan_perplannum_kurt_before', expand=False,drop=False),

        #剩余期数
        # NOTE:新增的planum_avg max,min提升成绩

        StatFeature(prefix='loan_plannum_sum_before', startid=1, name='loan_plannum_sum_before', expand=False,drop=False),
        StatFeature(prefix='loan_plannum_avg_before',startid=1,name='loan_plannum_avg_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_max_before', startid=1, name='loan_plannum_max_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_min_before', startid=1, name='loan_plannum_min_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_mad_before', startid=1, name='loan_plannum_mad_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_skew_before', startid=1, name='loan_plannum_skew_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_kurt_before', startid=1, name='loan_plannum_kurt_before', expand=False, drop=False),


        StatFeature(prefix='loan_balance_plannum', startid=1, name='loan_balance_plannum', expand=False),
        #总的贷款次数
        StatFeature(prefix='loan_count_before', startid=1, name='loan_count_before', expand=False),
        StatFeature(prefix='loan_count_previous', startid=1, name='loan_count_previous', expand=False),



        #点击某商品对应的平均贷款额度
        # 特征重要度低
        StatFeature(prefix='click_avg_loan', startid=1,  expand=True,  name='click_avg_loan',idfile='click_pid.id', drop=True) ,
        StatFeature(prefix='click_param_avg_loan', startid=1,  expand=True, name='click_param_avg_loan', idfile='click_pid_param.id' ,drop=True),

    ])

def initSuperFeatureList():
    """
    TODO

    :return:
    """
    superstat_list.extend([])

def loadData():
    pass

def alignFeatureID():

    start = 1
    id = []
    name = []
    for fea in  feature_list: #type:Feature
        prefix_set.append(fea.prefix)
        start = fea.alignFeatureID(start )
        
        if fea.drop == False:
            id.extend([ i for i in range(fea.start_feaid, fea.end_feaid)])
            name.extend([ fea.name() for i in range(fea.end_feaid - fea.start_feaid)])

        logging.info(fea.coverRange())
    for fea in stat_list:
        prefix_set.append(fea.prefix)
        start = fea.alignFeatureID(start)
        if fea.drop == False:
            id.extend([ i for i in range(fea.start_feaid, fea.end_feaid)])
            name.extend([ fea.name() for i in range(fea.end_feaid - fea.start_feaid)])
        logging.info(fea.coverRange())

    for fea in superstat_list:
        prefix_set.append(fea.prefix)
        start = fea.alignFeatureID(start)
        if fea.drop == False:
            id.extend([ i for i in range(fea.start_feaid, fea.end_feaid)])
            name.extend([ fea.name() for i in range(fea.end_feaid - fea.start_feaid)])
        logging.info(fea.coverRange())
    pd.DataFrame({'id': id, 'name': name}).to_csv('idname.txt',index=False)

    #sys.exit(1)

    pass
##===========================================================
def addUser():
    logging.info('addUser ing...')
    user = pd.read_csv(t_user,header=0)
    for age in user['age'].unique():
        feastr = 'age:{0}'.format(age)

        for fea in feature_list:
            o = fea #type: Feature
            o.tryAdd('age',feastr,sep=':')

    for sex in user['sex'].unique():
        feastr = 'sex:{0}'.format(sex)
        for fea in feature_list:
            o = fea #type: Feature
            o.tryAdd('sex',feastr,sep=':')

    for sex in user['sex'].unique():
        for age in user['age'].unique():
            feastr = 'age_sex:{0}_{1}'.format(age, sex)

            for fea in feature_list:
                o = fea  # type: Feature
                o.tryAdd('age_sex', feastr, sep=':')



    for active_date in user['active_date'].unique():
        feastr = 'active_date:{0}'.format(active_date)
        for fea in feature_list:
            o = fea  # type: Feature
            o.tryAdd('active_date', feastr, sep=':')

    for limit in user['limit'].unique():
        feastr = 'limit:{0}'.format(limit)
        for fea in feature_list:
            o = fea  # type: Feature
            o.tryAdd('limit', feastr, sep=':')

    for fea in feature_list:
        feastr = 'days_to_now'
        fea.tryAdd('days_to_now', feastr)

        fea.tryAdd('uid', 'uid')

    pass

def addUserStat():
    """
    user 相关的统计特征
    数据格式:

    month:col:colval  stat

    pivot_table默认是取均值
    这里获取的都是 上个月的平均贷款额度 ，因为没有通过时间过滤出 loan_train
    :return:
    """
    logging.info('addUserStat ing....')
    user = pd.read_csv(t_user, header=0)
    loan = pd.read_csv(t_loan, header=0)

    loan['loan_time_month'] = loan['loan_time'].apply(lambda x: x[5:7])
    loan['loan_time_hour'] = loan['loan_time'].apply(lambda x: x[11:13])
    loan['loan_time_day'] = loan['loan_time'].apply(lambda x: x[:10])

    user_loan = pd.merge(user,loan , on='uid')
    ul = pd.pivot_table(user_loan, index=['sex'],   values=['loan_amount'], columns=['loan_time_month'], aggfunc=np.mean).reset_index()

    buf = [] # (feastr,val)
    for month in ul['loan_amount'].columns.values:
        for i, v in enumerate(ul['loan_amount'][month]):

            next_month = str((int(month) + 1)).zfill(2)   # 方便后续join


            buf.append( ("sex" ,  "{0}:{1}:{2}".format(next_month, "sex", ul['sex'][i] ) , v ) )


    ul = pd.pivot_table(user_loan, index=['age'],   values=['loan_amount'], columns=['loan_time_month']).reset_index()
    for month in ul['loan_amount'].columns.values:
        for i, v in enumerate(ul['loan_amount'][month]):
            next_month = str((int(month) + 1)).zfill(2)  # 方便后续join
            buf.append(("age", "{0}:{1}:{2}".format(next_month, "age", ul['age'][i]), v))




    ul = pd.pivot_table(user_loan, index=['limit'], values=['loan_amount'], columns=['loan_time_month']).reset_index()
    for month in ul['loan_amount'].columns.values:
        for i, v in enumerate(ul['loan_amount'][month]):
            next_month = str((int(month) + 1)).zfill(2)  # 方便后续join
            buf.append(("limit", "{0}:{1}:{2}".format(next_month, "limit", ul['limit'][i]), v))

    ul = pd.pivot_table(user_loan, index=['active_date'], values=['loan_amount'], columns=['loan_time_month']).reset_index()
    for month in ul['loan_amount'].columns.values:
        for i, v in enumerate(ul['loan_amount'][month]):
            next_month = str((int(month) + 1)).zfill(2)  # 方便后续join
            buf.append(("active_date", "{0}:{1}:{2}".format(next_month, "active_date", ul['active_date'][i]), v))


    for prefix, feastr, val in buf:
        for stat_feature in stat_list:
            o = stat_feature #type:StatFeature
            o.tryAdd(prefix, feastr,val , sep=':')


def addOrderStat():
    """
    训练集: 购买品类 在训练集中的平均 贷款金额


    购买品类的平均贷款额度

    必须放在
    :return:
    """
    order = pd.read_csv(t_order, header=0)
    loan = pd.read_csv(t_loan, header=0)


    loan['month'] = loan['loan_time'].apply(lambda x: x[5:7])
    loan['raw_loan_amount'] = loan['loan_amount']  # skew and kurt 用
    loan['loan_amount'] = 5 ** loan['loan_amount'] - 1


    loan_train = loan[loan['month'].isin(train_month)]

    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order_train = order[order['month'].isin(train_month)]

    uid_order = order_train[['uid','cate_id','qty']].groupby(['uid', 'cate_id']).sum().reset_index()


    uid_loan= loan_train[['uid','loan_amount']].groupby(['uid']).sum().reset_index()



    uid_order_loan = pd.merge(uid_order, uid_loan, on=['uid'])

    cate_loan_mean = uid_order_loan[['cate_id', 'loan_amount']].groupby(
        ['cate_id']).mean().reset_index()

    magicRecover(cate_loan_mean, 'loan_amount')

    r, c = cate_loan_mean.shape
    """
      month  cate_id  loan_amount
0    10        1    12.770418
1    10        2    12.939553
2    10        3    12.622335
3    10        5    14.845864
4    10        6    12.718818

    """

    for i in range(r):
        row = cate_loan_mean.iloc[i, :]

        cate_id , loan_amount = row[0],row[1]



        feastr = '{0}:buy_cate_avg_loan:{1}'.format( train_next , int( cate_id) )

        prefix = 'buy_cate_avg_loan'


        for stat_feature in stat_list:
            o = stat_feature  # type:StatFeature
            o.tryAdd(prefix, feastr, loan_amount, sep=':')



def tryAdd(prefix, feastr, fealist):
    """
    WARN： 只能处理 Feature  类型 ；
    StatFeature不适用
    :param prefix:
    :param feastr:
    :param fealist:
    :return:
    """
    for fea in fealist:
        o = fea  # type: Feature
        o.tryAdd(prefix, feastr, sep=':')

def addOrder():
    logging.info('addOrder ing..')
    order = pd.read_csv(t_order, header=0)

    #购买的总金额
    #购买品类的 平均贷款额度
    for cate in order['cate_id'].unique():
        feastr = 'buy_cate:{0}'.format(cate)
        tryAdd('buy_cate', feastr, feature_list)
        
        tryAdd('buy_cate_count', 'buy_cate_count:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_min', 'buy_cate_min:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_max', 'buy_cate_max:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_std', 'buy_cate_std:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_mad', 'buy_cate_mad:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_mean', 'buy_cate_mean:{0}'.format(cate), feature_list)
        tryAdd('buy_cate_skew', 'buy_cate_skew:{0}'.format(cate), feature_list)
        feastr = 'buy_cate_discount:{0}'.format(cate)
        print feastr
        tryAdd('buy_cate_discount', feastr, feature_list)

    tryAdd('buy_large1_count', 'buy_large1_count', feature_list)
    tryAdd('buy_large2_count', 'buy_large2_count', feature_list)
    tryAdd('buy_large3_count', 'buy_large3_count', feature_list)
    tryAdd('buy_large4_count', 'buy_large4_count', feature_list)
    tryAdd('buy_sum', 'buy_sum', feature_list)
    tryAdd('buy_count', 'buy_count', feature_list)
    tryAdd('buy_discount_sum', 'buy_discount_sum', feature_list)

    tryAdd('buy_mean', 'buy_mean',feature_list)
    tryAdd('buy_min', 'buy_min',feature_list)
    tryAdd('buy_max', 'buy_max',feature_list)
    tryAdd('buy_mad', 'buy_mad',feature_list)
    tryAdd('buy_skew', 'buy_skew',feature_list)


    for i in range(1,6):
        tryAdd('svd_order_param{0}'.format(i) ,
               'svd_order_param{0}'.format(i),
               feature_list)

        # feastr = 'buy_cate_loan:{0}'.format(cate)
        # tryAdd('buy_cate_loan', feastr, feature_list)
    pass
def addClick():
    click = pd.read_csv(t_click,header = 0)


    for pid in click['pid'].unique():
        feastr = 'click:{0}'.format(int(pid))
        tryAdd('click', feastr,feature_list)

    pair =  click[['pid', 'param']].groupby(['pid', 'param']).count().index.values

    for pid,param in pair:
        feastr = 'click_param:{0}_{1}'.format(int(pid),int(param))
        tryAdd('click_param', feastr, feature_list)

    # svd_click_params.
    for i in range(1,6):
        tryAdd('svd_click_param{0}'.format(i),
               'svd_click_param{0}'.format(i),
               feature_list)
    pass

def addClickStat():
    cover_prefix = ['click_avg_loan','click_param_avg_loan']
    if not hitPrefix(cover_prefix, stat_list): return 

    click = pd.read_csv(t_click, header=0)
    loan = pd.read_csv(t_loan,header=0)
    loan['month'] = loan['loan_time'].apply(lambda x: x[5:7])
    click['month'] = click['click_time'].apply(lambda x: x[5:7])


    click_train = click[click['month'].isin(train_month)]
    loan_train = loan[loan['month'].isin(train_month)]

    click_train_cnt = click_train.groupby(['uid', 'pid']).count().reset_index()
    click_train_param_cnt = click_train.groupby(['uid', 'pid','param']).count().reset_index()

    loan_train_sum = loan_train[['uid', 'loan_amount']].groupby(['uid']).sum().reset_index()

    df = pd.merge(loan_train_sum, click_train_cnt, on='uid')
    pid_loan = df[['pid','loan_amount']].groupby('pid').mean().reset_index()


    param_df = pd.merge(loan_train_sum, click_train_param_cnt, on='uid')
    pid_param_loan = param_df[['pid', 'param' , 'loan_amount']].groupby(['pid','param']).mean().reset_index()


    r,c  =  pid_loan.shape
    for i in range(r):
        row = pid_loan.iloc[i,:]
        pid = row[0]
        loan_amount = row[1]
        feastr = '{0}:click_avg_loan:{1}'.format(train_next, int(pid))
        for fea in stat_list:
            fea.tryAdd('click_avg_loan', feastr, loan_amount)

    r, c = pid_param_loan.shape
    for i in range(r):
        row = pid_param_loan.iloc[i, :]
        pid = row[0]
        param = row[1]
        loan_amount = row[2]
        feastr = '{0}:click_param_avg_loan:{1}_{2}'.format(train_next, int(pid),int(param))
        for fea in stat_list:
            fea.tryAdd('click_param_avg_loan', feastr, loan_amount)







def addLoanSumPrevious(df, month):
    r, c = df.shape
    dvals = df.values
    for i in range(r):
        row = dvals[i]

        uid = int(row[0])
        amount = row[1]

        for fea in stat_list:
            o = fea  # type:StatFeature
            prefix = 'loan_sum_previous'
            feastr = '{month}:loan_sum_previous:{uid}'.format(month= month,  uid=uid)
            val = amount
            o.tryAdd(prefix, feastr, val)

def addLoanMaxBefore(df,month):
    r, c = df.shape
    dvals = df.values
    for i in range(r):
        row = dvals[i]

        uid = int(row[0])
        amount = row[1]

        for fea in stat_list:
            o = fea  # type:StatFeature
            prefix = 'loan_max_before'
            feastr = '{month}:loan_max_before:{uid}'.format(month=month, uid=uid)
            val = amount
            o.tryAdd(prefix, feastr, val)


def addLoanAvgBefore(df,month):
    r, c = df.shape
    dvals = df.values
    for i in range(r):
        row = dvals[i]

        uid = int(row[0])
        amount = row[1]

        for fea in stat_list:
            o = fea  # type:StatFeature
            prefix = 'loan_avg_before'
            feastr = '{month}:loan_avg_before:{uid}'.format(month=month, uid=uid)
            val = amount
            o.tryAdd(prefix, feastr, val)

def addLoanSumBefore(df, month):
    r, c = df.shape
    dvals = df.values
    for i in range(r):
        row = dvals[i]

        uid = int(row[0])
        amount = row[1]

        for fea in stat_list:
            o = fea  # type:StatFeature
            prefix = 'loan_sum_before'
            feastr = '{month}:loan_sum_before:{uid}'.format(month= month,  uid=uid)
            val = amount
            o.tryAdd(prefix, feastr, val)
def addLoanCountBefore(df,month):
    r, c = df.shape
    for i in range(r):
        row = df.iloc[i, :]
        uid = int(row[0])
        count = row[1]

        prefix = 'loan_count_before'
        feastr = '{month}:loan_count_before:{uid}'.format(month=month, uid=uid)
        val = count

        for fea in stat_list:
            o = fea  # type:StatFeature
            o.tryAdd(prefix, feastr, val)

#def addLoanWeightSumBefore(df,month):
#    r, c = df.shape
#    for i in range(r):
#        row = df.iloc[i, :]
#        uid = int(row[0])
#        count = row[1]
#
#        prefix = 'loan_weight_sum_before'
#        feastr = '{month}:loan_weight_sum_before:{uid}'.format(month=month, uid=uid)
#        val = count
#
#        for fea in stat_list:
#            o = fea  # type:StatFeature
#            o.tryAdd(prefix, feastr, val)
def addLoanStatBefore(df, month, prefix):
    r, c = df.shape
    dvals = df.values
    for i in range(r):
        row = dvals[i]
        uid = int(row[0])
        amount = row[1]

        feastr = '{month}:{prefix}:{uid}'.format(month=month,prefix=prefix, uid=uid)

        for fea in stat_list:
            o = fea  # type:StatFeature
            val = amount
            o.tryAdd(prefix, feastr, val)

def addLoanBalancePlannumBefore(df, month):
    r, c = df.shape
    for i in range(r):
        row = df.iloc[i, :]
        uid = int(row[0])
        left_plannum = row[1]

        prefix = 'loan_balance_plannum'
        feastr = '{month}:loan_balance_plannum:{uid}'.format(month=month, uid=uid)
        val = int(left_plannum)

        for fea in stat_list:
            o = fea  # type:StatFeature
            o.tryAdd(prefix, feastr, val)
def addLoanBalance(df, month):
    r, c = df.shape
    for i in range(r):
        row = df.iloc[i, :]
        uid = int(row[0])
        loan_balance = row[1]

        prefix = 'loan_balance'
        feastr = '{month}:loan_balance:{uid}'.format(month=month, uid=uid)
        val = int(loan_balance)

        for fea in stat_list:
            o = fea  # type:StatFeature
            o.tryAdd(prefix, feastr, val)
# def addPlanumStatTool(loan, time_range,month):

def addPreviousLoanStatTool(loan_train_prev, train_next):
    gdf = loan_train_prev[['uid', 'loan_amount']].groupby(['uid'])


    loan_train_prev_sum = gdf.sum().reset_index()
    magicRecover(loan_train_prev_sum, 'loan_amount')
    addLoanStatBefore(loan_train_prev_sum, train_next, 'loan_sum_previous')


    loan_train_prev_mean = gdf.mean().reset_index()
    magicRecover(loan_train_prev_mean, 'loan_amount')
    addLoanStatBefore(loan_train_prev_mean, train_next, 'loan_avg_previous')

    loan_mad = gdf.mad().reset_index()
    magicRecover(loan_mad, 'loan_amount')
    addLoanStatBefore(loan_mad, train_next, 'loan_mad_previous')


    loan_skew = gdf.skew().fillna(0).reset_index()
    loan_skew.columns = ['uid','skew']
    addLoanStatBefore(loan_skew,train_next,'loan_skew_previous')

    loan_kurt = gdf.apply(lambda  x : x['loan_amount'].kurt()).fillna(0).reset_index()
    loan_kurt.columns = ['uid', 'kurt']
    addLoanStatBefore(loan_kurt, train_next, 'loan_kurt_previous')



    groupdf = loan_train_prev[['uid','raw_loan_amount']].groupby('uid')

    loan_min = groupdf.min().reset_index()
    addLoanStatBefore(loan_min, train_next, 'loan_min_previous')

    loan_max = groupdf.max().reset_index()
    addLoanStatBefore(loan_max, train_next, 'loan_max_previous')

    loan_median = groupdf.median().reset_index()
    addLoanStatBefore(loan_median, train_next, 'loan_median_previous')



    loan_train_prev_cnt = loan_train_prev[['uid', 'loan_time']].groupby(['uid']).count().reset_index()
    addLoanStatBefore(loan_train_prev_cnt, train_next, 'loan_count_previous')

    loan_train_prev_sumlog = loan_train_prev[['uid', 'raw_loan_amount']].groupby(['uid']).sum().reset_index()
    addLoanStatBefore(loan_train_prev_sumlog, train_next, 'loan_sumlog_previous')


def lookahead_plannum_cnt(loan_train,ahead=0):
    """
    repay_plannum: 下个月要还多少笔贷款
    repay_clear_count: 下个月还以后，共结清多少贷款
    repay_marginal_clear_count:  下个月当月 能结清多少贷款
    :param loan_train:
    :param ahead:
    :return:
    """

    loan_train['repay_plannum'] = loan_train['left_plannum'].apply(lambda x: 0 if x - ahead < 0  else 1)
    loan_train['repay_clear_count'] = loan_train['left_plannum'].apply(lambda x: 1 if x -ahead <= 0 else 0)

    loan_train['repay_clear_amount'] = loan_train['repay_clear_count'] * loan_train['loan_amount']
    loan_train['repay_marginal_clear_count'] = loan_train['left_plannum'].apply(lambda x: 1 if x -ahead == 0  else 0)
    loan_train['repay_marginal_clear_amount'] = loan_train['repay_marginal_clear_count'] * loan_train['loan_amount']

    repay_plannum = loan_train[['uid','repay_plannum']].groupby('uid').sum().reset_index()
    repay_clear_count = loan_train[['uid','repay_clear_count']].groupby('uid').sum().reset_index()
    repay_clear_amount= loan_train[['uid','repay_clear_amount']].groupby('uid').sum().reset_index()
    repay_marginal_clear_count = loan_train[['uid','repay_marginal_clear_count']].groupby('uid').sum().reset_index()
    repay_marginal_clear_amount = loan_train[['uid','repay_marginal_clear_amount']].groupby('uid').sum().reset_index()

    return repay_plannum , repay_clear_count, repay_clear_amount, repay_marginal_clear_count ,repay_marginal_clear_amount
def addLimitStat(limit_use_rate, month, ptag = ''):
    addLoanStatBefore(limit_use_rate[['uid', 'use_rate']], month, ptag + 'limit_use_rate')

    magicRecover(limit_use_rate, 'limit_free')
    addLoanStatBefore(limit_use_rate[['uid', 'limit_free']], month, ptag + 'limit_free')

    magicRecover(limit_use_rate, 'limit_in_use')
    addLoanStatBefore(limit_use_rate[['uid', 'limit_in_use']], month, ptag + 'limit_in_use')
def loan_day_plannum(user,loan_train, month):
    for i in [1,3,6,12]:
        logging.info('loan_day_plannum {0}'.format(i))
        loan_target = loan_train[loan_train['plannum'] == i]

        loan_day = pd.merge(user, loan_target, on='uid')
        loan_day['loan_day_diff'] = pd.to_datetime(loan_day['loan_time']).subtract(pd.to_datetime(loan_day['active_date'])).astype('timedelta64[D]')
        loan_day = loan_day[['uid','loan_day_diff']].groupby('uid')
        loan_day_min  = loan_day.min().reset_index()
        loan_day_max = loan_day.max().reset_index()
        loan_day_median = loan_day.median().reset_index()
        loan_day_mad = loan_day.mad().reset_index()
        addLoanStatBefore(loan_day_min, month,'loan_day_min{0}'.format(i))
        addLoanStatBefore(loan_day_max,month,'loan_day_max{0}'.format(i))
        addLoanStatBefore(loan_day_median,month,'loan_day_median{0}'.format(i))
        addLoanStatBefore(loan_day_mad,month,'loan_day_mad{0}'.format(i))
    
def loan_sum_rank(loan_train, month,prefix):
    loan_rank = loan_train[['uid','loan_amount']].groupby('uid').sum().rank().reset_index()

    loan_rank.columns = ['uid','rank']

    addLoanStatBefore(loan_rank, month, prefix)

def lookahead(loan_train, user, ahead=0):
    """
    非常重要的特征
    :param loan_train:
    :param user:
    :param ahead:
    :return:
    """

    loan_train['repay_plannum'] = loan_train['left_plannum'].apply(lambda x: 0 if x < ahead  else 1)
    loan_train['repay_pressure'] = loan_train['loan_amount'] / loan_train['plannum'] * loan_train['repay_plannum']

    repay_pressure = loan_train[['uid', 'repay_pressure']].groupby('uid').sum().reset_index()


    loan_train['cur_left_plannum'] = loan_train['left_plannum'].apply(lambda x: 0 if x - ahead < 0   else x - ahead +1  )
    loan_train['left_balance'] = loan_train['loan_amount'] / loan_train['plannum'] * loan_train['cur_left_plannum']
    loan_train_balance = loan_train[['uid','left_balance']].groupby('uid').sum().reset_index()


    loan_left_limit = pd.merge(user, loan_train_balance, on='uid')
    loan_left_limit['left_limit'] = loan_left_limit['limit'] - loan_left_limit['left_balance']
    loan_left_limit['left_limit'] = loan_left_limit['left_limit'].apply(lambda x: 0 if x < 0 else x)


    loan_train['limit_in_use'] = loan_train['loan_amount'] * loan_train['repay_plannum']
    limit_in_use = loan_train[['uid', 'limit_in_use']].groupby('uid').sum().reset_index()

    limit_use_rate = pd.merge(user, limit_in_use, on='uid')
    limit_use_rate['use_rate'] = limit_use_rate['limit_in_use'] / limit_use_rate['limit']
    limit_use_rate['limit_free'] = limit_use_rate['limit'] - limit_use_rate['limit_in_use']
    limit_use_rate['limit_free'] = limit_use_rate['limit_free'].apply(lambda x: 0 if x < 0 else x)


    return repay_pressure, loan_train_balance,loan_left_limit, limit_use_rate
def lookback_stack_day(df, date_pivot, month):
    logging.info('lookback_stack_day ....')
    df['close_time'] = pd.to_datetime(df['loan_time']) + df['plannum'] * pd.DateOffset(days=30) 
    def stack_tool(date_pivot, start):
        logging.info('stack_tool date_pivot = {0}'.format(date_pivot))
        df['loan_close_day_diff'] = pd.to_datetime(df['close_time']).apply(lambda  x : date_pivot - x ).astype('timedelta64[D]')
        df['loan_time_day_diff'] = pd.to_datetime(df['loan_time']).apply(lambda  x : date_pivot - x ).astype('timedelta64[D]')
    #for back in range(1,20):
        for i, back in enumerate([2, 15, 21 , 30 ,45]):
            lookback_df = df[ (df['loan_close_day_diff'] <= back) & (df['loan_time_day_diff'] >=0) ]
            lookback_df['loan_day_amount'] = lookback_df['loan_amount']/lookback_df['plannum']/30
            lookback_df_day_sum = lookback_df[['uid','loan_day_amount']].groupby('uid').sum().reset_index()
            magicRecover(lookback_df_day_sum ,'loan_day_amount')
            addLoanStatBefore(lookback_df_day_sum, month,'lookback_stack_sum{0}'.format(start +i))
        return start +  i +1 
    start = stack_tool(date_pivot, 1) 
    date_pivot  = date_pivot + pd.DateOffset(days=30)
    start = stack_tool(date_pivot, start)
    logging.info('lookback_stack_day DONE') 

def lookback_day(user , df, date_pivot, month):
    """
    [date_pivot - back, date_pivot)  loan_sum 
    """
    
    df['loan_past_day_diff'] = pd.to_datetime(df['loan_time']).apply(lambda  x : date_pivot - x ).astype('timedelta64[D]')
    for back in [7,21,45,57]:
        lookback = df[df['loan_past_day_diff'] <= back][['uid','loan_amount']].groupby('uid').sum().reset_index()
        if lookback.shape[0] ==0: continue 

        mdf = pd.merge(user[['uid']], lookback, on='uid', how='left')[['uid','loan_amount']]
        lookback = mdf.fillna(0)
        magicRecover(lookback, 'loan_amount')
        prefix = 'lookback{i}_loan_sum'.format(i=back)
        addLoanStatBefore(lookback, month, prefix)
def lookahead_day(user, df, date_pivot, month):
    logging.info('lookahead_day ing..')
    """
    [date_pivot , date_pivot + ahead) repay_sum 
    """
    df['loan_ahead_day_diff'] = pd.to_datetime(df['loan_time']).apply(lambda  x : date_pivot - x).astype('timedelta64[D]')
    
    for ahead in [7, 21, 45, 57]:
        df['loan_ahead_plannum'] = ((df['loan_ahead_day_diff'] + ahead)/30.).astype('int')  - (df['loan_ahead_day_diff']/30.).astype('int')
        repay_df = df[( df['loan_ahead_plannum'] >= 1.) & (df['loan_ahead_day_diff']/30. < df['plannum'] )]
        repay_df['rl_left'] = repay_df['plannum'] - (df['loan_ahead_day_diff']/30.).astype('int')
        repay_df['real_ahead_plannum'] = repay_df[['rl_left','loan_ahead_plannum']].min(axis=1)
        repay_df['real_ahead_amount'] = repay_df['real_ahead_plannum']* repay_df['loan_amount']/repay_df['plannum']
        repay_ahead_df = repay_df[['uid','real_ahead_amount']].groupby('uid').sum().reset_index()
        if repay_ahead_df.shape[0] == 0 : continue 
        mdf = pd.merge(user[['uid']], repay_ahead_df, on='uid', how='left')
        repay_ahead_df = mdf.fillna(0)
        magicRecover(repay_ahead_df, 'real_ahead_amount')
        prefix = 'lookahead{0}_repay'.format(ahead)
        addLoanStatBefore(repay_ahead_df, month, prefix)
    logging.info('lookahead_day DONE')
def addLoanStatTool(loan,  time_range , month):

    loan_train = loan[loan['month'].isin(time_range)]
    date_pivot = pd.to_datetime(train_next_day)
    user = pd.read_csv(t_user,header=0)
    loan_day_plannum(user, loan_train,month)

    lookback_stack_day(loan_train, date_pivot, month)
    # for i in range(1,60):
    lookback_day(user, loan_train, date_pivot, month)
    # for i in range(1,60):
    lookahead_day(user,loan_train, date_pivot, month)
    ###################
    # rank feature to stablize efficience.
    loan_sum_rank(loan_train,month, 'loan_sum_rank')
    #####################

    loan_train_sumlog = loan_train[['uid','raw_loan_amount']].groupby('uid').sum().reset_index()
    addLoanStatBefore(loan_train_sumlog, month, 'loan_sumlog_before')

    loan_consecutive = loan_train[['uid','consecutive_loan']].groupby('uid').max().reset_index()

    addLoanStatBefore(loan_consecutive, month, 'loan_consecutive_sum_before')
    #loan_train_avg = loan_train[['uid','loan_amount']].groupby(['uid']).sum()/len(time_range)
    #loan_train_avg = loan_train_avg.reset_index()
    groupdf = loan_train[['uid','loan_amount']].groupby(['uid'])
    loan_train_sum = groupdf.sum().reset_index()
    loan_train_avg = groupdf.mean().reset_index()
    loan_train_median = groupdf.median().reset_index()
    loan_train_max = groupdf.max().reset_index()
    loan_train_min = groupdf.min().reset_index()
    loan_train_avg_std = (groupdf.mean() + groupdf.std().fillna(0)).reset_index()
    loan_train_avg_half_std = (groupdf.mean() + 0.5 * groupdf.std().fillna(0)).reset_index()
    loan_train_avg_double_std = (groupdf.mean() + 2 * groupdf.std().fillna(0)).reset_index()
    loan_train_mad = groupdf.mad().fillna(0).reset_index()

    loan_train_skew = groupdf.skew().fillna(0).reset_index()
    loan_train_skew.columns = ['uid', 'skew']

    loan_train_kurt = groupdf.apply(lambda x: x['loan_amount'].kurt()).fillna(0).reset_index()
    loan_train_kurt.columns = ['uid', 'kurt']
    # ==============================
    rawgroupdf = loan_train[['uid', 'raw_loan_amount']].groupby(['uid'])
    loan_train_skewlog = rawgroupdf.skew().fillna(0).reset_index()
    loan_train_skewlog.columns = ['uid','skew']

    loan_train_kurtlog = rawgroupdf.apply(lambda x: x['raw_loan_amount'].kurt()).fillna(0).reset_index()
    loan_train_kurtlog.columns = ['uid','kurt']

    loan_train_madlog = rawgroupdf.mad().fillna(0).reset_index()
    loan_train_madlog.columns = ['uid','mad']
    addLoanStatBefore(loan_train_madlog, month, "loan_madlog_before")


    loan_train_avglog = rawgroupdf.mean().reset_index()
    addLoanStatBefore(loan_train_avglog, month, "loan_avglog_before")
    #============================plannum check
    loan_train['plannum1_cnt'] = loan_train['plannum'].apply(lambda x: 1 if x ==1 else 0)
    loan_train['plannum1_amount'] = loan_train['plannum1_cnt'] * loan_train['loan_amount']

    loan_train['plannum3_cnt'] = loan_train['plannum'].apply(lambda x: 1 if x ==3 else 0)
    loan_train['plannum3_amount'] = loan_train['plannum3_cnt'] * loan_train['loan_amount']

    loan_train['plannum6_cnt'] = loan_train['plannum'].apply(lambda x: 1 if x ==6 else 0)
    loan_train['plannum6_amount'] = loan_train['plannum6_cnt'] * loan_train['loan_amount']

    loan_train['plannum12_cnt'] = loan_train['plannum'].apply(lambda x: 1 if x ==12 else 0)
    loan_train['plannum12_amount'] = loan_train['plannum12_cnt'] * loan_train['loan_amount']

    loan_train_p1_cnt = loan_train[['uid','plannum1_cnt']].groupby('uid').sum().reset_index()
    loan_train_p3_cnt = loan_train[['uid','plannum3_cnt']].groupby('uid').sum().reset_index()
    loan_train_p6_cnt = loan_train[['uid','plannum6_cnt']].groupby('uid').sum().reset_index()
    loan_train_p12_cnt = loan_train[['uid','plannum12_cnt']].groupby('uid').sum().reset_index()
    addLoanStatBefore(loan_train_p1_cnt, month, "loan_plannum1_count")
    addLoanStatBefore(loan_train_p3_cnt, month, "loan_plannum3_count")
    addLoanStatBefore(loan_train_p6_cnt, month, "loan_plannum6_count")
    addLoanStatBefore(loan_train_p12_cnt, month, "loan_plannum12_count")
    
    loan_train_p1_sum = loan_train[['uid','plannum1_amount']].groupby('uid').sum().reset_index()
    loan_train_p3_sum = loan_train[['uid','plannum3_amount']].groupby('uid').sum().reset_index()
    loan_train_p6_sum = loan_train[['uid','plannum6_amount']].groupby('uid').sum().reset_index()
    loan_train_p12_sum = loan_train[['uid','plannum12_amount']].groupby('uid').sum().reset_index()
    addLoanStatBefore(loan_train_p1_sum, month, 'loan_plannum1_sum')
    addLoanStatBefore(loan_train_p3_sum, month, 'loan_plannum3_sum')
    addLoanStatBefore(loan_train_p6_sum, month, 'loan_plannum6_sum')
    addLoanStatBefore(loan_train_p12_sum, month, 'loan_plannum12_sum')
    #==============================
    loan_train_count = loan_train[['uid', 'loan_time']].groupby('uid').count().reset_index()

    loan_train['delta_month'] = int(train_next) - loan_train['month'].astype(int)
    loan_train['left_plannum'] = loan_train['plannum'] - loan_train['delta_month']

    repay_plannum, repay_clear_count , repay_clear_amount, repay_marginal_clear_count, repay_marginal_clear_amount= lookahead_plannum_cnt(loan_train,ahead=0)

    addLoanStatBefore(repay_plannum, month, 'repay_plannum')

    addLoanStatBefore(repay_clear_count, month, 'repay_clear_count')

    magicRecover(repay_clear_amount, 'repay_clear_amount') 
    addLoanStatBefore(repay_clear_amount, month, 'repay_clear_amount')


    addLoanStatBefore(repay_marginal_clear_count, month, 'repay_marginal_clear_count')

    magicRecover(repay_marginal_clear_amount, 'repay_marginal_clear_amount')
    addLoanStatBefore(repay_marginal_clear_amount, month, 'repay_marginal_clear_amount')

    lookahead_repay_plannum, lookahead_repay_clear_count , lookahead_repay_clear_amount, lookahead_repay_marginal_clear_count, lookahead_repay_marginal_clear_amount= lookahead_plannum_cnt(loan_train,ahead=1)

    addLoanStatBefore(lookahead_repay_plannum, month, 'lookahead_repay_plannum')

    addLoanStatBefore(lookahead_repay_clear_count, month, 'lookahead_repay_clear_count')

    magicRecover(lookahead_repay_clear_amount, 'repay_clear_amount') 
    addLoanStatBefore(lookahead_repay_clear_amount, month, 'lookahead_repay_clear_amount')


    addLoanStatBefore(lookahead_repay_marginal_clear_count, month, 'lookahead_repay_marginal_clear_count')

    magicRecover(lookahead_repay_marginal_clear_amount, 'repay_marginal_clear_amount')
    addLoanStatBefore(lookahead_repay_marginal_clear_amount, month, 'lookahead_repay_marginal_clear_amount')


    

    #loan_train['limit_in_use'] = loan_train['loan_amount'] * loan_train['repay_plannum']
    #limit_in_use = loan_train[['uid','limit_in_use']].groupby('uid').sum().reset_index() 
    #magicRecover(limit_in_use, 'limit_in_use')
    #addLoanStatBefore(limit_in_use, month, 'limit_in_use')
    #下个与还款金额

    loan_train['cur_left_plannum'] = loan_train['left_plannum'].apply(lambda x: 0 if x < 0  else x +1 )
    loan_train_balance_plannum = loan_train[['uid','cur_left_plannum']].groupby('uid').sum().reset_index()



      # 到train_next的距离.
    def loan_past_day_feature(date_pivot, df, pretag = ''):

        df['loan_past_day_diff'] = pd.to_datetime(df['loan_time']).apply(lambda  x : date_pivot - x ).astype('timedelta64[D]')
        past_day = df[['uid','loan_past_day_diff']].groupby('uid')
        past_day_min = past_day.min().reset_index()
        past_day_max = past_day.max().reset_index()
        past_day_mad = past_day.mad().reset_index()
        past_day_mean = past_day.mean().reset_index()
        past_day_median = past_day.median().reset_index()
        past_day_skew = past_day.skew().fillna(0).reset_index()

        addLoanStatBefore(past_day_min, month, pretag + 'loan_past_day_min')
        addLoanStatBefore(past_day_max, month, pretag + 'loan_past_day_max')
        addLoanStatBefore(past_day_mad, month, pretag+ 'loan_past_day_mad')
        addLoanStatBefore(past_day_mean, month, pretag + 'loan_past_day_mean')
        addLoanStatBefore(past_day_median, month, pretag + 'loan_past_day_median')
        addLoanStatBefore(past_day_skew, month, pretag + 'loan_past_day_skew')

    date_pivot = pd.to_datetime(train_next_day)
    loan_past_day_feature(date_pivot, loan_train)

    loan_clear_train = loan_train[loan_train['repay_clear_count'] == 1 ] #repay clear
    loan_past_day_feature(date_pivot, loan_clear_train, pretag='clear_')


    #========================================================================================================
    #                                     User Merge
    # ========================================================================================================
    #loan_left_limit 实验
    user = pd.read_csv(t_user, header=0)
    user['limit'] = 5 ** user['limit'] - 1
    loan_day = pd.merge(user, loan_train, on='uid')

    loan_day['loan_day_diff'] = pd.to_datetime(loan_day['loan_time']).subtract(pd.to_datetime(loan_day['active_date'])).astype('timedelta64[D]')


    loan_day = loan_day[['uid','loan_day_diff']].groupby('uid') #type:pd.DataFrame

    loan_day_min  = loan_day.min().reset_index()
    loan_day_max = loan_day.max().reset_index()
    loan_day_mean = loan_day.mean().reset_index()
    loan_day_median = loan_day.median().reset_index()
    loan_day_mad = loan_day.mad().reset_index()
    loan_day_skew = loan_day.skew().fillna(0).reset_index()

    addLoanStatBefore(loan_day_min,month,'loan_day_min')
    addLoanStatBefore(loan_day_max,month,'loan_day_max')
    addLoanStatBefore(loan_day_mean,month,'loan_day_mean')
    addLoanStatBefore(loan_day_median,month,'loan_day_median')
    addLoanStatBefore(loan_day_mad,month,'loan_day_mad')
    addLoanStatBefore(loan_day_skew,month,'loan_day_skew')



    
    repay_pressure, loan_train_balance, loan_left_limit, limit_use_rate = lookahead(loan_train,user, ahead=0)
    lookahead_repay_pressure, lookahead_loan_train_balance, lookahead_loan_left_limit, lookahead_limit_use_rate = lookahead(loan_train, user, ahead=1)
    lookahead2_repay_pressure, lookahead2_loan_train_balance, lookahead2_loan_left_limit, lookahead2_limit_use_rate = lookahead(loan_train, user,ahead=2)
    lookahead3_repay_pressure, lookahead3_loan_train_balance, lookahead3_loan_left_limit, lookahead3_limit_use_rate  = lookahead(loan_train, user, ahead=3)

    loan_consecutive_left_limit = pd.merge(loan_train_balance, loan_consecutive, on='uid')
    loan_consecutive_left_limit['left_limit'] = 5 ** loan_consecutive_left_limit['consecutive_loan'] - 1 - loan_consecutive_left_limit['left_balance']
    loan_consecutive_left_limit['left_limit'] = loan_consecutive_left_limit['left_limit'].apply(lambda x: 0 if x < 0 else x)
    magicRecover(loan_consecutive_left_limit, 'left_limit')
    addLoanStatBefore(loan_consecutive_left_limit, month, 'loan_consecutive_left_limit')

    repay_df = pd.merge(repay_pressure,  repay_plannum, on='uid')
    repay_df['repay_mean_amount'] = repay_df['repay_pressure'] / (repay_df['repay_plannum'] + 0.0001)

    repay_limit_rate =  pd.merge(user, repay_pressure,on='uid')
    repay_limit_rate['repay_limit_rate'] = repay_limit_rate['repay_pressure']/ repay_limit_rate['limit']


    addLoanStatBefore(repay_limit_rate, month , 'repay_limit_rate')

    ###################
    """
    t_user , 以 12-01为基准 ，看看 active_date到目前有多远，越远的，越可能有提额 .
    count    90993.000000
    mean         8.138780
    std        look  2.058783
    min          5.000000
    25%          6.000000
    50%          8.000000
    75%         10.000000
    max         12.000000

    """
    ##################
    # user['limit'] = user['limit'] * 1.5
    # user['limit'] = user['limit'].apply( lambda  x: 200000 if x > 200000 else x )
    #算一个比率
    loan_left_magic_limit = pd.merge(user,loan_train_balance, on='uid')
    loan_left_magic_limit['left_magic_limit'] = loan_left_magic_limit['limit'] - loan_left_magic_limit['left_balance']
    loan_left_magic_limit['left_magic_limit'] = loan_left_magic_limit['left_magic_limit']/loan_left_magic_limit['limit']

    loan_left_magic_limit['left_magic_limit'] = loan_left_magic_limit['left_magic_limit'].apply(lambda  x: 0 if x < 0 else x)

    # magicRecover(loan_left_magic_limit, 'left_magic_limit')
    addLoanStatBefore(loan_left_magic_limit[['uid', 'left_magic_limit']], month, 'loan_left_magic_limit')


    loan_max_rate = pd.merge(user,loan_train_max,on='uid')
    loan_max_rate['max_rate'] = loan_max_rate['loan_amount']/loan_max_rate['limit']
    addLoanStatBefore(loan_max_rate[['uid','max_rate']],month, 'loan_max_rate')

    #limit_use_rate = pd.merge(user,limit_in_use, on='uid') 
    #limit_use_rate['use_rate'] = limit_use_rate['limit_in_use']/limit_use_rate['limit']
    #limit_use_rate['limit_free'] = limit_use_rate['limit'] - limit_use_rate['limit_in_use']
    #limit_use_rate['limit_free'] = limit_use_rate['limit_free'].apply(lambda x: 0 if x <0 else x)
    addLimitStat(lookahead_limit_use_rate, month)  #模仿之前的代码.  直接采用  limit_use_rate 分数会下降.
    # addLimitStat(limit_use_rate,month )
    # addLimitStat(lookahead_limit_use_rate, month, ptag='lookahead_')
    # addLimitStat(lookahead2_limit_use_rate, month, ptag='lookahead2_')
    # addLimitStat(lookahead3_limit_use_rate, month, ptag='lookahead3_')

    loan_train['weight'] = 0
    imonth = int(month)

    wbuf = [4, 2, 1 ]
    for i in range(1,4):
        cur = str(imonth - i ).zfill(2)
        loan_train.loc[loan_train['month'] == cur, 'weight' ] = wbuf[i-1]

    loan_train['wsum'] = loan_train['weight'] * loan_train['loan_amount']

    loan_train_sum_weight = loan_train[['uid', 'wsum']].groupby(['uid']).sum().reset_index()

    magicRecover(loan_train_sum_weight,'wsum')
    addLoanStatBefore(loan_train_sum_weight, month, 'loan_weight_sum_before')
    #weight distribution 
    #wmatrix = np.random.normal(2,2,(10,3)) # to keep train and test consistent ,we need to store the weight before.
#    for j in range(len(wmatrix)):
#        loan_train['weight'] = 0 
#        wbuf = wmatrix[j]
#        logging.info('j = {0} , weight = {1} '.format(j, wbuf))
#        for i in range(1,4):
#            cur = str(imonth -i).zfill(2)
#            loan_train.loc[loan_train['month'] == cur,'weight'] = wbuf[i-1]
#        loan_train['wsum'] = loan_train['weight'] * loan_train['loan_amount']
#        loan_train_sum_weight = loan_train[['uid', 'wsum']].groupby(['uid']).sum().reset_index()
#        addLoanStatBefore(loan_train_sum_weight, month, 'loan_weight_sum_before_{0}'.format(j))
#
    magicRecover(loan_train_sum, 'loan_amount')
    addLoanSumBefore(loan_train_sum, month)

    magicRecover(loan_train_avg, 'loan_amount')
    addLoanAvgBefore(loan_train_avg, month)

    magicRecover(loan_train_median, 'loan_amount')
    addLoanStatBefore(loan_train_median,month,'loan_median_before')

    magicRecover(loan_train_min, 'loan_amount')
    addLoanStatBefore(loan_train_min,month,'loan_min_before')

    magicRecover(loan_train_max, 'loan_amount')
    addLoanMaxBefore(loan_train_max, month)

    magicRecover(loan_train_avg_half_std, 'loan_amount')
    addLoanStatBefore(loan_train_avg_half_std, month, 'loan_avg_half_std_before')

    magicRecover(loan_train_avg_double_std, 'loan_amount')
    addLoanStatBefore(loan_train_avg_double_std, month, 'loan_avg_double_std_before')

    magicRecover(loan_train_avg_std, 'loan_amount')
    addLoanStatBefore(loan_train_avg_std,month,'loan_avg_std_before')


    magicRecover(loan_train_mad, 'loan_amount')
    addLoanStatBefore(loan_train_mad,month,'loan_mad_before')


    magicRecover(loan_left_limit, 'left_limit')
    addLoanStatBefore(loan_left_limit[['uid','left_limit']],month, 'loan_left_limit')

    magicRecover(lookahead_loan_left_limit, 'left_limit')
    addLoanStatBefore(lookahead_loan_left_limit[['uid','left_limit']], month, 'lookahead_loan_left_limit')

    magicRecover(lookahead2_loan_left_limit, 'left_limit')
    addLoanStatBefore(lookahead2_loan_left_limit[['uid','left_limit']], month, 'lookahead2_loan_left_limit')

    magicRecover(lookahead3_loan_left_limit, 'left_limit')
    addLoanStatBefore(lookahead3_loan_left_limit[['uid','left_limit']], month, 'lookahead3_loan_left_limit')

    magicRecover(loan_train_balance, 'left_balance')
    addLoanBalance(loan_train_balance, month)

    magicRecover(lookahead_loan_train_balance, 'left_balance')
    addLoanStatBefore(lookahead_loan_train_balance, month , 'lookahead_loan_balance')

    magicRecover(lookahead2_loan_train_balance, 'left_balance')
    addLoanStatBefore(lookahead2_loan_train_balance, month , 'lookahead2_loan_balance')

    magicRecover(lookahead3_loan_train_balance, 'left_balance')
    addLoanStatBefore(lookahead3_loan_train_balance, month , 'lookahead3_loan_balance')
    # =====================================Repay
    magicRecover(repay_pressure, 'repay_pressure')
    addLoanStatBefore(repay_pressure, month, 'repay_pressure')

    magicRecover(lookahead_repay_pressure, 'repay_pressure')
    addLoanStatBefore(lookahead_repay_pressure,month,'lookahead_repay_pressure')

    magicRecover(lookahead2_repay_pressure, 'repay_pressure')
    addLoanStatBefore(lookahead2_repay_pressure,month,'lookahead2_repay_pressure')

    magicRecover(lookahead3_repay_pressure, 'repay_pressure')
    addLoanStatBefore(lookahead3_repay_pressure,month,'lookahead3_repay_pressure')

    magicRecover(repay_df, 'repay_mean_amount')
    addLoanStatBefore(repay_df[['uid', 'repay_mean_amount']], month, 'repay_mean_amount')




    #==下面的不需要调整==============================================
    addLoanStatBefore(loan_train_skewlog, month, 'loan_skewlog_before')
    addLoanStatBefore(loan_train_kurtlog,month,'loan_kurtlog_before')

    addLoanStatBefore(loan_train_skew, month, 'loan_skew_before')
    addLoanStatBefore(loan_train_kurt, month, 'loan_kurt_before')
    addLoanCountBefore(loan_train_count, month)

    ## plannum相关的信息 不需要调整
    addLoanBalancePlannumBefore(loan_train_balance_plannum, month)

    if hitPrefix(['loan_plannum_avg_before','loan_plannum_min_before', 'loan_plannum_sum_before'],stat_list):
        pdf = loan_train[['uid','plannum']].groupby('uid')
        loan_plannum_sum = pdf.sum().reset_index()
        loan_plannum_mean = pdf.mean().reset_index()
        loan_plannum_min = pdf.min().reset_index()
        loan_plannum_max = pdf.max().reset_index()
        loan_plannum_mad = pdf.mad().reset_index()

        loan_plannum_skew = pdf.skew().fillna(0).reset_index()
        loan_plannum_kurt = pdf.apply(lambda  x : x['plannum'].kurt()).fillna(0).reset_index()
        loan_plannum_kurt.columns = ['uid','kurt']

        # 'loan_plannum_avg_before', 'loan_plannum_max_before', 'loan_plannum_min_before',
        # 'loan_plannum_mad_before', 'loan_plannum_skew_before', 'loan_plannum_kurt_before',
        addLoanStatBefore(loan_plannum_sum, month, 'loan_plannum_sum_before')

        addLoanStatBefore(loan_plannum_mean, month, 'loan_plannum_avg_before')
        addLoanStatBefore(loan_plannum_min, month, 'loan_plannum_min_before')
        addLoanStatBefore(loan_plannum_max, month, 'loan_plannum_max_before')
        addLoanStatBefore(loan_plannum_mad, month, 'loan_plannum_mad_before')
        addLoanStatBefore(loan_plannum_skew, month, 'loan_plannum_skew_before')
        addLoanStatBefore(loan_plannum_kurt, month, 'loan_plannum_kurt_before')
    else:
        logging.info('NOT hit loan_plannum_avg_before')

    if hitPrefix(['loan_perplannum_avg_before'],stat_list):
        logging.info('hit loan_perplannum_avg_before')
        loan_train['per_amount'] = loan_train['loan_amount'] / loan_train['plannum']

        pdf = loan_train[['uid','per_amount']].groupby('uid')

        loan_perplannum_mean = pdf.mean().reset_index()
        loan_perplannum_min = pdf.min().reset_index()
        loan_perplannum_max = pdf.max().reset_index()
        loan_perplannum_mad = pdf.mad().reset_index()

        loan_perplannum_skew = pdf.skew().fillna(0).reset_index()
        loan_perplannum_kurt = pdf.apply(lambda x: x['per_amount'].kurt()).fillna(0).reset_index()
        loan_perplannum_kurt.columns = ['uid', 'kurt']

        magicRecover(loan_perplannum_mean,'per_amount')
        addLoanStatBefore(loan_perplannum_mean, month, 'loan_perplannum_avg_before')

        magicRecover(loan_perplannum_min, 'per_amount')
        addLoanStatBefore(loan_perplannum_min, month, 'loan_perplannum_min_before')

        magicRecover(loan_perplannum_max, 'per_amount')
        addLoanStatBefore(loan_perplannum_max, month, 'loan_perplannum_max_before')

        magicRecover(loan_perplannum_mad, 'per_amount')
        addLoanStatBefore(loan_perplannum_mad, month, 'loan_perplannum_mad_before')


        addLoanStatBefore(loan_perplannum_skew, month, 'loan_perplannum_skew_before')
        addLoanStatBefore(loan_perplannum_kurt, month, 'loan_perplannum_kurt_before')




def magicRecover(df, column):
    df[column] = df[column].apply(lambda x: math.log(x + 1, 5))

def addLoan():
    for i in range(6):
        tag = 'svd_loan_param{i}'.format(i=i)
        tryAdd(tag,tag, feature_list)

    tryAdd('ts_loan_mean','ts_loan_mean',feature_list)
    tryAdd('ts_loan_mad','ts_loan_mad',feature_list)
    tryAdd('ts_loan_skew','ts_loan_skew',feature_list)
    tryAdd('ts_loan_kurt','ts_loan_kurt',feature_list)

def addLoanStat():
    """
    历史借贷总额 10:loan_sum_before:{uid} value

    上个月借贷总额

    截止目前的贷款余额
    :return:
    """
    logging.info('addLoanStat ing...')
    loan = pd.read_csv(t_loan,header = 0 )
    loan['month'] = loan['loan_time'].apply(lambda x: x[5:7])
    loan['raw_loan_amount'] = loan['loan_amount'] #skew and kurt 用
    loan['loan_amount'] = 5 ** loan['loan_amount'] - 1
    #loan['limit'] = 5 ** loan['limit'] - 1

    addLoanStatTool(loan, train_month, train_next)
    # addLoanStatTool(loan, test_month, '12')

    # addPlannumStatTool(loan, train_month, '11')
    # addPlannumStatTool(loan, test_month, '12')


    loan_train_prev = loan[loan['month'].isin([train_prev])]


    addPreviousLoanStatTool(loan_train_prev, train_next)
    # addLoanSumPrevious(loan_train_prev_sum, '11')

    # loan_pred_prev = loan[loan['month'].isin([test_prev])]
    # loan_pred_prev_sum = loan_pred_prev[['uid', 'loan_amount']].groupby(['uid']).sum().reset_index()
    #
    # magicRecover(loan_pred_prev_sum, 'loan_amount')
    # addLoanStatBefore(loan_pred_prev_sum, '12', 'loan_sum_previous')
    #
    # loan_pred_prev_cnt = loan_pred_prev[['uid', 'loan_time']].groupby(['uid']).count().reset_index()
    # addLoanStatBefore(loan_pred_prev_cnt, '12', 'loan_count_previous')
    # addLoanSumPrevious(loan_pred_prev_sum, '12')


    ########两个previous，看是否管用

    loan_train_prev = loan[loan['month'].isin([train_prev2])]
    loan_train_prev_sum = loan_train_prev[['uid', 'loan_amount']].groupby(['uid']).sum().reset_index()

    magicRecover(loan_train_prev_sum, 'loan_amount')
    addLoanStatBefore(loan_train_prev_sum, train_next, 'loan_sum_previous2')

    loan_train_prev_sumlog = loan_train_prev[['uid', 'raw_loan_amount']].groupby(['uid']).sum().reset_index()
    addLoanStatBefore(loan_train_prev_sumlog, train_next, 'loan_sumlog_previous2')




    # loan_pred_prev = loan[loan['month'].isin([test_prev2])]
    # loan_pred_prev_sum = loan_pred_prev[['uid', 'loan_amount']].groupby(['uid']).sum().reset_index()
    #
    # magicRecover(loan_pred_prev_sum, 'loan_amount')
    # addLoanStatBefore(loan_pred_prev_sum, '12', 'loan_sum_previous2')



    pass
def addOrderClick():
    order = pd.read_csv(t_order, header=0)
    click = pd.read_csv(t_click, header=0)

    # 购买的总金额
    # 购买品类的 平均贷款额度
    for cate in order['cate_id'].unique():
        for pid in click['pid'].unique():
            feastr = 'order_click:{0}_{1}'.format(cate,pid)
            tryAdd('order_click', feastr, feature_list)

    pass
def addOrderClickStat():
    pass

def addFeature():
    """
    根据数据填充 Feature 等对象.
    :return:
    """

    addUser()
    addUserStat()
    addLoan()
    addLoanStat()
    addOrder()
    addOrderStat()


    addClick()
    addClickStat()

    # addOrderClick()
    # addOrderClickStat()

    pass
##===========================================================

def transformStatFeatureStr(prefix,feastr):
    buf = []
    for fea in stat_list:
        o = fea #type: StatFeature
        x = o.transform(prefix, feastr,':')
        if x!=-1 :
            buf.append(x)
    return buf

def transformFeatureStr(prefix, feastr, val=1):
    buf = []
    for fea in feature_list:
        o =  fea #type:Feature
        x = o.transform(prefix, feastr , ':',val )
        if x != -1 and float(val) != 0 :
            buf.append(x)




    # for fea in superstat_list:
    #     o = fea #type: SuperStatFeature
    #     x = o.tran


    return buf

def transformOrderStat():
    logging.info('transformOrderStat begin')

    uidmap = {}
    if not hitPrefix(['buy_cate_avg_loan'], stat_list):
        return uidmap

    order = pd.read_csv(t_order,header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])


    order_train = order[order['month'].isin(train_month)]  # train sample 8 -10


    uid_order = order_train[['uid','cate_id','qty']].groupby(['uid','cate_id']).sum().reset_index()

    ptuid = pd.pivot_table(uid_order , index='uid', values='qty', columns='cate_id',fill_value=0).reset_index()



    df = ptuid

    cols = df.columns.values

    r, c = df.shape


    col_uid = df.columns.values.tolist().index('uid')

    uidmap = {}

    missset = ['uid', 'month', 'loan_sum','qty']

    dvals = df.values
    for i in range(r):
        rbuf = []
        row = dvals[i]



        uid = int(row[col_uid])
        if uid in fullset:
            for j in range(c):
                if cols[j] not in missset:
                    qty = row[j]

                    feastr = '{0}:buy_cate_avg_loan:{1}'.format(train_next , int(cols[j]))
                    if qty > 0:
                        buf = transformStatFeatureStr('buy_cate_avg_loan', feastr)
                        rbuf.extend(buf)

            # rbuf.sort(key=lambda x: x.split(":")[0])  # sort in place
            uidmap[uid] =' '.join(rbuf)
        pass
    logging.info('transformOrderStat end')
    return uidmap

def transformOrderDiscountSum():
    logging.info('transformOrderDiscountSum Begin')
    order = pd.read_csv(t_order, header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order['discount'] = 5 ** order['discount'] - 1

    order_train = order[order['month'].isin(train_month)]  # train sample 8 -10


    order_sum = order_train[['uid', 'discount']].groupby('uid').sum().reset_index()
    magicRecover(order_sum, 'discount')

    df = order_sum

    r, c = df.shape

    col_uid = df.columns.values.tolist().index('uid')

    uidmap = {}
    dvals = df.values
    for i in range(r):
        rbuf = []
        row = dvals[i]

        uid = int(row[col_uid])
        buy_discount_sum = float(row[1])
        if uid in fullset:
            buf = transformFeatureStr('buy_discount_sum', 'buy_discount_sum', buy_discount_sum)
            rbuf.extend(buf)

            uidmap[uid] = ' '.join(rbuf)
        pass
    logging.info('transformOrderDiscountSum end')

    return uidmap
def transformOrderSum():
    """
    order stat
    """
    logging.info('transformOrderSum Begin')
    order = pd.read_csv(t_order, header=0)
    order['price'] = 5 ** order['price'] -1
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])

    order_train = order[order['month'].isin(train_month)]  # train sample 8 -10


    order_train['total_amount'] = order_train['price'] * order_train['qty']
    # 65 is median
    order_train['large_l1'] = order_train['total_amount'].apply(lambda x: 1 if x > 65 * 2 else 0)
    order_train['large_l2'] = order_train['total_amount'].apply(lambda x: 1 if x > 65 * 10 else 0)
    order_train['large_l3'] = order_train['total_amount'].apply(lambda x: 1 if x > 65 * 100 else 0)
    order_train['large_l4'] = order_train['total_amount'].apply(lambda x: 1 if x > 65 * 1000 else 0)

    order_large1 = order_train[['uid', 'large_l1']].groupby('uid').sum().reset_index() 
    order_large2 = order_train[['uid', 'large_l2']].groupby('uid').sum().reset_index() 
    order_large3 = order_train[['uid', 'large_l3']].groupby('uid').sum().reset_index() 
    order_large4 = order_train[['uid', 'large_l4']].groupby('uid').sum().reset_index() 
    

    gdf = order_train[['uid',  'total_amount']].groupby('uid')

    logging.info('order suming...')
    order_sum  = gdf.sum().reset_index()
    order_sum.columns = ['uid','sum']

    logging.info('order cnting...')
    order_count= gdf.count().reset_index()
    order_count.columns = ['uid','count']

    logging.info('order min ing...')
    order_min  = gdf.min().reset_index()
    order_min.columns = ['uid','min']

    logging.info('order max ing...')
    order_max  = gdf.max().reset_index()
    order_max.columns = ['uid','max']

    logging.info('order mean ing...')
    order_mean  = gdf.mean().reset_index()
    order_mean.columns = ['uid','mean']

    logging.info('order mad ing...')
    order_mad  = gdf.std().fillna(0).reset_index()
    order_mad.columns = ['uid','mad']

    logging.info('order skew ing...')
    order_skew  = gdf.std().fillna(0).reset_index()
    order_skew.columns = ['uid','skew']

    magicRecover(order_sum,'sum')
    magicRecover(order_min,'min')
    magicRecover(order_max,'max')
    magicRecover(order_mean,'mean')
    magicRecover(order_mad,'mad')
    #magicRecover(order_skew,'skew')

    logging.info('order concat ing...')
    #df = pd.concat( [order_sum, order_min, order_max, order_mean, order_mad, order_skew, order_count, order_large1,order_large2, order_large3, order_large4],  axis =1)
    logging.info('order concat end ...')

    #df = order_sum

    #r, c = df.shape

    #col_uid = df.columns.values.tolist().index('uid')
    def transformOrderTool(df, prefix):
        logging.info('transformOrdertool ing ..'+ prefix)
        uidmap = {}
        r,c = df.shape
        dvals = df.values
        for i in range(r):
            rbuf = []
            row =dvals[i] 
    
            uid , val = int(row[0]) , row[1]
            #feastr = '{0}:{1}'.format(prefix, cate_id)
            buf = transformFeatureStr(prefix, prefix, val)
            if len(buf) > 0:
                fs = ' '.join(buf)
                if uid in uidmap:
                    uidmap[uid] = uidmap[uid] + ' ' + fs
                else:
                    uidmap[uid] = fs
        return uidmap
    summap = transformOrderTool(order_sum,'buy_sum')
    minmap = transformOrderTool(order_min,'buy_min')
    maxmap = transformOrderTool(order_max,'buy_max')
    meanmap = transformOrderTool(order_mean,'buy_mean')
    madmap = transformOrderTool(order_mad,'buy_mad')
    skewmap = transformOrderTool(order_skew,'buy_skew')
    cntmap = transformOrderTool(order_count,'buy_count')
    large1map = transformOrderTool(order_large1,'buy_large1_count')
    large2map = transformOrderTool(order_large2,'buy_large2_count')
    large3map = transformOrderTool(order_large3,'buy_large3_count')
    large4map = transformOrderTool(order_large4,'buy_large4_count')
    
    uidmap = mergeUidMap(summap, minmap, maxmap, meanmap, madmap, skewmap, cntmap, large1map, large2map, large3map, large4map)
    logging.info('transformOrderSum end')


    return uidmap
def transformOrderDiscount():
    logging.info('transformOrderDiscount Begin')
    order = pd.read_csv(t_order, header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order['discount'] = 5 ** order['discount'] - 1

    order_train = order[order['month'].isin(train_month)]  # train sample 8 -10


    order_clean = order_train[['uid', 'cate_id', 'discount']].groupby(['uid', 'cate_id']).sum().reset_index()

    magicRecover(order_clean , 'discount')
    def transformOrderTool(df, prefix):
        logging.info('transformOrderDiscounttool ing ..'+ prefix)
        uidmap = {}
        r,c = df.shape
        dvals = df.values
        for i in range(r):
            rbuf = []
            row =dvals[i] 
    
            uid , cate_id  = int(row[0]) , int(row[1])
            val= row[2]
            feastr = '{0}:{1}'.format(prefix, cate_id)
            buf = transformFeatureStr(prefix, feastr, val)
            
            if len(buf) > 0:
                fs = ' '.join(buf)
                if uid in uidmap:
                    uidmap[uid] = uidmap[uid] + ' ' + fs
                else:
                    uidmap[uid] = fs
        return uidmap



    uidmap =  transformOrderTool(order_clean, 'buy_cate_discount')

    logging.info('transformOrderDiscount end')

    return uidmap
def transformSVDOrder():
    logging.info('transformSVD order ...')
    order_svd = pd.read_csv(t_order_svd,header=0)
    dvals = order_svd.values
    uidmap = {}
    for row in dvals:
        uid = int(row[0])
        buf = []
        for i in range(1, 6):
            tag = 'svd_order_param{0}'.format(i)
            buf.extend(transformFeatureStr(tag, tag, row[i]))
        uidmap[uid] = ' '.join(buf)
    logging.info('DONE transformSVD order')
    return uidmap
def transformOrder():
    logging.info('transformOrder Begin')
    order = pd.read_csv(t_order, header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order['price'] = 5 ** order['price'] - 1 #转化成原来的值 才可以

    order_train = order[order['month'].isin(train_month)] # train sample 8 -10

    order_train['total_amount'] = order_train['price'] * order_train['qty']
    
    gdf = order_train[['uid','cate_id','total_amount']].groupby(['uid','cate_id'])
    logging.info('order_sum ing..')
    order_sum = gdf.sum().reset_index()
    logging.info('order_cnt ing..')
    order_cnt = gdf.count().reset_index()
    logging.info('order_min ing..')
    order_min = gdf.min().reset_index()
    logging.info('order_max ing..')
    order_max = gdf.max().reset_index() 
    logging.info('order_mean ing..')
    order_mean = gdf.mean().reset_index() 

    logging.info('order_std ing..')
    order_std = gdf.std().fillna(0).reset_index() 

    #logging.info('order_mad ing..')
    #order_mad = gdf.mad().reset_index() 
    #logging.info('order_skew ing..')
    #order_skew = gdf.skew().fillna(0).reset_index() 


    magicRecover(order_sum, 'total_amount')
    magicRecover(order_min, 'total_amount')
    magicRecover(order_max, 'total_amount')
    #magicRecover(order_mad, 'total_amount')
    magicRecover(order_mean, 'total_amount')
    logging.info('concat order sum ,order cnt,order min ,order max ,etc') 
    #df =pd.concat([order_sum,order_cnt,order_min,order_max, order_mean, order_mad, order_skew], axis=1)
    #ptorder = pd.pivot_table(order_clean , index= 'uid', values='total_amount', columns='cate_id' ,fill_value= 0).reset_index()
    #cols = df.columns.values
    #r, c = df.shape

    #col_uid = df.columns.values.tolist().index('uid')

    def transformOrderTool(df, prefix):
    #missset = ['uid','month','loan_sum']
        logging.info('transformOrdertool ing ..'+ prefix)
        uidmap = {}
        r,c = df.shape
        dvals = df.values
        for i in range(r):
            rbuf = []
            row =dvals[i] 
    
            uid , cate_id  = int(row[0]) , int(row[1])
            val= row[2]
#            ocnt = row[5]
#            omin = row[8]
#            omax = row[11]
#            omean = row[14]
#            omad = row[17]
#            oskew = row[20]
#    
            feastr = '{0}:{1}'.format(prefix, cate_id)
            buf = transformFeatureStr(prefix, feastr, val)
            if len(buf) > 0:
                fs = ' '.join(buf)
                if uid in uidmap:
                    uidmap[uid] = uidmap[uid] + ' ' + fs
                else:
                    uidmap[uid] = fs
#            for prefix, val in [('buy_cate', osum) , ('buy_cate_count',ocnt), ('buy_cate_min', omin) ,('buy_cate_max', omax), ('buy_cate_mean', omean), ('buy_cate_mad',omad),('buy_cate_skew',oskew)]:
#                feastr = '{0}:{1}'.format(prefix, cate_id)
#                buf = transformFeatureStr(prefix, feastr, val )
#                rbuf.extend(buf)
#    
#                # rbuf.sort(key=lambda x: x.split(":")[0])  # sort in place
#            uidmap[uid] =' '.join(rbuf)
        return uidmap
    summap = transformOrderTool(order_sum,'buy_cate')
    cntmap= transformOrderTool(order_cnt,'buy_cate_count')
    minmap = transformOrderTool(order_min,'buy_cate_min')
    maxmap = transformOrderTool(order_max,'buy_cate_max')
    #madmap = transformOrderTool(order_mad,'buy_cate_mad')
    meanmap = transformOrderTool(order_mean,'buy_cate_mean')
    stdmap = transformOrderTool(order_std,'buy_cate_std')
#    skewmap = transformOrderTool(order_skew,'buy_cate_skew')

    uidmap = mergeUidMap(summap, cntmap, minmap, maxmap,stdmap , meanmap) # , madmap,skewmap)
    logging.info('transformOrder end')

    return uidmap


def transformOrder_():
    logging.info('transformOrder Begin')
    order = pd.read_csv(t_order, header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order['price'] = 5 ** order['price'] - 1 #转化成原来的值 才可以

    order_train = order[order['month'].isin(train_month)] # train sample 8 -10

    order_train['total_amount'] = order_train['price'] * order_train['qty']

    order_clean = order_train[['uid','cate_id','total_amount']].groupby(['uid','cate_id']).sum().reset_index()

    magicRecover(order_clean, 'total_amount')

    ptorder = pd.pivot_table(order_clean , index= 'uid', values='total_amount', columns='cate_id' ,fill_value= 0).reset_index()

    df = ptorder

    cols = df.columns.values

    r, c = df.shape

    col_uid = df.columns.values.tolist().index('uid')

    uidmap = {}

    missset = ['uid','month','loan_sum']
    for i in range(r):
        rbuf = []
        row = df.iloc[i, :]

        uid = row[col_uid]
        if uid in fullset:

            for j in range(c):
                if cols[j] not in missset:
                    val = row[j]

                    feastr = 'buy_cate:{0}'.format(cols[j])

                    buf = transformFeatureStr('buy_cate', feastr, val )
                    rbuf.extend(buf)

            # rbuf.sort(key=lambda x: x.split(":")[0])  # sort in place
            uidmap[uid] =' '.join(rbuf)
        pass
    logging.info('transformOrder end')

    return uidmap




def transformUser():
    logging.info("transfromUser begin")

    df = pd.read_csv(t_user, header=0)



    cols = df.columns.values

    # missset = ['uid']
    missset = ['uid'] # uid can descrease rmse.
    r, c = df.shape


    col_uid  = df.columns.values.tolist().index('uid')
    col_active_date = df.columns.values.tolist().index('active_date')


    uidmap =  {}
    dvals = df.values
    for i in range(r):

        rbuf = []
        row = dvals[i]
        uid = int(row[col_uid])
        active_date  = str(row[col_active_date])
        if uid in fullset:
            days_to_now = (datetime.datetime.strptime('2016-12-01', '%Y-%m-%d') - datetime.datetime.strptime(active_date,'%Y-%m-%d')).days

            buf = transformFeatureStr('days_to_now', 'days_to_now', val=days_to_now)
            rbuf.extend(buf)
            buf = transformFeatureStr('uid', 'uid', val=uid)
            rbuf.extend(buf)
            for j in range(c):
                if cols[j] not in missset:
                    feastr = '{0}:{1}'.format(cols[j], row[j])
                    # if j == col_uid:
                    #     buf = transformFeatureStr(cols[j], feastr, val=int(row[j]))
                    # else:
                    buf = transformFeatureStr(cols[j], feastr)
                    rbuf.extend(buf)

            # rbuf.sort(key=lambda x: x.split(":")[0])  # sort in place
            uidmap[uid] =' '.join(rbuf)
    logging.info("transfromUser end")
    return uidmap


def mergeUidMap(*params):
    uidmap = {}
    for umap in params:

        for uid in umap:
            if uid in uidmap:
                uidmap[uid] = '{0} {1}'.format(uidmap[uid], umap[uid])
            else:
                uidmap[uid] = umap[uid]
    return uidmap
def transformLoan():
    logging.info('transformLoan  Begin')

    # loan_svd = pd.read_csv(t_loan_svd, header=0)
    # dvals = loan_svd.values
    # uidmap = {}
    # for row in dvals:
    #     uid = int(row[0])
    #     buf = []
    #     for i in range(1, 6):
    #         tag = 'svd_loan_param{0}'.format(i)
    #         buf.extend(transformFeatureStr(tag, tag, row[i]))
    #     uidmap[uid] = ' '.join(buf)

    loan_ts = pd.read_csv(t_loan_ts, header=None)
    loan_ts.columns = ['uid','mean','mad','skew','kurt','cnt']
    dvals = loan_ts.values
    uidmap = {}
    for row in dvals:
        uid = int(row[0])
        buf = []
        buf.extend( transformFeatureStr('ts_loan_mean','ts_loan_mean', row[1]))
        buf.extend( transformFeatureStr('ts_loan_mad','ts_loan_mad', row[2]))
        buf.extend( transformFeatureStr('ts_loan_skew','ts_loan_skew', row[3]))
        buf.extend(transformFeatureStr('ts_loan_kurt', 'ts_loan_kurt', row[4]))
        uidmap[uid] = ' '.join(buf)
    logging.info('transformLoan  Done')
    return uidmap

def transformLoanStat():
    logging.info('transformLoanStat Begin')
    uidmap = {}

    pbuf  = [
            'loan_sum_rank',
'lookback_stack_sum1',
'lookback_stack_sum2',
'lookback_stack_sum3',
'lookback_stack_sum4',
'lookback_stack_sum5',
'lookback_stack_sum6',
'lookback_stack_sum7',
'lookback_stack_sum8',
'lookback_stack_sum9',
'lookback_stack_sum10',
'lookback_stack_sum11',
'lookback_stack_sum12',
'lookback_stack_sum13',
'lookback_stack_sum14',
'lookback_stack_sum15',
'lookback_stack_sum16',
'lookback_stack_sum17',
'lookback_stack_sum18',
'lookback_stack_sum19',
        'lookback1_loan_sum', 'lookback2_loan_sum', 'lookback3_loan_sum', 'lookback4_loan_sum', 'lookback5_loan_sum',
        'lookback6_loan_sum', 'lookback7_loan_sum', 'lookback8_loan_sum', 'lookback9_loan_sum', 'lookback10_loan_sum',
        'lookback11_loan_sum', 'lookback12_loan_sum', 'lookback13_loan_sum', 'lookback14_loan_sum',
        'lookback15_loan_sum', 'lookback16_loan_sum', 'lookback17_loan_sum', 'lookback18_loan_sum',
        'lookback19_loan_sum', 'lookback20_loan_sum', 'lookback21_loan_sum', 'lookback22_loan_sum',
        'lookback23_loan_sum', 'lookback24_loan_sum', 'lookback25_loan_sum', 'lookback26_loan_sum',
        'lookback27_loan_sum', 'lookback28_loan_sum', 'lookback29_loan_sum', 'lookback30_loan_sum',
        'lookback31_loan_sum', 'lookback32_loan_sum', 'lookback33_loan_sum', 'lookback34_loan_sum',
        'lookback35_loan_sum', 'lookback36_loan_sum', 'lookback37_loan_sum', 'lookback38_loan_sum',
        'lookback39_loan_sum', 'lookback40_loan_sum', 'lookback41_loan_sum', 'lookback42_loan_sum',
        'lookback43_loan_sum', 'lookback44_loan_sum','lookback45_loan_sum',

        'lookback46_loan_sum', 'lookback47_loan_sum', 'lookback48_loan_sum', 'lookback49_loan_sum',
        'lookback50_loan_sum', 'lookback51_loan_sum', 'lookback52_loan_sum', 'lookback53_loan_sum',
        'lookback54_loan_sum', 'lookback55_loan_sum', 'lookback56_loan_sum', 'lookback57_loan_sum',
        'lookback58_loan_sum', 'lookback59_loan_sum',

        'lookahead1_repay','lookahead2_repay','lookahead3_repay','lookahead4_repay','lookahead5_repay','lookahead6_repay','lookahead7_repay','lookahead8_repay','lookahead9_repay','lookahead10_repay','lookahead11_repay','lookahead12_repay','lookahead13_repay','lookahead14_repay','lookahead15_repay','lookahead16_repay','lookahead17_repay','lookahead18_repay','lookahead19_repay','lookahead20_repay','lookahead21_repay','lookahead22_repay','lookahead23_repay','lookahead24_repay','lookahead25_repay','lookahead26_repay','lookahead27_repay','lookahead28_repay','lookahead29_repay','lookahead30_repay','lookahead31_repay','lookahead32_repay','lookahead33_repay','lookahead34_repay','lookahead35_repay','lookahead36_repay','lookahead37_repay','lookahead38_repay','lookahead39_repay','lookahead40_repay','lookahead41_repay','lookahead42_repay','lookahead43_repay','lookahead44_repay','lookahead45_repay',
        'lookahead46_repay', 'lookahead47_repay', 'lookahead48_repay', 'lookahead49_repay', 'lookahead50_repay',
        'lookahead51_repay', 'lookahead52_repay', 'lookahead53_repay', 'lookahead54_repay', 'lookahead55_repay',
        'lookahead56_repay', 'lookahead57_repay', 'lookahead58_repay', 'lookahead59_repay',

        'loan_sumlog_before',
            'loan_sum_before',  'loan_consecutive_sum_before','loan_consecutive_left_limit',
             'loan_weight_sum_before',

            'loan_median_before','loan_min_before','loan_max_before',

            'loan_skew_before','loan_kurt_before',
            'loan_skewlog_before', 'loan_kurtlog_before',

             'loan_mad_before',      'loan_madlog_before',
            'loan_avg_before', 'loan_avglog_before',
             'loan_avg_std_before','loan_avg_half_std_before','loan_avg_double_std_before',


             'loan_sum_previous','loan_sum_previous2',
             'loan_avg_previous','loan_min_previous','loan_median_previous','loan_max_previous','loan_mad_previous','loan_skew_previous','loan_kurt_previous',
             'loan_sumlog_previous','loan_sumlog_previous2',

             'loan_balance',
             'limit_in_use', 'limit_free','limit_use_rate',
             'lookahead_limit_in_use', 'lookahead_limit_free', 'lookahead_limit_use_rate',
            'lookahead2_limit_in_use', 'lookahead2_limit_free', 'lookahead2_limit_use_rate',
            'lookahead3_limit_in_use', 'lookahead3_limit_free', 'lookahead3_limit_use_rate',

             'loan_left_limit','loan_left_magic_limit',
              'loan_max_rate',


              'loan_day_min', 'loan_day_max','loan_day_mean','loan_day_median','loan_day_mad', 'loan_day_skew',
              'loan_day_min1','loan_day_max1','loan_day_mad1','loan_day_median1',
              'loan_day_min3','loan_day_max3','loan_day_mad3','loan_day_median3',
              'loan_day_min6','loan_day_max6','loan_day_mad6','loan_day_median6',
              'loan_day_min12','loan_day_max12','loan_day_mad12','loan_day_median12',
               'loan_past_day_min', 'loan_past_day_max','loan_past_day_mad','loan_past_day_mean','loan_past_day_median','loan_past_day_skew',
              'loan_plannum1_count','loan_plannum1_sum',
              'loan_plannum3_count','loan_plannum3_sum',
              'loan_plannum6_count','loan_plannum6_sum',
              'loan_plannum12_count','loan_plannum12_sum',
             'repay_limit_rate', 'repay_pressure', 'repay_mean_amount',  #下个月的还款状态



                'repay_plannum',            'repay_marginal_clear_count',           'repay_marginal_clear_amount',              'repay_clear_count',        'repay_clear_amount',
                'lookahead_repay_plannum',  'lookahead_repay_marginal_clear_count', 'lookahead_repay_marginal_clear_amount',  'lookahead_repay_clear_count', 'lookahead_repay_clear_amount',

              # 'loan_previous_left_limit',

              'lookahead_loan_balance' , 'lookahead_repay_pressure' , 'lookahead_loan_left_limit',
               'lookahead2_loan_balance', 'lookahead2_repay_pressure', 'lookahead2_loan_left_limit',
             'lookahead3_loan_balance', 'lookahead3_repay_pressure', 'lookahead3_loan_left_limit',


             'loan_plannum_sum_before','loan_plannum_avg_before', 'loan_plannum_max_before','loan_plannum_min_before','loan_plannum_mad_before', 'loan_plannum_skew_before','loan_plannum_kurt_before',
             'loan_perplannum_avg_before', 'loan_perplannum_max_before', 'loan_perplannum_min_before', 'loan_perplannum_mad_before', 'loan_perplannum_skew_before', 'loan_perplannum_kurt_before',

             'loan_balance_plannum',
             'loan_count_before',
             'loan_count_previous',

             ]

    #for i in range(10):
    #    pbuf.append('loan_weight_sum_before_{0}'.format(i))

    for uid in fullset:

        rbuf = []
        for prefix in pbuf:
            feastr = '{0}:{1}:{2}'.format(train_next, prefix, uid)
            buf = transformStatFeatureStr(prefix, feastr)
            rbuf.extend(buf)

        uidmap[uid] = ' '.join(rbuf)

    logging.info('transformLoanStat end')

    return uidmap


def transformUserStat():
    logging.info('transformUserStat Begin')
    # loan_sum = pd.read_csv(t_loan_sum, header=0)
    df = pd.read_csv(t_user, header=0)

    # df = pd.merge(loan_sum, user, on='uid')  # type:pd.DataFrame
    cols = df.columns.values

    r, c = df.shape

    liveset = ['age','sex','active_date']

    col_uid = df.columns.values.tolist().index('uid')

    uidmap = {}
    dvals = df.values
    for i in range(r):
        rbuf = []
        row = dvals[i]
        uid = row[col_uid]

        if uid in fullset:
            for j in range(c):
                if cols[j]  in liveset:
                    feastr = '{0}:{1}:{2}'.format(train_next, cols[j], row[j])

                    buf = transformStatFeatureStr(cols[j], feastr)
                    rbuf.extend(buf)

            # rbuf.sort(key=lambda x: x.split(":")[0])  # sort in place
            uidmap[uid] =' '.join(rbuf)
    logging.info('transformUserStat end')
    return uidmap


def transformClickPid(click_train):
    logging.info('transformClickPid ing..')

    if not hitPrefix(['click'] , feature_list):
        logging.info("not hit click")
        return {}
    click_cnt_df = click_train[['uid', 'pid', 'click_time']].groupby(['uid', 'pid']).count().reset_index()

    logging.info('transformClickPid pivot ing..')
    pdf = pd.pivot_table(click_cnt_df, index='uid', values='click_time', columns='pid', fill_value=0).reset_index()
    logging.info('transformClickPid pivot done..')

    df =pdf

    cols = df.columns.values

    r, c = df.shape
    col_uid = df.columns.values.tolist().index('uid')
    uidmap = {}
    missset = ['uid', 'month', 'click_time', 'loan_sum']
    
    dvals = df.values
    for i in range(r):
        rbuf = []
        #row = df.iloc[i, :]
        row = dvals[i]

        uid = int(row[col_uid])
        if uid in fullset:
            for j in range(c):
                if cols[j] not in missset:
                    click_cnt = row[j]
                    if click_cnt > 0:
                        feastr = 'click:{pid}'.format(pid=cols[j])
                        buf = transformFeatureStr('click', feastr, click_cnt)
                        rbuf.extend(buf)
            uidmap[uid] = ' '.join(rbuf)
    return uidmap


def transformClickPidParam(click_train):


    if not hitPrefix(['click_param'] , feature_list):
        logging.info('not hit click_param')
        return {}
    click_cnt_df = click_train[['uid', 'pid', 'param','click_time']].groupby(['uid', 'pid','param']).count().reset_index()

    pdf = pd.pivot_table(click_cnt_df, index='uid', values='click_time', columns=['pid','param'], fill_value=0).reset_index()

    if sys.argv[1] == 'train':
        loan_sum = pd.read_csv(t_loan_sum,header= 0)
        df =pd.merge(loan_sum, pdf,on='uid')
    else:
        user = pd.read_csv(t_user,header= 0)
        df = pd.merge(user, pdf,on='uid')

    cols = df.columns.values

    r, c = df.shape
    col_uid = df.columns.values.tolist().index('uid') #col uid = 0  这里columns 是  tuple形式  直接 index('uid')报错
    uidmap = {}
    missset = ['uid', 'month', 'click_time', 'loan_sum','age','sex','active_date','limit']
    dvals = df.values
    for i in range(r):
        rbuf = []
        #row = df.iloc[i, :]
        row = dvals[i]

        uid = int(row[col_uid])
        if uid in fullset:
            for j in range(c):
                if cols[j] not in missset:
                    click_cnt = row[j]
                    try:
                        if click_cnt > 0:
                            pid, param = cols[j]
                            feastr = 'click_param:{pid}_{param}'.format(pid=pid,param=param)
                            # print feastr
                            buf = transformFeatureStr('click_param', feastr, click_cnt)
                            rbuf.extend(buf)
                    except Exception as e:
                        traceback.print_exc()
            uidmap[uid] = ' '.join(rbuf)
    return uidmap
def transformSVDClick():
    logging.info('transform svd click')
    click_svd = pd.read_csv(t_click_svd,header=0)

    dvals = click_svd.values
    uidmap = {}
    for row in dvals:
        uid= int(row[0])
        # s1 = row[1]
        # s2 = row[2]
        # s3 = row[3]
        # s4 = row[4]
        # s5 = row[5]
        buf = []
        for i in range(1,6):
            tag = 'svd_click_param{0}'.format(i)
            buf.extend( transformFeatureStr(tag,tag, row[i] ) )
        uidmap[uid] = ' '.join(buf)
    logging.info('DONE  transform svd click')
    return uidmap

def transformClick():
    """
    Feature(prefix='click', startid=1,name='click') , #历史是否点击某个商品
    Feature(prefix='click_param', startid=1, name='click_param'),  # 历史是否点击某个商品+param;
    :return:
    """
    logging.info('transformClick start')
    click = pd.read_csv(t_click,header=0)
    click['month'] = click['click_time'].apply(lambda x: x[5:7])
    click_train = click[click['month'].isin(train_month)]



    uidmap1 = transformClickPid(click_train)
    uidmap2 = transformClickPidParam(click_train)

    uidmap3 = transformSVDClick()

    # uidmap = {}

    uidmap = mergeUidMap(uidmap1,uidmap2,uidmap3)

    # for uid in uidmap1:
    #     if uid in uidmap2:
    #         uidmap[uid] = '{0} {1}'.format(uidmap1[uid],uidmap2[uid])
    #     else:
    #         uidmap[uid] = uidmap1[uid]



    logging.info('transformClick end')
    return uidmap

def transformOrderClick():
    """
    经过测试，这组特征用处不大
    :return:
    """
    logging.info('transformOrderClick start')
    click = pd.read_csv(t_click, header=0)
    click['month'] = click['click_time'].apply(lambda x: x[5:7])
    click_train = click[click['month'].isin(train_month)]

    order = pd.read_csv(t_order, header=0)
    order['month'] = order['buy_time'].apply(lambda x: x[5:7])
    order_train = order[order['month'].isin(train_month)]


    click_df = click_train[['uid', 'pid', 'click_time']].groupby(['uid', 'pid']).count().reset_index()
    order_df = order_train[['uid', 'cate_id', 'buy_time']].groupby(['uid', 'cate_id']).count().reset_index()

    mdf = pd.merge(click_df, order_df, on='uid')
    pt = pd.pivot_table(mdf, index='uid', columns=['pid', 'cate_id'], values='click_time', fill_value=0).reset_index()

    r, c = pt.shape
    uidmap = {}
    for i in range(r):
        rbuf = []
        row = pt.iloc[i,:]
        uid = int(row[0])
        for j in range(1,c):
            pid,cate_id = pt.columns.values[j]
            val = row[pid,cate_id]
            if int(val) > 0:
                feastr = 'order_click:{0}_{1}'.format(cate_id,pid)
                rbuf.extend(transformFeatureStr('order_click', feastr, 1.0))

        uidmap[uid] = ' '.join(rbuf)

    logging.info('transformOrderClick end')
    return uidmap


def transformClickStatPidParam(click_train):
    click_cnt_df = click_train[['uid', 'pid', 'param', 'click_time']].groupby(
        ['uid', 'pid', 'param']).count().reset_index()

    pdf = pd.pivot_table(click_cnt_df, index='uid', values='click_time', columns=['pid', 'param'],
                         fill_value=0).reset_index()

    if sys.argv[1] == 'train':
        loan_sum = pd.read_csv(t_loan_sum,header= 0)
        df =pd.merge(loan_sum, pdf,on='uid')
    else:
        user = pd.read_csv(t_user,header= 0)
        df = pd.merge(user, pdf,on='uid')


    cols = df.columns.values

    r, c = df.shape
    col_uid = df.columns.values.tolist().index('uid')
    uidmap = {}
    missset = ['uid', 'month', 'click_time', 'loan_sum','age','sex','limit','active_date']

    for i in range(r):
        rbuf = []
        row = df.iloc[i, :]

        uid = row[col_uid]
        if uid in fullset:
            for j in range(c):
                if cols[j] not in missset:
                    click_cnt = row[j]
                    try:
                        if click_cnt > 0:
                            pid, param = cols[j]
                            feastr = '{month}:click_param_avg_loan:{pid}_{param}'.format(month=train_next, pid=pid, param=param)
                            buf = transformStatFeatureStr('click_param_avg_loan', feastr)
                            rbuf.extend(buf)
                    except Exception as e:
                        traceback.print_exc()

            uidmap[uid] = ' '.join(rbuf)
    return uidmap


def transformClickStatPid(click_train):
    click_cnt_df = click_train[['uid', 'pid', 'click_time']].groupby(['uid', 'pid']).count().reset_index()

    pdf = pd.pivot_table(click_cnt_df, index='uid', values='click_time', columns='pid', fill_value=0).reset_index()

    df = pdf

    cols = df.columns.values

    r, c = df.shape
    col_uid = df.columns.values.tolist().index('uid')
    uidmap = {}
    missset = ['uid', 'month', 'click_time', 'loan_sum']

    for i in range(r):
        rbuf = []
        row = df.iloc[i, :]

        uid = row[col_uid]
        if uid in fullset:
            for j in range(c):
                if cols[j] not in missset:
                    click_cnt = row[j]
                    if click_cnt > 0:
                        feastr = '{month}:click_avg_loan:{pid}'.format(month=train_next, pid=cols[j])
                        buf = transformStatFeatureStr('click_avg_loan', feastr)
                        rbuf.extend(buf)
            uidmap[uid] = ' '.join(rbuf)
    return uidmap




def transformClickStat():
    logging.info('transformClickStat start')

    uidmap = {}
    if not hitPrefix(['click_param_avg_loan','click_avg_loan'], stat_list):
        return uidmap
    click = pd.read_csv(t_click, header=0)
    click['month'] = click['click_time'].apply(lambda x: x[5:7])
    click_train = click[click['month'].isin(train_month)]


    uidmap1 = transformClickStatPid(click_train)
    uidmap2 = transformClickStatPidParam(click_train)


    for uid in uidmap1:
        if uid in uidmap2:
            uidmap[uid] = '{0} {1}'.format(uidmap1[uid], uidmap2[uid])
        else:
            uidmap[uid] = uidmap1[uid]

    logging.info('transformClickStat end')


    return uidmap





def transform():
    """
    转换成样本
    训练样本和测试样本 关联
    :return:
    """


    user = transformUser()
    user_stat = transformUserStat()

    loan = transformLoan()
    loan_stat = transformLoanStat() #取代哦这个与

    order = transformOrder()
    order_svd = transformSVDOrder()
    order_sum = transformOrderSum()

    order_discount = transformOrderDiscount()
    order_discount_sum = transformOrderDiscountSum()

    order_stat = transformOrderStat()

    click = transformClick()
    click_stat = transformClickStat()


    # order_click = transformOrderClick()  经过测试，用处不大



    mapbuf = [
        user,
        user_stat,
        loan,
        loan_stat,
        order,
        order_svd,
        order_sum,

        order_discount,
        order_discount_sum,

        order_stat ,
        click,
        click_stat,
        # order_click,
    ]

    if sys.argv[1] == 'train':
        loan_sum = pd.read_csv(t_loan_sum, header=0)
        r, c = loan_sum.shape
        col_loan_sum = loan_sum.columns.values.tolist().index('loan_sum')
        col_uid  = loan_sum.columns.values.tolist().index('uid')

        for i in range(r):

            rbuf = []
            row = loan_sum.iloc[i,:]
            target  = row[col_loan_sum]
            uid = row[col_uid]
            for mp in mapbuf:
                val = mp.get(uid,-1)
                if val != -1:
                    rbuf.append(val)


            rbuf = ' '.join(rbuf).split()
            rbuf.sort(key=lambda x: int(x.split(":")[0]))  # sort in place

            yield '{0} {1}'.format(target, ' '.join(rbuf))
    else:

        for uid in fullset:

            rbuf = []
            for mp in mapbuf:
                val = mp.get(uid, -1)
                if val != -1:
                    rbuf.append(val)

            rbuf = ' '.join(rbuf).split()
            rbuf.sort(key=lambda x: int(x.split(":")[0]))  # sort in place

            yield (uid, ' '.join(rbuf))

    pass
def main():

    """
    特征工程的代码，全部转化成libsvm格式再考虑别的
    :return:
    """


    if sys.argv[1] not in ['train','test','train10','test1','test2']:
        raise Exception('choose your mode : train or test ')

    logging.info('initFeatureList begin')
    initFeatureList()
    logging.info('initFeatureList end ')

    logging.info('initStatFeatureList begin')
    initStatFeatureList()
    logging.info('initStatFeatureList end ')

    logging.info('initSuperStatFeatureList begin')
    initSuperFeatureList()
    logging.info('initSuperStatFeatureList end')


    logging.info('addFeature begin')

    addFeature()

    logging.info('addFeature end ')

    alignFeatureID()
    logging.info('alignFeatureId  end ')

    if sys.argv[1] in [ 'train','train10']:
        with open(trainfile, 'w') as f:
            for L in transform():
                print  'line = ' , L
                f.write(L+'\n')
    else:
        with open(testfile,'w') as f1:
            with open(testidfile,'w') as f2:
                for uid, feature in transform():
                    f2.write('{0}\n'.format(uid))
                    f1.write('0 {0}\n'.format(feature))


    logging.info('======================================= done')





if __name__ == '__main__':
    main()
