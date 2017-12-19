# encoding=utf-8
"""
@author : pengalg
"""

import numpy as  np , pickle,subprocess
import sys, datetime,math
import lightgbm as lgb
import pandas as pd
import re
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor,DMatrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
import argparse,logging
import seaborn as sns
from minepy import MINE

from stacking import stacking
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',
                    filemode='a')


LEVEL2_TRAIN = './level2.train'
LEVEL2_LABEL = './level2.train.label'

# with open(LEVEL2_TRAIN, 'r') as f:
#     X_train = pickle.load(f)
# with open(LEVEL2_LABEL, 'r') as f:
#     y_train = pickle.load(f)
#
# with open(LEVEL2_LABEL, 'r') as f:
#     y_train = pickle.load(f)
# class Monitor:
#     def __call__(self, i ,e,  local):
#
#         y_pred = local['y_pred']
#         print 'iter = ' , iter, 'auc = ' , roc_auc_score(y_train,y_pred)
#         return False



train_file  = '/home/mi/zeroplan/sample.txt'


train_file_binary = '/home/mi/zeroplan/sample.binary.txt'
train_file_bigloan_binary = '/home/mi/zeroplan/sample.big.binary.txt'
train_file_superbigloan_binary = '/home/mi/zeroplan/sample.superbig.binary.txt'


train_file_with_binary_score = '/home/mi/zeroplan/sample.binaryscore.txt'

train_file_o = '/home/mi/zeroplan/sample.txt.o'

test_file ='/home/mi/zeroplan/test.txt'
test_file_with_binary_score = '/home/mi/zeroplan/test.binaryscore.txt'
test_file_o='/home/mi/zeroplan/test.txt.o'

testidfile= '/home/mi/zeroplan/test.id.txt'





def addFakeRow(fakefile,file):
    with open(fakefile,'w') as iff:
        a = ['{0}:1'.format(i) for i in range(1, 1000)]
        iff.write('0 '+ ' '.join(a) +'\n')

        with open(file,'r') as f:
            for L in f:
                us = re.split('\\s+', L.strip())
                target = us[0]
                fea = us[1:]
                map = {}
                for i in fea:
                    fid, val = i.split(':')
                    map[int(fid)] = val

                buf = []
                for i in range(1,1000):
                    if i in map:
                        buf.append('{0}:{1}'.format(i,map[i]))
                    else:
                        buf.append('{0}:0'.format(i))

                L  = target +' '+ ' '.join(buf)
                iff.write(L+'\n')

def addFakeRow_(fakefile,file):
    with open(fakefile,'w') as iff:
        a = ['{0}:1'.format(i) for i in range(1, 1000)]
        iff.write('0 '+ ' '.join(a) +'\n')

        with open(file,'r') as f:
            for L in f:
                iff.write(L+'\n')





#
#addFakeRow(train_file_o, train_file)
#addFakeRow(test_file_o, test_file)
#
#train_file = train_file_o
#test_file = test_file_o




# X_vldt, y_vldt = load_svmlight_file(vldt_file)

import os

def get_active_reg_model(infile):
    params = {
        'objective': 'mse',
        'num_leaves': 16,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        # 'reg_alpha': 1, #L1
        # 'reg_lambda': 1, #L2

        'silent': False,
    }
    gbm = lgb.LGBMRegressor(**params)

    X , y = load_svmlight_file(infile,1000)
    X_train, X_test , y_train ,y_test = train_test_split(X, y ,test_size=0.2 ,random_state= 0)

    gbm.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric='mse',early_stopping_rounds=5)

    params['n_estimators'] = gbm.best_iteration_

    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X, y)

    return gbm




def build_active_test(infile, buf):
    outfile = infile + '.active'
    with open(outfile,'w') as of:
        with open(infile,'r') as f:
            i = 0
            for L in f:
                if i in buf:
                    of.write(L+'\n')
                i+=1
    return outfile

def build_sample_active_loan(infile,outfile ,threshold=0):
    logging.info('build_sample_active_loan begin')

    with open(outfile, 'w') as outf:
        with open(infile, 'r') as f:
            for L in f:
                y, fea = re.split('\\s+', L.strip(), maxsplit=1)

                y = float(y)
                if y > threshold:
                    outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_sample_active_loan ok')


def filter_fea(fea, active):
    nodes = re.split('\\s+', fea )

    buf = []
    for node in nodes:
        fid, val = node.split(':')
        fid = int(fid)
        if fid in active:
            buf.append(node)
    return ' '.join(buf)


def build_brute_sample(infile, outfile, active):

    logging.info('build_subset_sample begin')

    with open(outfile,'w') as outf:
        with open(infile,'r') as f:
            for L in f:
                y, fea = re.split('\\s+', L.strip(), maxsplit=1)
                fea = filter_fea(fea, active)
                outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_subset_sample done')

def build_subset_sample(infile, outfile):

    logging.info('build_subset_sample begin')
    rs = np.random.RandomState(0)
    for i in range(10):
        active = rs.randint(1,1000, 100)
        with open(outfile + str(i),'w') as outf:
            with open(infile,'r') as f:
                for L in f:
                    y, fea = re.split('\\s+', L.strip(), maxsplit=1)
                    fea = filter_fea(fea, active)
                    outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_subset_sample done')

def build_sample_fake(infile,outfile,threshold=0):
    logging.info('build_sample_fake begin')
    rs = np.random.RandomState(0)
    with open(outfile,'w') as outf:
        with open(infile,'r') as f:
            for L in f:
                y, fea =  re.split('\\s+', L.strip(), maxsplit=1)

                y = float(y)
                if y <= threshold and np.random.random() < 1.0:
                    y = math.log ( max(0, rs.normal(200, 800, 1)) +1 ,5) 
                outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_sample_fake ok')



def build_sample_binary(infile=train_file,outfile=train_file_binary ,threshold=0):
    logging.info('build_sample_binary begin')

    with open(outfile,'w') as outf:
        with open(infile,'r') as f:
            for L in f:
                y, fea =  re.split('\\s+', L.strip(), maxsplit=1)

                y = float(y)
                if y > threshold:
                    y = 1
                else:
                    y = 0
                outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_sample_binary ok')

def build_sample_with_binary_score_leaf(infile,  outfile,  scorebuf, fid):
    logging.info('build_sample_with_binary_score_leaf begin')
    with open(outfile, 'w') as outf:
        with open(infile, 'r') as f:
            i = 0
            for L in f:
                buf = []
                row = scorebuf[i]
                for j, score in enumerate(row):
                    fea = '{0}:{1}'.format(fid + j , score)
                    buf.append(fea)

                fea = ' '.join(buf)
                outf.write('{0} {1}\n'.format(L.strip(), fea))
                i +=1
    logging.info('build_sample_with_binary_score_leaf ok')

def build_sample_with_leaf_binary_score(infile,  outfile,  scorebuf, leafbuf, fid):
    """

    :param infile:
    :param outfile:
    :param scorebuf:   多个分类器，会补充多个 score进来
    :param fid:
    :return:
    """

    logging.info('build_sample_with_leaf_binary_score begin {0} {1}'.format(infile,outfile))

    with open(outfile, 'w') as outf:
        with open(infile, 'r') as f:
            i = 0
            for L in f:
                buf = []
                for j, score in enumerate(scorebuf):
                    fea = '{0}:{1}'.format(fid + j , score[i][0])
                    buf.append(fea)
                curid = fid + len(scorebuf)
                for leaf in leafbuf:
                    row = leaf[i]
                    for j, score in enumerate(row):
                        fea = '{0}:{1}'.format(curid+j, score)
                        buf.append(fea)
                    curid = curid + len(row)
                fea = ' '.join(buf)
                outf.write('{0} {1}\n'.format(L.strip(), fea))
                i +=1
    logging.info('build_sample_with_leaf_binary_score ok')







def build_sample_with_binary_score(infile,  outfile,  scorebuf, fid):
    """

    :param infile:
    :param outfile:
    :param scorebuf:   多个分类器，会补充多个 score进来
    :param fid:
    :return:
    """

    logging.info('build_sample_with_binary_score begin {0} {1}'.format(infile,outfile))

    with open(outfile, 'w') as outf:
        with open(infile, 'r') as f:
            i = 0
            for L in f:
                buf = []
                for j, score in enumerate(scorebuf):
                    if type(score[i]) == np.ndarray or type(score[i]) == list:
                        l = len(score[i])
                        fea = '{0}:{1}'.format(fid + j , score[i][l-1])
                    else:
                        fea = '{0}:{1}'.format(fid + j, score[i])
                    buf.append(fea)

                fea = ' '.join(buf)
                outf.write('{0} {1}\n'.format(L.strip(), fea))
                i +=1
    logging.info('build_sample_with_binary_score ok')

def get_files(path):
    v = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "part-split" in file:
                v.append(root + '/' + file)
    return v
def predict(cls, threshold=0.):
    uid = pd.read_csv(testidfile,header=None)
    uid = uid[0].values
    with open('out.csv', 'w') as f:
        X_test, y_test = load_svmlight_file(test_file, 1000)

        score = cls.predict(X_test)
        print 'score size = ' , len(score) , ' uid size = ' , len(uid)
        lenscore = len(score)
        lenuid = len(uid)
        for i, v in enumerate(uid):
            if lenscore > lenuid:
                k = i +1
            else:
                k = i
            if score[k] > threshold: # 这种处理分数能影响多少呢?
                cur = score[k]
            else:
                cur = 0.001
            f.write('{0},{1}\n'.format(int(v), cur))
    subprocess.check_call('cat out.csv | sort -t, -k1 -g >out.sorted.csv', shell=True)
    subprocess.check_call('cat out.csv >out/out.{0}.csv'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M')), shell=True)



def lgb_cls_then_reg():

    ##param

    ACTIVE_AMOUNT = 0
    POS_THRESHOLD = 0.4

    train_file_active = train_file + '.active'


    build_sample_binary(train_file, train_file_binary, math.log(ACTIVE_AMOUNT +1 , 5))
    # =====================================================
    binary_file = train_file_binary
    params = {
        'objective': 'binary',
        'num_leaves': 16,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'silent': False,
    }
    X, y = load_svmlight_file(binary_file, 1000)



    gbm = lgb.LGBMClassifier(**params)

    X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)
    gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric=['auc'], early_stopping_rounds=5)

    # models = [gbm]
    # train_score, test_score = stacking(models, X, y ,X_test, regression=True, metric=roc_auc_score, n_folds=3, shuffle=False,random_state=0, verbose=2)
    # train_score = gbm.predict_proba(X)

    X_test, y_test = load_svmlight_file(test_file, 1000)
    test_score = gbm.predict_proba(X_test)
    # =====================================================
    buf, silent_buf = [], []
    i = 0
    for neg, pos in test_score:
        if pos > POS_THRESHOLD:
            buf.append(i)
        else:
            silent_buf.append(i)
        i +=1

    print 'active test = ' , len(buf)

    logging.info(' active test = {0}'.format(len(buf)))


    test_file_active = build_active_test(test_file, buf)
    X_test, y_test = load_svmlight_file(test_file_active, 1000)
    #get the regression model
    build_sample_active_loan(train_file, train_file_active, 0)
    reggbm = get_active_reg_model(train_file_active)
    score = reggbm.predict(X_test)

    pbuf = []
    for i, v  in enumerate(buf):
        pbuf.append( (v , score[i] ))
    for i, v in enumerate(silent_buf):
        pbuf.append( ( v, 0))

    pbuf = sorted(pbuf, key=lambda  x : x[0])

    with open('out.csv','w') as f:
        for uid,score in pbuf:
            uid = int(uid) + 1
            f.write('{0},{1}\n'.format(uid, score))

    logging.info('cls_then_reg done')






def lgbcv():
    """
    ('Best parameters found by grid search are:', {'num_leaves': 16, 'learning_rate': 0.05})
    """
    active = set(range(1, 1001))
    active = active.difference(range(155,156))
    build_brute_sample(train_file, train_file + '.brute', active)
    X, y = load_svmlight_file(train_file + '.brute', 1000)
    # X, y = load_svmlight_file(train_file, 1000)
    X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)


    param_grid = {
        'learning_rate': [ 0.02, 0.05, 0.1],
        'num_leaves': [8,16],
    }

    estimator =  lgb.LGBMRegressor(objective='mse',n_estimators=200)
    gbm = GridSearchCV(estimator, param_grid,verbose=2,scoring='neg_mean_squared_error',cv=5)
    params = {
        'eval_set': [(X_vldt, y_vldt)],
        'eval_metric': 'mse',
        'early_stopping_rounds': 5,
    }
    gbm.fit(X_train, y_train, **params)
    print('Best parameters found by grid search are:', gbm.best_params_)
    print('Best score:', gbm.best_score_)
    print('Best score:', math.sqrt(abs(gbm.best_score_)))

    logging.info('cv Best score: {0}'.format(math.sqrt(abs(gbm.best_score_))))

    logging.info('Best parameters found by grid search are: {0}'.format(gbm.best_params_))

    print ('cv result ', gbm.cv_results_)
class MagicMSE():
    # def __init__(self,x,y):
    #     pass
    def __call__(self, y_true,y_pred):
        return self.magic_mse(y_true, y_pred)
    def magic_mse(self, y_true,y_pred):
        o = y_pred #type:  np.array
        buf = []
        for i in y_pred:
            if i < 2:
                i = 0
            buf.append(i)
        y = np.array(buf)
        return np.mean( (y_true  - y )**2 )

loan_confidence = pd.read_csv('loan_confidence.csv',header=0)
confi_map = {}

magic_buf ,mse_buf = [], []
for row in loan_confidence.values:
    uid = int(row[0])
    mean = float(row[1])
    std = float(row[2])
    confi_map[uid] = (mean,std)

from scipy.stats import norm
def confidence(y ,mean ,std):
    """
    Normal Distribution to check confidence
    :param y:
    :param mean:
    :param std:
    :return:
    """
    if std == 0 : std = 1.

    z = ( y - mean )/std

    if z >= 0 :
        v = (1 - norm.cdf(z)) *2
    else:
        v = norm.cdf(z) * 2
    return v
def     magic_mse(y_true,y_pred):
    # raise Exception('forbiddent the magic_mse')
    """
    引入 可信度的概念， 凡是偏离 历史均值太远的，可信度降低

    尝试1：  y_pred < 1. 则设置为 0 ， 这种反而造成了 mse上升； 说明有些 真实值预测的很低 []
    尝试2：  假设 loan_sum ~  N ( hat_loan_sum ,sigma )  , 看看实际的 loan_sum 在多少的范围内,偏离越远，越不可信
            最终的loss 改成    c_i(pi - ri)^2 ,c_i 就用来度量 偏离度

            这种设计，会使得模型尽量忽略突然出现的用户,从而把有把握的用户预测得更准

            实际效果需要验证

    :param y_true:
    :param y_pred:
    :return:
    """
    # o = y_pred #type:  np.array
    # buf = []
    # for i in y_pred:
    #     if i < 1.:view
    #         i = 0
    #     buf.append(i)
    # y = np.array(buf)
    # mse =  np.mean( (y_true  - y )**2 )


    confi = []
    for i, score  in enumerate(y_pred):
        uid = int(vldt_uids[i])
        if uid in confi_map:
            mean , std = confi_map[uid]

            # print 'hit ' , uid  , mean ,std
        else:
            mean ,std = 0 ,1

        y_confidence = confidence( y_true[i] , mean ,std)
        confi.append(y_confidence)


    y_confidence = np.array(confi)

    magic_v = np.mean ( y_confidence*(y_true - y_pred)**2)

    # (eval_name, eval_result, is_bigger_better)
    mse = np.mean( ( y_true - y_pred)**2 )
    print '\t\t\t mse = {0} , rmse = {1} '.format(mse , math.sqrt(mse))

    magic_buf.append(magic_v )
    mse_buf.append(mse)
    # return  ('l2', np.mean( (y_true - y_pred) ** 2 ) , False)
    return  ('l2', magic_v , False)

def lgbclassifier_withleaf(mode):
    """
    先做分类，当成一个特征
    :param mode:
    :return:
    """
    build_sample_binary(train_file,train_file_binary, math.log(0+1,5) )

    trainbuf ,testbuf = [], []
    for binary_file in  [train_file_binary]:
        logging.info('training {0}'.format(binary_file))
        params = {
            'objective': 'binary',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'silent': False,
        }
        X, y = load_svmlight_file(binary_file, 1000)
        r, c  = X.shape
        X_train , X_vldt, y_train , y_vldt = train_test_split(X,y, test_size=0.2,random_state=0)

        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric=['auc'], early_stopping_rounds=5)


        train_score = gbm.predict_proba(X)

        X_test, y_test = load_svmlight_file(test_file, 1000)
        test_score = gbm.predict_proba(X_test)

        trainbuf.append(train_score)
        testbuf.append(test_score)

        ##leaf  
        train_leaf = gbm.apply(X)
        test_leaf = gbm.apply(X_test)


    #build_sample_with_binary_score(train_file, train_file_with_binary_score, trainbuf,800 )
    #build_sample_with_binary_score(test_file, test_file_with_binary_score, testbuf,800 )
    build_sample_with_binary_score_leaf(train_file, train_file_with_binary_score, train_leaf,800 )
    build_sample_with_binary_score_leaf(test_file, test_file_with_binary_score, test_leaf,800 )

def build_sample_count_binary(infile,outfile ,threshold):
    logging.info('build_sample_count_binary begin')
    t_loan_cnt = './11.csv.cnt'
    loan_cnt = pd.read_csv(t_loan_cnt,header=0)
    with open(outfile,'w') as outf:
        with open(infile,'r') as f:
            i = 0
            for L in f:
                cnt = loan_cnt['loan_cnt'].values[i]
                i += 1
                y, fea =  re.split('\\s+', L.strip(), maxsplit=1)


                if cnt > threshold:
                    y = 1
                else:
                    y = 0
                outf.write('{0} {1}\n'.format(y, fea))
    logging.info('build_sample_count_binary ok')

def lgbclassifier(mode):
    """
    先做分类，当成一个特征
    :param mode:
    :return:
    """
    #
    # build_sample_count_binary(train_file, train_file + '.cnt_1' , 1)
    # build_sample_count_binary(train_file,train_file + '.cnt_2', 2)
    # build_sample_count_binary(train_file, train_file + '.cnt_3', 3)
    # build_sample_count_binary(train_file, train_file + '.cnt_4', 4)
    # build_sample_count_binary(train_file, train_file + '.cnt_5', 5)


    build_sample_binary(train_file,train_file_binary, 0 )
    build_sample_binary(train_file,train_file_bigloan_binary, math.log(2000 +1 , 5) )
    build_sample_binary(train_file,train_file_bigloan_binary + '_5000', math.log(5000 +1 , 5) )
    build_sample_binary(train_file,train_file_bigloan_binary + '_10000', math.log(10000 +1 , 5) )
    build_sample_binary(train_file,train_file_bigloan_binary + '_15000', math.log(15000 +1 , 5) )
    build_sample_binary(train_file,train_file_bigloan_binary + '_1', math.log(20000 +1 , 5) )
    build_sample_binary(train_file,train_file_bigloan_binary + '_25000', math.log(25000 +1 , 5) )
    build_sample_binary(train_file, train_file_bigloan_binary + '_2', math.log(30000 + 1, 5))
    build_sample_binary(train_file, train_file_bigloan_binary + '_35000', math.log(35000 + 1, 5))
    build_sample_binary(train_file,train_file_superbigloan_binary , math.log(50000 +1 , 5) )

    trainbuf ,testbuf = [], []
    for binary_file in  [
                    #     train_file + '.cnt_1' ,
                    # train_file + '.cnt_2',
                    # train_file + '.cnt_3',
                    # train_file + '.cnt_4',
                    # train_file + '.cnt_5',

                         train_file_binary,
                         train_file_bigloan_binary, #2000
                         train_file_bigloan_binary+'_5000',
                         train_file_bigloan_binary+'_10000',
                         train_file_bigloan_binary+'_15000',
                         train_file_bigloan_binary+'_25000',
                         train_file_bigloan_binary+'_35000',
                         train_file_bigloan_binary+'_1',#20000
                         train_file_bigloan_binary+'_2', # 30000
                         train_file_superbigloan_binary #50000
                         ]:
        logging.info('training {0}'.format(binary_file))
        params = {
            'objective': 'binary',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'n_estimators': 150,
            'silent': False,
        }
        X, y = load_svmlight_file(binary_file, 1000)
        r, c  = X.shape


        X_test, y_test = load_svmlight_file(test_file, 1000)
        gbm = lgb.LGBMClassifier(**params)

        X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)
        gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric=['auc'], early_stopping_rounds=5)

        # models = [gbm]
        # train_score, test_score = stacking(models, X, y ,X_test, regression=True, metric=roc_auc_score, n_folds=3, shuffle=False,random_state=0, verbose=2)

        train_score = gbm.predict_proba(X)
        test_score = gbm.predict_proba(X_test)

        trainbuf.append(train_score)
        testbuf.append(test_score)

    #build_sample_with_binary_score(train_file, train_file_with_binary_score, trainbuf,800 )
    #build_sample_with_binary_score(test_file, test_file_with_binary_score, testbuf,800)
    #return

    train_leafbuf, test_leafbuf = [], []
    """
    经过实验：添加更多的 文件 并没有提升成绩.
    """
    for binary_file in  [train_file_binary, train_file_bigloan_binary + '_5000', train_file_bigloan_binary+'_15000']:
        logging.info('training {0}'.format(binary_file))
        params = {
            'objective': 'binary',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'silent': False,
        }
        X, y = load_svmlight_file(binary_file, 1000)
        r, c  = X.shape
        X_train , X_vldt, y_train , y_vldt = train_test_split(X,y, test_size=0.2,random_state=0)

        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric=['auc'], early_stopping_rounds=5)


#        train_score = gbm.predict_proba(X)

        X_test, y_test = load_svmlight_file(test_file, 1000)
#        test_score = gbm.predict_proba(X_test)


        ##leaf  
        train_leaf = gbm.apply(X)
        test_leaf = gbm.apply(X_test) 

        train_leafbuf.append(train_leaf)
        test_leafbuf.append(test_leaf)



    build_sample_with_leaf_binary_score(train_file, train_file_with_binary_score, trainbuf,train_leafbuf,800 )
    build_sample_with_leaf_binary_score(test_file, test_file_with_binary_score, testbuf,test_leafbuf,800 )



def silent_rule_check(cls, X_vldt,y_vldt):
    """
    经过实验，这种check，无法帮助提升mse
    :param cls:
    :param X_vldt:
    :param y_vldt:
    :return:
    """
    return
    score = cls.predict(X_vldt)
    X = np.reshape(X_vldt.toarray(), (-1, 1000))
    buf = []
    i  = 0

    silent = 0
    for row in X:
        cnt = row[530]

        print row[0], row[1], row[530],row[531],row[532]
        s = score[i]
        if cnt == 0:
            s = 0
            silent += 1
        buf.append(s)
        i+=1

    buf = np.array(buf)
    """
     3.20256684249 , after rule = 3.30013346512, silent = 11673, total = 18199
    """
    info = ' {0} , after rule = {1}, silent = {2}, total = {3}'.format(
        mean_squared_error(y_vldt,score) ,
        mean_squared_error(y_vldt,buf), silent, X_vldt.shape[0])
    print info
    logging.info(info)

def lgbbrute_predict(args):
    """
    find best parameter to check.
    :param args:
    :return:
    """
    active = set(range(1, 1001))
    active = active.difference([350, 389])
    build_brute_sample(train_file, train_file + '.brute', active)

    X, y = load_svmlight_file(train_file + '.brute', 1000)
    X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)

    params = {
        'objective': 'mse',
        'num_leaves': 18,
        'learning_rate': 0.11,
        'min_child_samples': 80,
        'subsample': 0.9,
        'n_estimators': 1000,
        # 'reg_alpha': 1, #L1
        # 'reg_lambda': 1, #L2

        'silent': False,
    }

    gbm = lgb.LGBMRegressor(**params)

    gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric='mse', early_stopping_rounds=5)
    logging.info('best_score = {0}'.format(math.sqrt(gbm.best_score_['valid_0']['l2'])))
    params['n_estimators'] = gbm.best_iteration_
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X, y)
    predict(gbm)

    logging.info('brute_predict done')




def lgbbrute_param(args):

    """
    18 0.11  是最高好  =1.78296912453
    :param args:
    :return:
    """
    # active = set(range(1, 1001))
    # active = active.difference([350, 389])
    # build_brute_sample(train_file, train_file + '.brute', active)

#    X, y = load_svmlight_file(train_file + '.amount', 1000)
    X, y = load_svmlight_file(train_file, 1000)
    X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)
#
#    num_leaves = range(8,  65 , 8 )
#    learning_rates = np.linspace(0.01, 0.2, 20 )
#
#    num_leaves = range(16, 24, 1)
#    learning_rates = np.linspace(0.01, 0.2, 20 )
#
    num_leaves =  [16, 18]
    learning_rates = [0.05, 0.11,0.12]
    min_child_samples = [20, 80, 85,90,95,100]  #best
    subsamples = [0.8, 0.9, 1.0]  # 1.7825

#    num_leaves =  [16]
#    learning_rates = [0.11]
#    min_child_samples = [60,80,82,84]  #best
#    subsamples = [0.8]  # 1.7825
    # reg_alpha  = range(0,10)
    # reg_lambda = range(0,10)

    # min_child_samples =  [80]
    # subsamples = [0.8 , 0.9 ,1.0 ]
    params = {
        'objective': 'mse',
        'num_leaves': 16,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        # 'reg_alpha': 1, #L1
        # 'reg_lambda': 1, #L2

        'silent': False,
    }
    #  hit 16 0.11 90 0.8  1.78215112832
    min_score = 1000
    buf = []
    for num_leave_ in num_leaves:
        for learning_rate_ in learning_rates:
            for min_s in min_child_samples:
                for subs in subsamples:
                    pstr = '{0} {1} {2} {3}'.format(num_leave_, learning_rate_ , min_s, subs)
            # for reg_a in reg_alpha:
            #     for reg_l in reg_lambda:

                    params['num_leaves'] = num_leave_
                    params['learning_rate'] = learning_rate_
                    params['min_child_samples'] = min_s

                    params['subsample'] = subs
                    # params['reg_alpha'] = reg_a
                    # params['reg_lambda'] = reg_l

                    num_leave , learning_rate = num_leave_, learning_rate_
                    gbm = lgb.LGBMRegressor(**params)

                    gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric='mse', early_stopping_rounds=5)
                    best_score = gbm.best_score_['valid_0']['l2']


                    if min_score > best_score:
                        min_score = best_score

                    buf.append((math.sqrt(best_score), num_leave, learning_rate))
                    if math.sqrt(best_score) < 1.783:
                        logging.info('check brute param , hit {0} {1} {2} {3}  {4}'.format(num_leave, learning_rate, min_s, subs, math.sqrt(best_score) ))
                    logging.info('check brute param ,{pstr} best_score = {bs} , min_score ={mins}'.format(pstr=pstr, bs= math.sqrt(best_score),mins= math.sqrt(min_score)))

    scores = [ score for score, num_leave, learning_rate in buf]
    leaves  = [ num_leave for score, num_leave, learning_rate in buf]
    learning_rates = [learning_rate for score, num_leave, learning_rate in buf]


    pd.DataFrame({'learning_rate': learning_rates, 'leave': leaves, 'score': scores}).to_csv('brute.param.csv', index=False)
    logging.info('brute_param done')
def lgbbrute(args):
    """
    强行删除 一个 特征， 查看是否提高.
    :return:

    step 1:  remove 372 is better. 1.7846 ,maybe 1.777x ??
    step 3:  1.7841 xx 加入loan_sum_back  remove 350 loan_sum_previous.

    step 4:  389 stat_lookahead2_repay_pressure   1.78355148914

    milestone:  350 389  ; 1.783 .
    """
    check_range = range(329,422)
    # check_range = [389]
    buf = []
    min_score = 1000
    for i in check_range:
        active = set(range(1,1001))
        active = active.difference([350,389,  i])
        build_brute_sample(train_file, train_file + '.brute' , active)

        logging.info('check brute {0}'.format(i))
        params = {
            'objective': 'mse',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            # 'reg_alpha': 1, #L1
            # 'reg_lambda': 1, #L2

            'silent': False,
        }
        gbm = lgb.LGBMRegressor(**params)
        X, y  = load_svmlight_file(train_file +'.brute', 1000)
        X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)
        gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric='mse', early_stopping_rounds=5)
        best_score = gbm.best_score_['valid_0']['l2']
        if min_score > best_score:
            min_score = best_score


        buf.append((math.sqrt(best_score), i))
        if math.sqrt(best_score) <  1.78534412772:
            logging.info('check brute , hit {0}'.format(i))
        logging.info('check brute , {0} , best_score = {1},{2} , min_score ={3}'.format(i, best_score,math.sqrt(best_score), math.sqrt(min_score)))
        if args.predict == True:
            logging.info('check brute, predict..')
            ##predict
            params['n_estimators'] = gbm.best_iteration_
            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(X, y)
            predict(gbm)
            logging.info('predict ok')

    fids = [ i  for score ,i in  buf ]
    scores = [score for score, i in buf]

    pd.DataFrame({'id': fids , 'score': scores}).to_csv('brute.csv',index=False)



def lgbmulreg():
    """
    利用不同的特征子集 进行训练
    :return:
    """

    train_file = '/home/mi/zeroplan/backup/20171205-1711/sample.txt'
    test_file = '/home/mi/zeroplan/backup/20171205-1711/test.txt'

    # build_subset_sample(train_file, train_file + '_mulreg')

    gbmbuf = []
    for i in range(10):
        target = train_file + '_mulreg' + str(i)
        X, y = load_svmlight_file(target,1000)
        X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)

        params = {
            'objective': 'mse',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            # 'reg_alpha': 1, #L1
            # 'reg_lambda': 1, #L2

            'silent': False,
        }
        gbm = lgb.LGBMRegressor(**params)
        gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric='mse', early_stopping_rounds=5)
        gbmbuf.append(gbm)

    X , y = load_svmlight_file(test_file, 1000)

    df = pd.DataFrame()
    for i, gbm in enumerate(gbmbuf):
        score = gbm.predict(X)
        col = 'm{0}'.format(i)
        df[col] = score

    df['final'] = (df['m0'] + df['m4'])/2

    with open('mulreg.out.csv','w') as f:
        dvals = df['final'].values
        for i ,val in enumerate(dvals):
            f.write('{0},{1}\n'.format(i+1, val))

    # df.corr()

    cm = []
    for i in range(10):
        tmp = []
        for j in range(10):
            m = MINE()
            coli = 'm{0}'.format(i)
            colj = 'm{0}'.format(j)

            m.compute_score(df[coli].values, df[colj].values)
            tmp.append(m.mic())
        cm.append(tmp)

    colormap = plt.cm.viridis
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
    #             annot=True)

    sns.heatmap(cm, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)

    plt.show()


def lgbmulreg_old():
    """
    废弃
    :return:
    """
    params = {
        'objective': 'mse',
        'num_leaves': 16,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        # 'reg_alpha': 1, #L1
        # 'reg_lambda': 1, #L2

        'silent': False,
    }
    X , y = load_svmlight_file(train_file,1000)
    X_train, X_vldt, y_train, y_vldt= train_test_split(X,y, test_size=0.2,random_state=0)
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X_train, y_train, eval_set=[(X_vldt, y_vldt)], eval_metric='mse', early_stopping_rounds=5)

    print math.sqrt(gbm.best_score_['valid_0']['l2'])
    best_score = gbm.best_score_['valid_0']['l2']
    logging.info('best_iter = {0}  ,best_score = {1} , {2} , {3} '.format(gbm.best_iteration_, best_score, math.sqrt(best_score), params))

    testfile1 = test_file +'.1'
    testfile2 = test_file +'.2'
    sc = []
    for tf in [test_file, testfile1,testfile2]:
        X_test ,y_test = load_svmlight_file(tf, 1000)
        score = gbm.predict(X_test)
        sc.append(score)


    df = pd.DataFrame({'t1': sc[0], 't2': sc[1] , 't3': sc[2]})

    df['score'] =  0.7 * df['t1'] + 0.2 * df['t2'] + 0.1 * df['t3']

    scores = df['score'].values
    with open('out.csv','w') as f:
        for i in range(len(y_test)):
            f.write('{0},{1}\n'.format(i+1, scores[i]))
    logging.info('lgbmulreg done')


def lgbfakereg():
    """
    fake sample and reg
    """
    global train_file
    fakefile= train_file + '.fake'
    build_sample_fake(train_file, fakefile, 0)
    train_file = fakefile 
    lgbmain('train')
    
def lgbmain(args):

    """
    loss: fair,huber, mse
    三个loss 都比较接近，mse 验证效果最好
    :return:
    16  [99]	valid_0's l2: 3.27457
    8  [109]	valid_0's l2: 3.31043  [132]	valid_0's l2: 3.27417
    """
    mode = args.mode

    #active = set(range(1, 1001))
    #active = active.difference(range(329,333)) # remove it .
    #build_brute_sample(train_file, train_file + '.brute', active)
    #X, y = load_svmlight_file(train_file + '.brute', 1000)
    # X, y = load_svmlight_file(train_file, 1000)
    #X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        'objective': 'mse',
        'num_leaves': 16,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        #'reg_alpha': 1, #L1
        #'reg_lambda': 1, #L2

        'silent':False,
    }
    params = {
        'objective': 'mse',
        'num_leaves': 18,
        'learning_rate': 0.05,
        'min_child_samples': 80,
        'subsample': 0.9 ,
        'n_estimators': 1000,
        # 'reg_alpha': 1, #L1
        # 'reg_lambda': 1, #L2
    #
        'silent': False,
    }
    print params
    gbm = lgb.LGBMRegressor(**params)
    lgbparam_rf = {
         'objective': 'mse',
         'num_leaves': 8,
         'boosting_type': 'rf',
         'bagging_fraction': 0.8 , #necessary
         'feature_fraction': 0.8,  #necessary
         'n_estimators': 200,
    }
    #gbm = lgb.LGBMRegressor(**lgbparam_rf)
    if args.metric == 'mse':
        gbm.fit(X_train,y_train,eval_set = [(X_vldt,y_vldt)],eval_metric= 'mse' ,early_stopping_rounds = 5)
    elif args.metric == 'magic':
        gbm.fit(X_train,y_train,eval_set = [(X_vldt,y_vldt)],eval_metric= magic_mse ,early_stopping_rounds = 5)
    else:
        raise 'check your metric'
    print math.sqrt(gbm.best_score_['valid_0']['l2'])

    best_score = gbm.best_score_['valid_0']['l2']

    logging.info('best_iter = {0}  ,best_score = {1} , {2} , {3} '.format(gbm.best_iteration_, best_score, math.sqrt(best_score), params))

    #检验一条rule 是否能提升mse
    silent_rule_check(gbm,X_vldt,y_vldt)
    # y_pred = gbm.predict(X_vldt) # probability
    # LGBM预测的时候可能出现负数
    if mode == 'retrain':
        logging.info('retrain begin')
        params['n_estimators'] = gbm.best_iteration_
        gbm = lgb.LGBMRegressor(**params)
        gbm.fit(X,y)
        y_pred = gbm.predict(X)
        pd.DataFrame({'y': y , 'y_pred': y_pred}).to_csv('y_pred.csv', index=False)
        logging.info('retrain done')
    predict(gbm)
    #plt.plot(gbm.feature_importance())

    logging.info('feature importance len = {0}'.format( len(gbm.feature_importance())))
    lenfea = len(gbm.feature_importance())
    dfimp = pd.DataFrame(
        {'id': [i+1 for i in range(lenfea)],

        'importance': gbm.feature_importance()
         }
    )
    dfname = pd.read_csv('idname.txt',header=0)

    pd.merge(dfimp, dfname,how='left', on='id').to_csv('feaimp.csv',index=False)
    #plt.xticks([i for i in range(0, 1000, 10)],rotation=80)
    #plt.show()
    # for i in y_pred:
    #     print i
def xgbmain():
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    learning_rate = float(sys.argv[3])

    cls = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                       n_estimators=n_estimators, silent=False, objective='reg:linear',
                       subsample=0.8)


    print >>sys.stderr, 'begin to fit '
    if n_estimators == 100:
        cls.fit(X_train, y_train,eval_set=[(X_vldt,y_vldt)],eval_metric='rmse',early_stopping_rounds=3)
    else:
        cls.fit(X_train, y_train)




    predict(cls)
    subprocess.check_call('cp out.sorted.csv out.sorted.{0}-{1}-{2}.csv'.format(n_estimators, max_depth, learning_rate),
                           shell=True)

def main():


    #
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    learning_rate = float(sys.argv[3])


    # n_estimators = 100
    # max_depth = 3
    # learning_rate = 0.3

    cls = GradientBoostingClassifier(n_estimators= n_estimators, max_depth= max_depth, loss='deviance',  learning_rate= learning_rate ,
                                     min_samples_split = 10,
                                     min_samples_leaf= 10,
                                     subsample=0.8,verbose=2)
    cls.fit(X_train,y_train, monitor=Monitor(X_vldt,y_vldt))
    # cls.fit(X_train,y_train)



    print 'vldt auc = ' , roc_auc_score(y_vldt,  cls.predict_proba(X_vldt)[:,1] ),  ', vars = ' , cls.get_params()

    with open('out.csv','w') as f:
        f.write('instance_id,prob\n')

        tests = get_files(test_file)
        for t in tests:
            print t
            X_test , y_test = load_svmlight_file(t)
            proba = cls.predict_proba(X_test)[:,1]

            for i ,v in enumerate(y_test):
                f.write('{0},{1}\n'.format(int(v), proba[i]))
    subprocess.check_call('cat out.csv | sort -t, -k1 -n >out.sorted.csv',shell=True)

    subprocess.check_call('cp out.sorted.csv out.sorted.{0}-{1}-{2}.csv'.format(n_estimators, max_depth, learning_rate),shell=True)

def train_test_split_with_uids ( X, y, test_size, random_state):
    X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=test_size, random_state=random_state)

    t_loan_sum = '/home/mi/zeroplan/t_loan_sum2.csv'  # 11.csv
    train = pd.read_csv(t_loan_sum, header=0)
    uids = train['uid']

    assert len(uids) == len(y)

    n_test = int(np.ceil( len(y) * test_size))

    rng = np.random.RandomState(seed = random_state)

    puids = rng.permutation(uids)

    vldt_uids = puids[:n_test]


    return X_train, X_vldt, y_train, y_vldt , vldt_uids
if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='retrain')
    parser.add_argument('--metric', default='mse')
    parser.add_argument('--predict', type=bool, default=False)
    args = parser.parse_args()
    # xgbmain()
    if args.mode == 'brute_predict':
        lgbbrute_predict(args)
    elif args.mode == 'brute_check':
        lgbbrute(args)
    elif args.mode == 'brute_param':
        lgbbrute_param(args)
    elif args.mode == 'mulreg':
        lgbmulreg()
    elif args.mode == 'fake_reg':
        lgbfakereg()
    elif args.mode ==  'cls_then_reg':
        """
        先分类，然后再回归
        """
        lgb_cls_then_reg()

    elif args.mode in [ 'train','retrain']:
        ############ 如果是binary
        # train_file = train_file_with_binary_score + '.nn'
        # test_file = test_file_with_binary_score  + '.nn'
        #
        # train_file = train_file_with_binary_score
        # test_file = test_file_with_binary_score


        #{train_file = train_file + '.amount'
        #test_file= test_file+ '.amount'
        X, y = load_svmlight_file(train_file, 1000)
        X_train, X_vldt, y_train, y_vldt, vldt_uids = train_test_split_with_uids(X, y, test_size=0.2, random_state=0)
        lgbmain(args)

        pd.DataFrame({'magic': magic_buf, 'mse': mse_buf }).to_csv('magic.mse',index=False)
    elif args.mode =='cv':
        # train_file = train_file_with_binary_score + '.nn'
        # test_file = test_file_with_binary_score + '.nn'

        # train_file = train_file_with_binary_score
        # test_file = test_file_with_binary_score
        train_file = train_file + '.stack'
        test_file= test_file+ '.stack'
        X, y = load_svmlight_file(train_file, 1500)
        X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=0)

        X_train , y_train = X, y
        lgbcv()
    elif args.mode == 'cls':
        #X, y = load_svmlight_file(train_file, 1000)
        #X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=1)
        lgbclassifier(args.mode)
    elif args.mode == 'cls_leaf':
        X, y = load_svmlight_file(train_file, 1000)
        X_train, X_vldt, y_train, y_vldt = train_test_split(X, y, test_size=0.2, random_state=1)
        lgbclassifier_withleaf(args.mode)
    else:
        raise Exception("check your mode")
