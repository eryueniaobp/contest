# encoding=utf-8
"""
@author : X
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import numpy as np
import random
import re,datetime, subprocess,os

import logging
import lightgbm as lgb
import gc
import pickle
from collections import defaultdict
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',
                    filemode='a')

#自定义 load_svmlight_file,方便应对大数据量，单机无法运行的情况 
def _load_svmlight_file(path, n_feature , AID_CNT = 30000, mini_batch=200000, frac=1.0):

    data = []
    indices = []
    indptr = [0]
    query = []
    aidbuf  =[]

    labels = []


    qid_prefix='qid'
    COLON = ':'

    lnum = 0
    cntmap = defaultdict(int)
    logset = set() 

    csr_lnum = 0
    with open(path , 'r') as f:
        for line in f:
            # skip comments
            line_parts = line.split()
            if len(line_parts) == 0:
                continue

            try:


                target, qid, aid, features = line_parts[0], line_parts[1], line_parts[2], line_parts[3:]
            except Exception as e:
                print line
                raw_input('\t\t')
            if random.random() > frac: continue
            _, aid_value = aid.split(COLON, 1)
            if cntmap[aid_value] >= AID_CNT:
                if aid_value not in logset:
                    logging.info('{} already {} save memorty'.format(aid_value, AID_CNT))
                    logset.add(aid_value)
                continue
            aidbuf.append(int(aid_value))
        
            cntmap[aid_value] += 1
            labels.append (  float(target) )

            prev_idx = -1
            n_features = len(features)
            if n_features and features[0].startswith(qid_prefix):
                _, value = features[0].split(COLON, 1)


                query.append( int(value) )
                features.pop(0)
                n_features -= 1

            _, value = qid.split(COLON, 1)
            query.append(int(value))




            for i in xrange(0, n_features):
                idx_s, value = features[i].split(COLON, 1)
                idx = int(idx_s)

                if idx <= prev_idx:
                    raise ValueError("Feature indices in SVMlight/LibSVM data "

                                     "file should be sorted and unique.")
                if idx > n_feature:
                    break

                indices.append(idx)

                data.append(float(value))

                prev_idx = idx
            indptr.append(len(data))
            lnum +=1
            if lnum % 500000 == 0: logging.info('line = {}'.format(lnum))
            if lnum >= mini_batch :

                shape =(lnum, n_feature)

                X = sp.csr_matrix((data, indices, indptr), shape)
                X.sort_indices()

                yield ( X , np.array(labels), query, aidbuf)
                # clear data
                lnum = 0

                data = []
                indices = []
                indptr = [0]
                query = []
                labels = []
                aidbuf = []
    if lnum > 0:
        #尾巴上的数据共享.
        shape = (lnum, n_feature)

        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()

        yield (X, np.array(labels), query, aidbuf)

        lnum = 0

        data = []
        indices = []
        indptr = [0]
        query = []
        labels = []
        aidbuf = []

def train(NF=None, MIL=10000*10000,TEST_BATCH=300*10000,frac=1.0):
    train_path = 'train.libsvm'
    test_path = 'test2.libsvm'
    test_csv = 'test2.csv'
    logging.info('load train {} {} {} {} '.format(train_path, NF, MIL, TEST_BATCH))

    if NF is None:
        NF =  2674644 + 10
        NF = 100*10000

    #MIL = 10000 * 10000
    #TEST_BATCH = 300*10000

    #MIL = 1 * 10000 
    #TEST_BATCH = 1 * 10000
    X, y, uids, aids  = next(_load_svmlight_file(train_path, n_feature=NF, AID_CNT=100000, mini_batch=MIL, frac=frac))

    logging.info('size =  {}'.format(len(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    del X, y
    gc.collect()
    logging.info('train test split ok')
    estimator = lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.1, num_leaves=31,subsample=0.8 ,colsample_bytree=0.8, silent=False)
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc", early_stopping_rounds=20)
    cur_auc = estimator.evals_result_['valid_0']['auc'][-1]
    logging.info('cur_auc = {}'.format(cur_auc))
    try:
        joblib.dump(estimator, '{}.{}.model'.format(NF,frac), protocol = 0)
    except Exception as e:
        print "joblib dump model fail"
    del X_train, X_test
    gc.collect()
    logging.info('load {}'.format(test_path))

    X, y, uids , aids = next(_load_svmlight_file(test_path, n_feature=NF, AID_CNT=100*10000,mini_batch=TEST_BATCH, frac=1.0))


    logging.info('size {} begin to predict'.format(len(y)))
    score = estimator.predict_proba(X)[:, 1]

    day = datetime.datetime.now().strftime('%Y%m%d')
    df = pd.DataFrame({'aid': aids ,'uid': uids, 'score': score})
    # df = pd.read_csv('test.aux.csv', sep=' ', header=None, nrows=len(y))
    # df.columns = ['label', 'aid', 'uid']
    # df['score'] = score

    df.to_csv('{}.score.csv'.format(day))
    test = pd.read_csv(test_csv)
    df = pd.merge(test, df, on=['aid', 'uid'])
    df[['aid', 'uid', 'score']].to_csv('{}.{}.{}.submit.csv'.format(day, NF,frac),index=False, float_format='%.4f')
    logging.info('train done')

if __name__ == '__main__':
    train(NF=100*10000, MIL=10000,TEST_BATCH=10000, frac=1.0)
    #train(NF=100*10000, MIL=200*10000,TEST_BATCH=400*10000)
    train(NF=100*10000, MIL=10000*10000,TEST_BATCH=400*10000, frac=0.6) # use random
    train(NF=100*10000, MIL=10000*10000,TEST_BATCH=400*10000, frac=0.8) # use random
    train(NF=100*10000, MIL=10000*10000,TEST_BATCH=400*10000, frac=1.0) # use random
    #train(NF=200*10000, MIL=10000*10000,TEST_BATCH=400*10000)
    #train(NF=300*10000, MIL=10000*10000,TEST_BATCH=400*10000)
