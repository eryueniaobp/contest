# encoding=utf-8

import fire
import pandas as pd
import datetime
import lightgbm as lgb
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from mlxtend.classifier import StackingClassifier
import os
from stacking import stacking
import logging
import catboost

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train2.log',
                    filemode='a')


def today():
    return datetime.datetime.now().strftime('%Y%m%d')

def factorize(train_df, test_df):
    """
    训练集和测试集统一做factorize.

    把信息保留下来.
    :param df:
    :param test_df:
    :return:
    """

    # facdf = pd.read_csv('fac.csv', index_col=0)
    # cols = facdf[facdf['fac'] >= 50].index.values
    uniqdf  = pd.read_csv('uniq.corr.csv', index_col=0)
    cols = uniqdf[uniqdf['uniq'] > 10000]['col']


    colids = [  int(i[1:])  for i in cols ]
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    return df
    # for i in range(1, 298):
    for i in colids:
    # for i in [75]:

        col = 'f{}'.format(i)
        key = col + '_fac'

        convrate_path = key+'.convrate'
        hot_path = key + '.hot'

        qfac, bins = pd.qcut(df[col], q=np.linspace(0, 1, 11), retbins=True, duplicates='drop')

        fac, uniqs = pd.factorize(qfac, sort=True)

        logging.info('factorize {}  , bins = {} , uniqs = {}'.format(col, bins, uniqs))
        df[col + '_fac'] = fac

        if os.path.exists(convrate_path) and os.path.exists(hot_path):
            logging.info('load {} , {}'.format(convrate_path, hot_path))
            cnt = pd.read_csv(hot_path)
            fac_mean = pd.read_csv(convrate_path)
        else:
            ###转化率 和 分布
            train_df = df[df['label'] != -2]
            fac_mean = train_df.groupby(col +'_fac').label.mean().reset_index()
            fac_mean.columns = [col+'_fac',  'convrate']


            cnt =  train_df.groupby(['label', key ]).id.count().reset_index()
            cnt.columns = ['label', key , 'cnt']

            hotmap = cnt.groupby('label').cnt.transform(lambda  x :  x/(x.sum() + 1)  )
            cnt['hot'] = hotmap

            fac_mean.to_csv(key + '.convrate', index=False)

            cnt.to_csv(key + '.hot', index=False)

        pos_cnt = cnt[cnt['label'] == 1 ][[key , 'cnt', 'hot']]
        pos_cnt.columns = [key, 'pos_cnt', 'pos_hot']

        neg_cnt = cnt[cnt['label'] == 0][[key, 'cnt', 'hot']]
        neg_cnt.columns = [key, 'neg_cnt', 'neg_hot']


        df = pd.merge( df,   fac_mean , on=col+'_fac', how='left')

        df = pd.merge(df , pos_cnt, on=key, how='left')

        df = pd.merge(df , neg_cnt, on=key, how='left')


    return df






def poly_(df, columns):
    """
    先按照相关性选好，再poly.
    :param df:
    :param columns:
    :return:
    """
    for i ,  j in combinations(columns,2 ):
        col = '{}_div_{}'.format(i,j )
        df[col] = df[i]/df[j]

        col = '{}_mul_{}'.format(i,j)
        df[col] = df[i]*df[j]


def poly(df, columns):
    """
    根据接近的量纲 进行poly.
    :param df:
    :param columns:
    :return:
    """

    ff = pd.read_csv('fac.csv', index_col=0)

    facs = ff['fac'].unique()
    facs.sort()
    for fac in facs:
        columns = ff[ff['fac'] == fac].index.tolist()
        print ('poly fac= {}'.format(fac), columns)
        for i ,  j in combinations(columns,2 ):
            col = '{}_div_{}'.format(i,j )
            df[col] = df[i]/df[j]

            col = '{}_mul_{}'.format(i,j)
            df[col] = df[i]*df[j]



class SklearnHelperWithEarlyStopping(object):
    def __init__(self, clf, seed=0, params=None, eval_set=None, eval_metric='auc', early_stopping_rounds=3):
        # params['random_state'] = seed
        self.clf = clf(**params)
        self.eval_set = eval_set
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        """

        :param x:
        :return:  match  [:,1 ]
        """
        proba = self.clf.predict(x)
        a = [0] * len(proba)

        buf = []
        for i, v in enumerate(a):
            buf.append([v, proba[i]])

        return np.array(buf)

    def fit(self, x, y):
        # model = cls.fit(X_train, y_train, early_stopping_rounds=3, eval_set=[(X_vldt, y_vldt)], eval_metric='auc')
        return self.clf.fit(x, y, early_stopping_rounds=self.early_stopping_rounds, eval_set=self.eval_set,
                            eval_metric=self.eval_metric)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def fptpmetric(y_true, y_score):
    return metric(y_true, y_score)[0][1]


def metric(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    buf = []
    for i in range(len(fpr)):
        fp, tp, t = fpr[i], tpr[i], thresholds[i]

        if fp > 0.001:
            if len(buf) == 0:
                buf += [tp]
        if fp > 0.005:
            if len(buf) == 1:
                buf += [tp]
        if fp > 0.1 and len(buf) == 2:
            buf += [tp]

    auc = roc_auc_score(y_true, y_score)
    return [('strict_fptp', 0.4 * buf[0] + 0.3 * buf[1] + 0.3 * buf[2], True), ('auc', auc, True)]
def vote(debug=False):
    """
    本地的fptp，大概是在0.60，预计线上比较低的
    ===
    :param debug:
    :return:
    """
    train_path = 'data/atec_anti_fraud_train.csv'
    test_path = 'data/atec_anti_fraud_test_a.csv'
    if debug:
        nrows = 100000
    else:
        nrows = 10000 * 10000
    logging.info('begin main')
    train_df = pd.read_csv(train_path, nrows=nrows)

    train_df = train_df[train_df['label'] != -1]

    test_df = pd.read_csv(test_path, nrows=nrows)
    test_df['label'] = -2
    df = factorize(train_df, test_df)

    ### 关于 rate的特别处理 .
    df['frate_1'] = df['f83'] / (df['f84'] + 1)
    df['frate_2'] = df['f85'] / (df['f84'] + 1)
    df['frate_3'] = df['f86'] / (df['f84'] + 1)
    df['frate_82_84'] = df['f82'] / (df['f84'] + 1)

    df['frate_4'] = df['f82'] / (df['f85'] + 1)
    df['frate_5'] = df['f82'] / (df['f86'] + 1)
    df['frate_6'] = df['f85'] / (df['f86'] + 1)

    train_df = df[df['label'] != -2]
    test_df = df[df['label'] == -2]

    logging.info('traindf  shape = {}'.format(train_df.shape))
    logging.info('testdf shape = {} '.format(test_df.shape))

    y = train_df.pop('label')

    train_df.pop('id')
    train_df.pop('date')

    test_id = test_df.pop('id')
    test_df.pop('date')
    test_df.pop('label')

    # y_test = df_test.pop('label')
    #
    # df_test.pop('id')
    # df_test.pop('date')
    #
    # X_train = df_train
    # X_test = df_test
    X = train_df
    logging.info("will train-test split")

    logging.info("fitting..")

    cls1 = lgb.LGBMClassifier(objective='binary', n_estimators=100, subsample=0.8, subsample_freq=1,  colsample_bytree=0.8, num_leaves=31,
                             learning_rate=0.05, silent=False)


    rf1 = lgb.LGBMClassifier(boosting_type='rf' , objective='binary', n_estimators=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, num_leaves=31,learning_rate=0.05, silent=False)

    rf2 = lgb.LGBMClassifier(boosting_type='rf', objective='binary', n_estimators=400, subsample=0.8, subsample_freq=1,colsample_bytree=0.8, num_leaves=31, learning_rate=0.05, silent=False)

    cb = catboost.CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, loss_function='Logloss')
    vc = VotingClassifier(estimators=[('cb', cb) , ('gbdt', cls1), ('rf200', rf1), ('rf400', rf2) ], voting='soft', flatten_transform=False)

    vc.fit(X, y )
    # test_df = train_df
    # (n_classifiers, n_samples, n_classes)
    vc_score = vc.transform(test_df) #type: np.ndarray

    test_labels  = vc.predict(test_df)

    n_classifier, n_sample , n_classes  = vc_score.shape
    logging.info('orig shape = {}, test_labels = {},  test_mean = {} '.format(vc_score.shape, test_labels.shape, np.mean(test_labels)))

    vc_score = vc_score.swapaxes(0, 1 )
    logging.info('swapaxeis  shape = {}'.format(vc_score.shape))

    score = []
    for  i in range(n_sample):
        buf = []
        for j in range(n_classifier):
            k = np.argmax(vc_score[i][j], 0)
            if k == test_labels[i]:
                buf.append( vc_score[i][j][1])


        s = np.mean(buf,axis= 0)
        score.append(s)


    # logging.info('train metric = {}'.format(metric(y, score)))
    day = today()
    pd.DataFrame({'id': test_id, 'score': score}).to_csv('{}.{}.submit.csv'.format(day, 'vc'), index=False,
                                                         float_format='%.6f')

    logging.info('done')







def main(tag='', debug=False, use_kfold=True):
    train_path = 'data/atec_anti_fraud_train.csv'
    test_path = 'data/atec_anti_fraud_test_a.csv'
    if debug:
        nrows = 100000
    else:
        nrows = 10000 * 10000
    logging.info('begin main')
    train_df = pd.read_csv(train_path, nrows=nrows)

    train_df = train_df[train_df['label'] != -1]

    test_df = pd.read_csv(test_path, nrows=nrows)
    test_df['label'] = -2
    df = factorize(train_df ,  test_df)

    ### 关于 rate的特别处理 .
    df['frate_1'] = df['f83'] / (df['f84'] + 1)
    df['frate_2'] = df['f85'] / (df['f84'] + 1)
    df['frate_3'] = df['f86'] / (df['f84'] + 1)
    df['frate_82_84'] = df['f82'] / (df['f84'] + 1)

    df['frate_4'] = df['f82'] / (df['f85'] + 1)
    df['frate_5'] = df['f82'] / (df['f86'] + 1)
    df['frate_6'] = df['f85'] / (df['f86'] + 1)


    train_df = df[df['label'] != -2]
    test_df = df[df['label'] == -2]

    logging.info ('traindf  shape = {}'.format(train_df.shape))
    logging.info ('testdf shape = {} '.format(test_df.shape))


    y = train_df.pop('label')

    train_df.pop('id')
    train_df.pop('date')

    test_id = test_df.pop('id')
    test_df.pop('date')
    test_df.pop('label')


    # y_test = df_test.pop('label')
    #
    # df_test.pop('id')
    # df_test.pop('date')
    #
    # X_train = df_train
    # X_test = df_test
    X = train_df
    logging.info("will train-test split")


    logging.info("fitting..")


    

    if use_kfold:
        buf = []
        kf = KFold(n_splits=5, shuffle=False)
        for train_index, test_index in kf.split(y):
            estimator = lgb.LGBMClassifier(objective='binary', n_estimators=10000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     num_leaves=31,
                                     learning_rate=0.05, silent=False)
            estimator.fit(X.iloc[train_index], y.iloc[train_index],eval_set=[(X.iloc[test_index], y.iloc[test_index])], eval_metric=metric, early_stopping_rounds=80)

            logging.info('result = {}'.format(estimator.evals_result_['valid_0']['strict_fptp'][estimator.best_iteration_ - 1]))
            y_pred = estimator.predict_proba(test_df, num_iteration=estimator.best_iteration_)[:, 1]
            buf.append(y_pred)

        score = np.vstack(buf).mean(axis=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        cls = lgb.LGBMClassifier(objective='binary', n_estimators=10000, subsample=0.8, colsample_bytree=0.8, num_leaves=31,
                                 learning_rate=0.05, silent=False)
        # cls = lgb.LGBMClassifier(boosting_type='rf', objective='binary',n_estimators=10000, subsample=0.8, colsample_bytree=0.8 ,  num_leaves=31, learning_rate=0.05 ,silent=False)

        cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=metric, early_stopping_rounds=80)
        try:
            logging.info('result = {}'.format(cls.evals_result_['valid_0']['strict_fptp'][cls.best_iteration_ - 1]))
        except Exception as e:
            logging.warn(e)
    # cls = lgb.LGBMClassifier(objective='binary', n_estimators=100, num_leaves=31, learning_rate=0.05, silent=False)
    # cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=metric)
    #



        logging.info('will predict_proba')
        score = cls.predict_proba(test_df)[:, 1]
    
    
    if use_kfold: tag = '{}.kfold'.format(tag)
    day = today()
    pd.DataFrame({'id': test_id, 'score': score}).to_csv('{}.{}.submit.csv'.format(day, tag), index=False,
                                                         float_format='%.6f')

    logging.info('done')


def stack(tag='', debug=False):
    train_path = 'data/atec_anti_fraud_train.csv'
    test_path = 'data/atec_anti_fraud_test_a.csv'
    if debug:
        nrows = 100000
    else:
        nrows = 10000 * 10000
    df = pd.read_csv(train_path, nrows=nrows)
    df = df[df['label'] != -1].reset_index(drop=True).fillna(0).replace([np.inf], 0)

    # print (df.describe())
    # raw_input('\t\t')
    y = df.pop('label').values

    df.pop('id')
    df.pop('date')

    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lgbparam = {
        'objective': 'binary', 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 31,
        'learning_rate': 0.05, 'silent': False
    }
    rf_lgbparam = {
        'boosting_type': 'rf', 'objective': 'binary', 'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'num_leaves': 31, 'learning_rate': 0.05, 'silent': False
    }
    rf1_lgbparam = {
        'boosting_type': 'rf', 'objective': 'binary', 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'num_leaves': 31, 'learning_rate': 0.05, 'silent': False
    }
    rf2_lgbparam = {
        'boosting_type': 'rf', 'objective': 'binary', 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'num_leaves': 31, 'learning_rate': 0.05, 'silent': False
    }
    models = [
        # GradientBoostingClassifier(n_estimators=100, max_depth=4, loss='deviance', learning_rate=0.3, subsample=0.8,verbose=2),

        SklearnHelperWithEarlyStopping(lgb.LGBMClassifier, params=rf_lgbparam, eval_metric=metric,
                                       eval_set=[(X_test, y_test)], early_stopping_rounds=80),
        SklearnHelperWithEarlyStopping(lgb.LGBMClassifier, params=rf1_lgbparam, eval_metric=metric,
                                       eval_set=[(X_test, y_test)], early_stopping_rounds=80),
        SklearnHelperWithEarlyStopping(lgb.LGBMClassifier, params=rf2_lgbparam, eval_metric=metric,
                                       eval_set=[(X_test, y_test)], early_stopping_rounds=80),

        SklearnHelperWithEarlyStopping(lgb.LGBMClassifier, params=lgbparam, eval_metric=metric,
                                       eval_set=[(X_test, y_test)], early_stopping_rounds=80),
        # AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6), n_estimators=200, learning_rate=1.),
        # AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6), n_estimators=200, learning_rate=0.06),
        # AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6), n_estimators=200, learning_rate=1.),
        # AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6), n_estimators=200, learning_rate=0.06),
        # verbose
        # ExtraTreesClassifier(n_estimators=100, max_depth=6, verbose=1),

        # LogisticRegression(penalty='l2', C=1.0, solver='liblinear', verbose=0),
    ]

    test_df = pd.read_csv(test_path, nrows=nrows).fillna(0).replace([np.inf], 0)
    test_id = test_df.pop('id')

    test_df.pop('date')

    s_train, s_test, s_pred = stacking(models, X_train, y_train, X_test, test_df.values, regression=False,
                                       metric=fptpmetric, n_folds=4, shuffle=True, random_state=0, verbose=2)
    # cls = lgb.LGBMClassifier(objective='binary',n_estimators=1000, subsample=0.8, colsample_bytree=0.8 ,  num_leaves=31, learning_rate=0.05 ,silent=False)
    # cls.fit(X_train ,y_train, eval_set=[(X_test,y_test)], eval_metric=metric, early_stopping_rounds=80)

    # lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear',verbose=2)
    # lr.fit(s_train ,y_train)
    #
    # score = lr.predict_proba(s_test)[:,1]

    cls = lgb.LGBMClassifier(objective='binary', n_estimators=1000, subsample=0.8, colsample_bytree=0.8, num_leaves=31,
                             learning_rate=0.05, silent=False)
    cls.fit(s_train, y_train, eval_set=[(s_test, y_test)], eval_metric=metric, early_stopping_rounds=80)

    score = cls.predict_proba(s_pred)[:, 1]

    day = today()
    pd.DataFrame({'id': test_id, 'score': score}).to_csv('{}.{}.stack.submit.csv'.format(day, tag), index=False,
                                                         float_format='%.6f')


def neighbor():
    logging.info('neighbor begins')
    nrows = 100000 * 10000
    # nrows=10000
    train_path = 'data/atec_anti_fraud_train.csv'
    test_path = 'data/atec_anti_fraud_test_a.csv'
    df = pd.read_csv(train_path, nrows=nrows)
    df = df[df['label'] != -1].reset_index(drop=True).fillna(0).replace([np.inf], 0)
    # df = df[df['label'] != -1 ].reset_index(drop=True)

    y = df.pop('label')
    df.pop('id')
    df.pop('date')

    X = df
    logging.info('neighbor fitting..')
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf1.fit(X, y)

    # test_df = pd.read_csv(test_path, nrows=nrows)
    test_df = pd.read_csv(test_path, nrows=nrows).fillna(0).replace([np.inf], 0)
    test_id = test_df.pop('id')
    test_df.pop('date')
    logging.info('neighbor predicting...')
    score = clf1.predict_proba(test_df)[:, 1]
    day = today()
    pd.DataFrame({'id': test_id, 'score': score}).to_csv('{}.neigh.submit.csv'.format(day), index=False,
                                                         float_format='%.6f')
    logging.info('neighbor done...')


if __name__ == '__main__':
    fire.Fire()

