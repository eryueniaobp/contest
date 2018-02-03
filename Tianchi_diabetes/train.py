#encoding=utf-8
"""
@author : pengalg
"""
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from dateutil.parser import parse
import pickle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import  pandas as pd
import argparse
import logging
import numpy as np
import datetime
# from feature import *
#font = FontProperties(fname = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=14)
#matplotlib.use('qt4agg')
#matplotlib.rcParams['font.family'] = 'Droid Sans'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',

                    filemode='a')

def lgbx(X, y, X_test):

    estimator =  lgb.LGBMRegressor(objective='mse',n_estimators=100, learning_rate=0.05, num_leaves=8, subsample=0.8)
    pipe = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('scaler', MaxAbsScaler()),
        # ('select', SelectKBest(f_regression,k='all')),
        ('lgb', estimator),
        ])
    pipe.fit(X,y)
    cols = pipe.named_steps['poly'].get_feature_names(X.columns)
    pd.DataFrame({ 'col': cols, 'imp': pipe.named_steps['lgb'].feature_importances_}).to_csv('imp.csv', index=False)
def lgbcls(X, y, X_test):
    print X.shape, y.shape
    y = pd.cut(y , bins=[ 3, 6, 10,40],retbins=False)
    #print y
    y = LabelEncoder().fit_transform(y)
    print y
    logging.info('num class = ' + str(y.max() + 1))
    cls = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, num_class=y.max()+1)
    pipe = Pipeline([
    ('scaler',MaxAbsScaler()),
    ('lgb', cls)
    ])

    param_grid = [
    {
        
       'lgb__learning_rate':[0.05], 
       'lgb__num_leaves':[8],
       'lgb__subsample':[1.0],
    }
    ]

    gbm = GridSearchCV(pipe, param_grid, verbose=2, scoring='accuracy', cv=5)
    gbm.fit(X,y)
    logging.info(  "lgbclscv {0}".format(gbm.best_score_ ))
    logging.info(  "lgbclscv {0}".format(gbm.best_params_))
    score = gbm.predict(X_test)
    prob = gbm.predict_proba(X_test)
    print score
    print prob
    day = datetime.datetime.now().strftime('%Y%m%d')
    pd.DataFrame({'cls': score}).to_csv('cls.{0}.csv'.format(day), index=True,header=False)
    pd.DataFrame(prob).to_csv('prob.{0}.csv'.format(day), index=True,header=False)
    print score.min() , score.max()
    with open( day+'.cls.pickle','w') as f:
        pickle.dump(score, f)

def lgbcv(X, y, X_test):

    estimator =  lgb.LGBMRegressor(objective='mse',n_estimators=100)
    pipe = Pipeline([
        ('scaler', MaxAbsScaler()),
        #('poly', PolynomialFeatures(interaction_only=True)),
        # ('poly', InteractiveFeatures(degree =2, interaction_only=True,include_bias=False,inter_func=log1p)),
        # ('select', SelectKBest(f_regression,k='all')),
        ('lgb', estimator)
        ])
    param_grid = [
    {
        # 'select__k':[10,20,26,32,'all'],
       'lgb__learning_rate':[0.05,0.1], 
       'lgb__num_leaves':[6,8,16,31],
       'lgb__subsample':[0.8,1.0],
    }
    ]
    # param_grid = [
    # {
    #     'select__k':[40,80,100,200,300,400,600,'all'],
    #    'lgb__learning_rate':[0.05],
    #    'lgb__num_leaves':[8],
    #    'lgb__subsample':[0.8],
    # }
    # ]

    gbm = GridSearchCV(pipe, param_grid, verbose=2, scoring='neg_mean_squared_error', cv=5)
    gbm.fit(X,y)
    logging.info(  "lgbcv {0}".format(gbm.best_score_ ))
    logging.info(  "lgbcv {0}".format(gbm.best_params_))

    score = gbm.best_estimator_.predict(X_test)
    df = pd.DataFrame({'score':score})
    day = datetime.datetime.now().strftime('%Y%m%d')
    df.to_csv('diabetes.lgb.{0}.csv'.format(day),index=False,header=False)
def lgbkfold(X,y, X_test):
    """
    kfold  k estimators and predict with the mean of scores.
    :param X:
    :param y:
    :param X_test:
    :return:
    """

    estimator = lgb.LGBMRegressor(objective='mse',
                                  n_estimators=100,
                                  num_leaves=16,
                                  subsample=1.0,
                                  learning_rate=0.05,
                                  # verbose=2,
                                  )
    lgb_pipe = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('lgb', estimator)
    ])

    kf = KFold(n_splits=5)
    kfscore = np.zeros((X_test.shape[0], 5  ))
    i = 0
    for train_index, test_index in kf.split(X):
        X_train , X_train_test = X.iloc[train_index, :], X.iloc[test_index, :]

        y_train, y_train_test = y.iloc[train_index], y.iloc[test_index]

        lgb_pipe.fit( X_train,y_train)

        kfscore[:, i] = lgb_pipe.predict(X_test)
        i += 1

    score = np.mean(kfscore ,axis=1)

    df = pd.DataFrame({'score': score})
    day = datetime.datetime.now().strftime('%Y%m%d')
    df.to_csv('diabetes.nest.kflgb.{0}.csv'.format(day), index=False, header=False)

def make_category(df,dfA):
    gender = df.iloc[:,1].unique()
    print gender
    gender = [u'男', u'女']
    # age = df.iloc[:,2].unique()
    df[u'性别']= df[u'性别'].astype('category',categories=gender)
    #df[u'年龄']= df[u'年龄'].astype('category',categories=age)
    dfA[u'性别']= dfA[u'性别'].astype('category',categories=gender)
    #dfA[u'年龄']= dfA[u'年龄'].astype('category',categories=age)
    dfA[u'血糖'] = 1.
def transform(df):

    y = df[u'血糖']
    print np.sum(y.isna())
    deviation = 10
    logging.info('deviation = ' + str(deviation))
    mask = ~(np.abs( y - y.mean()) <= deviation * y.std()) #all need
    print (' outlier = {0}'.format(mask.sum()))
    logging.info(' outlier = {0}'.format(mask.sum()))

    df = df[~mask]
    y = y[~mask]

    gender = pd.get_dummies(df.iloc[:, 1])
    df['datelag'] = (pd.to_datetime(df[u'体检日期']) - parse('2017-10-09')).dt.days
    df1 = df.drop(columns = ['id',u'性别',u'体检日期'])

    df1.drop(columns = [u'血糖'],inplace=True)
    df2 = pd.concat([df1, gender], axis=1)
    df2 = df2.fillna(0)
    return df2, y

def main(args):
    df = pd.read_csv("d_train_20180102.csv",header=0,encoding='gb2312')
    dfA = pd.read_csv("d_test_B_20180128.csv",header=0,encoding='gb2312')
    print dfA.shape 
    print df[u'血糖'].describe()
    make_category(df,dfA)
    X, y = transform(df)
    X_test , _ = transform(dfA)

    print X[X.isna().values == True]
    print X.shape, X_test.shape, len(y)

    print X.shape , X_test.shape, len(y)
    print X[X.isna().values == True]




    if args.mode =='lgbcv':
        lgbcv(X, y,X_test)
    elif args.mode == 'lgbkf':
        lgbkfold(X, y, X_test)
    elif args.mode == 'lgbcls':
        lgbcls(X, y, X_test)
    elif args.mode == 'lgb':
        lgbx(X, y, X_test)
    else:
        raise "check your mode"

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='lgbkf')
    args = parser.parse_args()


    main(args)
