# encoding=utf-8
import lightgbm as lgb
from sklearn.datasets import  load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MaxAbsScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,ExtraTreesRegressor
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from  sklearn import  svm
import numpy as np
from collections import  namedtuple
from sklearn import linear_model
import logging,json, simplejson

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',

                    filemode='a')

def lgbcv():
    """
    train.cls.txt
    -0.035534853456
    {'num_leaves': 16, 'learning_rate': 0.02, 'min_child_samples': 20}


    train.txt
    -0.0349368112809
    {'num_leaves': 8, 'learning_rate': 0.05, 'min_child_samples': 60}

    :return:
    """
    X, y = load_svmlight_file('train.txt')
    param_grid = {
        'learning_rate': [0.02, 0.05, 0.1],
        'num_leaves': [8, 16],
        'min_child_samples': [5,10,20,60,80],
    }
    estimator =  lgb.LGBMRegressor(objective='mse',n_estimators=100)
    gbm = GridSearchCV(estimator, param_grid, verbose=2, scoring='neg_mean_squared_error', cv=5)
    gbm.fit(X,y)

    logging.info(  "lgbcv {0}".format(gbm.best_score_ ))
    logging.info(  "lgbcv {0}".format(gbm.best_params_))



def zoocv():
    X, y = load_svmlight_file('train.txt')
    Zoo = namedtuple('zoo', ['name','flag', 'param_grid', 'reg'],verbose=2)

    zoo = [
        Zoo('ridge', False, {'alpha': [0,0.5]}, linear_model.Ridge(alpha=0.5)),
        # Zoo('ridge with scaler', True, {'alpha':[0,0.5,1]}, Pipeline(
        #     [
        #         ('scaler', MaxAbsScaler()), ('ridge', linear_model.Ridge())
        #      ])
        #     ),
        Zoo('ridge with scaler', True, {'alpha':[0,0.5,1]},  linear_model.Ridge()),


        Zoo('svr', False, {'C': [1,10,100,1000] }  , svm.SVR(kernel='rbf', C=1e3)),
        Zoo('svr with scaler', True, {'C': [1, 10,100,1000]}, svm.SVR()),

        Zoo('gbdt', False, {'n_estimators':[60, 100,200] , 'max_depth':[4,6] , 'learning_rate': [0.05,0.1] } , GradientBoostingRegressor(n_estimators=100, max_depth=6,loss='ls')),
        Zoo('rf', False, {'n_estimators':[100,200,600], 'max_depth':[4,6,8]},  RandomForestRegressor(n_estimators=100, max_depth=6,max_features='auto')) ,
        Zoo('extraRF',False, {'n_estimators':[100,200,600], 'max_depth':[4,6,8]}, ExtraTreesRegressor(n_estimators=100, max_depth=6 , max_features='auto')),
        Zoo('lgb',False, {'n_estimators':[100,200,600], 'num_leaves':[6,8,16],'learning_rate': [0.05,0.1] } , lgb.LGBMRegressor(n_estimators=100, num_leaves=8 , objective='mse'))
    ]
    for name , flag,  param_grid, reg in zoo:

        if flag:
            X_g = X.toarray()
            X_g = MaxAbsScaler().fit(X_g).transform(X_g)
        else:
            X_g = X
        gs = GridSearchCV(estimator=reg, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=2 ,cv = 5)
        gs.fit(X_g,y)
        logging.info('zoo {0} best_result = {1} best_param = {2}'.format(name, gs.best_score_, gs.best_params_))


def trainlincv():
    """


     'mean_test_score': array([-0.03932149]),

    {'rank_test_score': array([1], dtype=int32),
    'split4_test_score': array([-0.0461944]),
    'mean_train_score': array([-0.02850378]),
    'split0_train_score': array([-0.0309599]),
    'std_test_score': array([ 0.00681591]),
    'std_train_score': array([ 0.00131567]),
    'split1_train_score': array([-0.02703828]),
    'split0_test_score': array([-0.03057691]),
    'mean_test_score': array([-0.03932149]),
    'split3_train_score': array([-0.02786938]),
    'split2_train_score': array([-0.02825937]),
    'std_score_time': array([  3.78773219e-05]),
    'params': [{'alpha': 0.5}],
     'std_fit_time': array([ 0.6320562]),
     'split4_train_score': array([-0.02839197]),
     'split2_test_score': array([-0.03484985]),
     'split3_test_score': array([-0.03664231]),
     'mean_score_time': array([ 0.00142102]),
     'mean_fit_time': array([ 6.05255213]),
     'param_alpha': masked_array(data = [0.5],mask = [False],fill_value = ?),
     'split1_test_score': array([-0.04834397])
}

    :return:
    """
    X, y = load_svmlight_file('train.txt')
    X = X.toarray()  # toarray()  or todense() 后， cv mse会变得非常高，出现 ill-conditioned matrix.
    X = X[:,56:] #采用这个以后，cv mse还是很高 .
    # scaler = StandardScaler().fit(X)
    #scaler = MaxAbsScaler().fit(X)
    # X = scaler.transform(X)
    #y = np.array(y).reshape((len(y),1))
    #scaler = StandardScaler().fit(y)
    #y = scaler.transform(y)
    estimator = linear_model.Ridge(alpha=0.5)
    param_grid = {
        'alpha': [0,0.5]
        #'alpha': [1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4]
    }
    gbm = GridSearchCV(estimator, param_grid,verbose=2,scoring='neg_mean_squared_error',cv=5)
    gbm.fit(X,y)
    logging.info('param:score = {0}:{1}'.format(gbm.best_params_ ,gbm.best_score_))
    print (gbm.best_params_)
    print (gbm.best_score_)
    print (gbm.cv_results_)

def predict_with_leaf(lgb, linr , sample , xlsx, tag):
    X_test, y_test = load_svmlight_file(sample)
    X_leaf = lgb.apply(X_test)

    y_pred = linr.predict(X_leaf)
    df = pd.read_excel(xlsx, header=0)
    pd.DataFrame({'id': df['ID'].values, 'score': y_pred})\
        .to_csv(sample + '.leafpred.{0}.csv'.format(tag),
                index=False,header=False)
def lgblin():
    """
    叶子节点  + Linear Regression
    :return:
    """
    X , y = load_svmlight_file("train.txt")
    params = {
        'objective': 'mse',
        'num_leaves': 8,
        'learning_rate': 0.05,
        'min_child_samples': 60,  # 这个题目比较关键 .
        # 'subsample': 0.9,
        'n_estimators': 100,
        'silent': False,
    }
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X,y , eval_metric='mse', eval_set=[(X,y)])

    X_leaf = gbm.apply(X)

    ridge = linear_model.Ridge(alpha=0.5)
    ridge.fit(X_leaf, y)
    mse = mean_squared_error(y , ridge.predict(X_leaf))
    logging.info('leaf mse = {0}'.format(mse))

    predict_with_leaf(gbm, ridge, 'testA.txt' , 'testA.xlsx' , '')
    predict_with_leaf(gbm, ridge, 'testB.txt', 'testB.xlsx', '')





def rf():

    X, y = load_svmlight_file('train.txt')
    reg = RandomForestRegressor(n_estimators=600, max_depth=8,max_features='auto',verbose=2)
    reg.fit(X,y)
    logging.info('rf mse = ' + str(mean_squared_error(y, reg.predict(X))))
    predict(reg, 'testA.txt', 'testA.xlsx', 'rf')
    predict(reg, 'testB.txt', 'testB.xlsx', 'rf')
    
def linrfe():
    """
    为了快速计算完成， step=xx 需要设置大一些.

    ridge : 0.28+
    ridge + RFE: 0.28+
    线上却有0.045 ; 线下的这个测试看来完全不准确
    """
    X, y = load_svmlight_file('train.txt')
    X = X.toarray()
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit(X,y)
    print 'r^2=', reg.score(X,y)
    print 'train mse = ', mean_squared_error(y, reg.predict(X))

    rfe = RFE(estimator=reg, n_features_to_select=500, step=1000,verbose=2)
    rfe.fit(X, y)
    print 'rfe r^2 = ' , rfe.score(X, y)
    print 'rfe mse =' , mean_squared_error(y, rfe.predict(X))

    X_rfe = rfe.transform(X)
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_poly = poly.fit_transform(X_rfe) #直接处理会有 MemoryError

    param_grid  = {'alpha' :[0.5,1,10,100,1000,1e4,3e4]} 
    gbm = GridSearchCV(reg, param_grid,verbose=2,scoring='neg_mean_squared_error',cv=5)
    gbm.fit(X_poly, y)
    logging.info('after rfe poly, best_result = {0}'.format(gbm.best_score_))
    logging.info('after rfe poly, best_param= {0}'.format(gbm.best_params_))
    #mse =  reg.score(X_poly, y)
    #print 'after poly ' ,mean_squared_error(y, reg.predict(X_poly)) 
    #logging.info('rfe r^2 score= ' + str(mse) )

    params = {
        'objective': 'mse',
        'num_leaves': 8,
        'learning_rate': 0.05,
        'min_child_samples': 60,  # 这个题目比较关键 .
        # 'subsample': 0.9,
        'n_estimators': 100,
        'silent': False,
    }
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X_poly, y, eval_metric='mse', eval_set=[(X_poly,y)])

    logging.info('train lgb of poly = {0}'.format( mean_squared_error(y , gbm.predict(X_poly, y))))


    # X = rfe.transform(X)
    # logging.info('begin to predict')
    # predict(rfe, 'testA.txt', 'testA.xlsx', 'ridge.rfe')
    # predict(rfe, 'testB.txt', 'testB.xlsx', 'ridge.rfe')

def trainlinear():
    X, y = load_svmlight_file('train.txt')
    X = X.toarray()
    estimators = [('MaxAbs', MaxAbsScaler()), ('ridge', linear_model.Ridge(alpha=0.5))]
    pipe = Pipeline(estimators)
    pipe.fit(X,y)

    y_pred = pipe.predict(X)
    logging.info('trainlinear mse = ' + str(mean_squared_error(y,y_pred)))

    predict(pipe, 'testA.txt', 'testA.xlsx', 'ridge')
    predict(pipe, 'testB.txt', 'testB.xlsx', 'ridge')
def trainlinmap():
    """
    mse ~ alpha
    探测 alpha 对整体mse的影响,几乎无影响.
    INFO alpha:mse = 0:0.0289213035695
    Fri, 22 Dec 2017 10:47:24 train.py[line:49] INFO alpha:mse = 1:0.0289152061282
    Fri, 22 Dec 2017 10:47:34 train.py[line:49] INFO alpha:mse = 0.5:0.0289196872644
    Fri, 22 Dec 2017 10:47:43 train.py[line:49] INFO alpha:mse = 0.1:0.0289158636722
    Fri, 22 Dec 2017 10:47:52 train.py[line:49] INFO alpha:mse = 0.01:0.0289159148539
    Fri, 22 Dec 2017 10:48:02 train.py[line:49] INFO alpha:mse = 0.001:0.0289158416954
    Fri, 22 Dec 2017 10:48:11 train.py[line:49] INFO alpha:mse = 0.0001:0.028912850027


    :return:
    """
    X, y = load_svmlight_file('train.txt')

    pbuf = [0, 1, 0.5, 0.1, 0.01 ,0.001,0.0001]
    buf = []
    for alpha in pbuf:
        reg =  linear_model.Ridge(alpha=alpha)
        reg.fit(X,y)
        y_pred = reg.predict(X)
        buf.append(mean_squared_error(y,y_pred))

        logging.info('alpha:mse = {0}:{1}'.format(alpha, buf[-1]))

    pd.DataFrame({'alpha': pbuf , 'mse': buf}).to_csv('linmap.csv',index=False)
    plt.plot(pbuf, buf)
    plt.show()

    # predict(reg, 'testA.txt', 'testA.xlsx', 'ridge')
    # predict(reg, 'testB.txt', 'testB.xlsx', 'ridge')




def trainsvm():
    """
    均值估计，完全不可用
    :return:
    """
    X, y = load_svmlight_file('train.txt')
    X = X.toarray()
    pipe = Pipeline([("scaler", MaxAbsScaler()) , ('svm', svm.SVR(C=1))])
    """
    kernel = rbf is default .
    """
    # clf = svm.SVR(kernel='rbf',C=1)
    pipe.fit(X,y)

    print mean_squared_error(y, pipe.predict(X))
    predict(pipe, 'testA.txt', 'testA.xlsx','svm')
    predict(pipe, 'testB.txt', 'testB.xlsx','svm')

def predict(gbm, sample , xlsx, tag=''):
    X_test ,y_test = load_svmlight_file(sample)
    X_test = X_test.toarray()
    y_pred = gbm.predict(X_test)

    y_pred = np.array([ 4 if i > 4 else i for i in y_pred ])
    df = pd.read_excel(xlsx,header=0)

    print len(y_pred)
    print y_pred
    print len(df['ID'].values)
    pd.DataFrame({'id': df['ID'].values , 'score': y_pred}).to_csv( sample+'.pred.{0}.csv'.format(tag), index=False,header=False)
def main():
    params = {
        'objective': 'mse',
        'num_leaves': 8,
        'learning_rate': 0.05,
        'min_child_samples': 60,  #这个题目比较关键 .
        # 'subsample': 0.9,
        'n_estimators': 100,
        'silent': False,
    }


    params = {
        'objective': 'mse',
        'num_leaves': 16,
        'learning_rate': 0.02,
        'min_child_samples': 20,  #这个题目比较关键 .
        # 'subsample': 0.9,
        'n_estimators': 100,
        'silent': False,
    }
    gbm = lgb.LGBMRegressor(**params)
    train_txt = 'train.cls.txt'
    testA_txt= 'testA.cls.txt'
    testB_txt= 'testB.cls.txt'
    tag = 'lgb.cls'
    X,  y = load_svmlight_file(train_txt)

    X_train , y_train , X_vldt, y_vldt = train_test_split(X,y)

    print 'begin to fit'
    gbm.fit(X,y,eval_metric='mse', eval_set=[(X,y)])





    predict(gbm, testA_txt, 'testA.xlsx',tag)
    predict(gbm, testB_txt, 'testB.xlsx',tag)

    imp = pd.DataFrame({
        'fid': range(1, len(gbm.feature_importances_) +1 ) ,
        'imp': gbm.feature_importances_
    })
    fidmap = pd.read_csv('fid.map',header=0)


    sns.kdeplot(gbm.feature_importances_)
    pd.merge(fidmap, imp, on ='fid').to_csv('fea.imp',index=False)
    # plt.plot(gbm.feature_importances_)
    #
    #
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='lgb')
    args = parser.parse_args()

    if args.method == 'lgb':
        main()
    elif args.method == 'zoo':
        zoocv()
    elif args.method == 'lgbcv':
        lgbcv()
    elif args.method == 'linrfe':
        linrfe()
    elif args.method == 'rf':
        rf()
    elif args.method == 'lgblin':
        lgblin()
    elif args.method == 'svm':
        trainsvm()
    elif args.method == 'linear':
        trainlinear()
    elif args.method == 'lincv':
        trainlincv()
    elif args.method == 'linmap':
        trainlinmap()
    else:
        pass

