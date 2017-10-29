# encoding=utf-8
"""
@author : baipeng1@xiaomi.com

前提： libsvm format  train and test  [trian ,validation and test better]
输出: model and prediction

关键技术:
base_model:
    Xgboost,lightGBM, sklearn.GradientBoostingTree
    RandomForest
    ExtraTree
    Ababoost
    Logisticregression
    libFM
    MLP ( heavy,if you have enough time , relu+dropout)
    SVC ( just suit to small feature-size dataset ,not recommend)
ensemble:
    5-fold- Stacking with gbdt

TODO:

Reference:
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
"""

import pandas as pd
import numpy as np
import  re
import sklearn
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.datasets import load_svmlight_file

from sklearn.metrics import roc_auc_score
from stacking import stacking
import logging

import numpy as np
import scipy.stats as st
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pickle



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='mlflow.log',
                    filemode='a')


def predict():
    """
    进行预测
    :return:
    """
    pass
def generate_xy(train_file, test_file,  mode):
    if mode == 'cv':
        X, y = load_svmlight_file(train_file)

        logging.info("load train ok ")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logging.info("train_test_split ok ")
    elif mode == 'predict':


        X_train, y_train = load_svmlight_file(train_file)
        logging.info("load train ok ")
        X_test, y_test = load_svmlight_file(test_file)

        logging.info("load test ok ")

    else:
        raise Exception("Wrong mode")

    return X_train, X_test, y_train, y_test

def main():
    """
    基础版本：
    升级版本： xgboost,lightgbm, libfm , mlp

    stacking过程产生的模型个数:
    len(models) * n_folds
    :return:
    """
    train_file  = '/home/mi/redplan/red-goldcf-sample-libsvm/100w'
    test_file ='/home/mi/redplan/red-goldcf-test-sample-libsvm/all'

    logging.info(train_file )
    logging.info(test_file)

    mode = "predict"
    X_train, X_test, y_train, y_test = generate_xy(train_file,test_file,mode)




    models = [
        AdaBoostClassifier(n_estimators=10, learning_rate=1.),  # verbose
        ExtraTreesClassifier(n_estimators=100, max_depth=6 ,verbose=1) ,
        RandomForestClassifier(n_estimators=100, max_depth= 6,verbose=1) ,
        GradientBoostingClassifier(n_estimators= 10 , max_depth= 3 , loss='deviance',  learning_rate= 0.2 , subsample=0.8,verbose=2) ,

        LogisticRegression(penalty='l2', C=1.0 ,solver='liblinear',verbose=0),
    ]

    S_train,S_test = stacking(models,X_train,y_train,X_test ,
                                       regression=True, metric=roc_auc_score, n_folds=4,shuffle=True,
                                       random_state=0, verbose=2)


    save(S_train , 'level2.train')
    save(y_train, 'level2.train.label')

    save(S_test, 'level2.test')
    save(y_test, 'level2.test.label')


    logging.info("stacking out , level TWO train begins ")

    #If use XGBClassifer, you will predict 0, 1 ,not probability.
    cls = XGBRegressor(max_depth=3, learning_rate=0.1,
                        n_estimators=20, silent=False,objective='binary:logistic',
                        subsample=0.8)

    model = cls.fit(S_train,y_train)

    logging.info("level TWO train ok")

    y_pred = model.predict(S_test)

    save(y_pred , 'level2.test.pred')

    print y_pred[:10]
    if mode == 'cv':
        auclog =   "Final auc {0}".format( roc_auc_score(y_test,y_pred))
        logging.info( auclog )

        print auclog
    else:
        logging.info("predict done")
def save(obj, path):
    with open(path,'w') as f:
        pickle.dump(obj, f)
if __name__ == '__main__':
    main()
