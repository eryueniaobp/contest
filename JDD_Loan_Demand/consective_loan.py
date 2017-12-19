# encoding=utf-8
"""
@author : pengalg

生成连续贷款的特征

有连续贷款的用户，非常着急,再次贷款的可能性很高


                  loan_amount  consecutive_loan
loan_amount          1.000000          0.994564
consecutive_loan     0.994564          1.000000
====================
outer  NOT FILLNA(0)
                      loan_amount  consecutive_loan  loan_sum  \
loan_amount              1.000000          0.994089  0.700524
consecutive_loan         0.994089          1.000000  0.694620
loan_sum                 0.700524          0.694620  1.000000
loan_consecutive_sum     0.727758          0.726157  0.993319

                      loan_consecutive_sum
loan_amount                       0.727758
consecutive_loan                  0.726157
loan_sum                          0.993319
loan_consecutive_sum              1.000000
====================
                      loan_amount  consecutive_loan  loan_sum  \
loan_amount              1.000000          0.993742  0.673521
consecutive_loan         0.993742          1.000000  0.671043
loan_sum                 0.673521          0.671043  1.000000
loan_consecutive_sum     0.706351          0.708540  0.991554

                      loan_consecutive_sum
loan_amount                       0.706351
consecutive_loan                  0.708540
loan_sum                          0.991554
loan_consecutive_sum              1.000000
====================
inner join
                      loan_amount  consecutive_loan  loan_sum  \
loan_amount              1.000000          0.994292  0.700524
consecutive_loan         0.994292          1.000000  0.694620
loan_sum                 0.700524          0.694620  1.000000
loan_consecutive_sum     0.727758          0.726157  0.993569

                      loan_consecutive_sum
loan_amount                       0.727758
consecutive_loan                  0.726157
loan_sum                          0.993569
loan_consecutive_sum              1.000000


loan_amount 与 consecutive_loan 的相关性非常高 ； 最后十一月份的 loan_sum 与之前的 loan_sum关联都很大>>
"""




import pandas as pd
import math

def is_consecutive(prev_loan_time, prev_plannum, loan_time):
    repay_clear= pd.to_datetime(prev_loan_time) + pd.DateOffset(months = prev_plannum)

    if (pd.to_datetime(loan_time) - repay_clear).days > 0:
        return False
    return True

def corr():
    """
    corr analysis
    :return:
    """
    loan = pd.read_csv('t_loan_consecutive.csv',header=0)
    loan['month'] =  loan['loan_time'].apply(lambda  x : x[5:7])

    loan_sum = loan[['uid','loan_amount','consecutive_loan']].groupby('uid').sum()
    """
                      loan_amount  consecutive_loan
loan_amount          1.000000          0.991327
consecutive_loan     0.991327          1.000000

    目前这种构造方式，总和相关性非常高.
    进一步验证下 预测能力
    """
    print loan_sum.corr()


    loan_train = loan[ loan['month'].isin(['08','09','10'])]

    loan_train_sum = loan_train[['uid','loan_amount','consecutive_loan']].groupby('uid').sum().reset_index()

    loan_test = loan[ loan['month'].isin(['11'])]

    loan_test_sum = loan_test[['uid', 'loan_amount', 'consecutive_loan']].groupby('uid').sum().reset_index()

    loan_test_sum.columns = ['uid','loan_sum','loan_consecutive_sum']

    df = pd.merge(loan_train_sum, loan_test_sum,on='uid',how = 'outer')

    print '=' * 20
    print 'outer  NOT FILLNA(0)'
    print df[['loan_amount', 'consecutive_loan', 'loan_sum', 'loan_consecutive_sum']].corr()
    print '=' * 20
    print df[['loan_amount','consecutive_loan','loan_sum','loan_consecutive_sum']].fillna(0).corr()

    print '=' * 20
    print 'inner join '
    df = pd.merge(loan_train_sum, loan_test_sum, on='uid')

    print df[['loan_amount', 'consecutive_loan', 'loan_sum', 'loan_consecutive_sum']].corr()






def build_consecutive():
    t_loan = '/home/mi/zeroplan/t_loan_sorted.csv'

    loan= pd.read_csv(t_loan,header=0)

    dvals = loan.values

    print 'uid,loan_time,loan_amount,plannum,consecutive_loan'
    dmap = {}
    for uid, loan_time , amount, plannum in dvals:

        if uid in dmap:
            prev_loan_time, prev_amount ,prev_plannum, prev_consecutive_amount = dmap[uid]
            if is_consecutive(prev_loan_time, prev_plannum, loan_time):
                consecutive_amount = math.log( 5**prev_consecutive_amount  -1  + 5**amount -1  +1 , 5)
                # consecutive_amount = math.log( 5**prev_consecutive_amount  -1  + 5**prev_amount -1  +1 , 5)
            else:
                consecutive_amount = amount
                # consecutive_amount = 0

            dmap[uid] = (loan_time, amount, plannum, consecutive_amount)
        else:
            consecutive_amount = amount
            # consecutive_amount = 0
            dmap[uid] = (loan_time, amount, plannum, consecutive_amount)


        print '{0},{1},{2},{3},{4}'.format(uid,loan_time,amount,plannum,consecutive_amount)


if __name__ =='__main__':
    # main()
    # build_consecutive()
    corr()
