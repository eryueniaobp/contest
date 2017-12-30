# 特征

机器编号: OneHot Encoding 
日期列： 去掉 
其余列： 作为 连续特征 加进来  

最后特征维度： 7945 

## Scaler 
尝试过两种Scaler, StandardScaler and MaxAbsScaler  ,在svr和nn中，用上scaler效果好一些   ；linear regression用了 scaler反倒效果差一些

# 模型
Thu, 28 Dec 2017 18:02:21 train.py[line:92] INFO zoo ridge best_result = -0.039319154423 best_param = {'alpha': 0.5}  
Thu, 28 Dec 2017 18:02:22 train.py[line:92] INFO zoo ridge with scaler best_result = -0.0936259998498 best_param = {'alpha': 1}  
Thu, 28 Dec 2017 18:03:23 train.py[line:92] INFO zoo svr best_result = -0.0405294117628 best_param = {'C': 1}   
Thu, 28 Dec 2017 18:04:11 train.py[line:92] INFO zoo svr with scaler best_result = -0.0356260165125 best_param = {'C': 1}  
Thu, 28 Dec 2017 18:41:00 train.py[line:92] INFO zoo gbdt best_result = -0.0365618965467 best_param = {'n_estimators': 100, 'max_depth': 4}   
Thu, 28 Dec 2017 23:30:44 train.py[line:92] INFO zoo rf best_result = -0.0344977008462 best_param = {'n_estimators': 600, 'max_depth': 8}     
Fri, 29 Dec 2017 02:37:54 train.py[line:92] INFO zoo extraRF best_result = -0.0344578013869 best_param = {'n_estimators': 100, 'max_depth': 6   
Fri, 29 Dec 2017 03:03:47 train.py[line:92] INFO zoo lgb best_result = -0.0368011522604 best_param = {'n_estimators': 100, 'num_leaves': 16}    
Fri, 29 Dec 2017 19:12:18 train.py[line:55] INFO lgbcv -0.0349368112809   
Fri, 29 Dec 2017 19:12:18 train.py[line:56] INFO lgbcv {'num_leaves': 8, 'learning_rate': 0.05, 'min_child_samples': 60}     

## nn 的说明 
 尝试了nn，实验了几种简单的激活函数和网络节点数，线下cv保持在0.04x附近，观察预测输出，大部分都是同一个数字，明显不行
# 线上分数 

目前可用的四分结果： 
svr , lgb , lr ,rf 
lgb + svr 采用平均值融合 以后，分数是 0.0419  ; 采用 调和平均数融合，也是0.0419,几乎无差异 

# 无效尝试
RFE: recursive feature elimination ,逐步去掉一些特征，最后保留500个左右，模型训练出来以后，效果并没有提升 
PolyFeature:  直接多项式构造特征会内存溢出，先用rfe 筛选，之后再Poly，最后训练，cv结果并没有提升    




