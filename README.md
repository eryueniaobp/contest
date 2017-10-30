# contest
## 流程
特征工程： 交叉特征 + 统计比率      
baseline: gbdt    
单模型调优:    
模型融合:  stacking 4-folds .    

## 经验

gbdt 有效   
模型融合有效   
特征工程决定最终效果上限 ；过多特征造成 模型调优困难   
NN 耗费资源，单NN效果可能略好于调好的 gbdt； relu造成节点萎缩，慎用 ； feature_size * nn_size 容易OOM ， nn_size 第一层一般为   512 256 
