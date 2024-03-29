# contest
## 提分要点
- 采样使得训练数据和测试数据分布一致
- Ensemble   

上述两项是通用提分要点
## 流程
特征工程： 交叉特征 + 统计比率 + 实时list特征 + 兴趣比例特征  + match特征

兴趣比例:   点击list [ a,  a , b] , 当前是a, 那么 a的兴趣比例是 2/3  



*baseline*: 交叉特征 + 统计比率 + gbdt   

异常值特征:  利用IsolationTree等找出异常值，标定为特征；该类特征在某些问题中是有效的. 
单模型调优:    
模型融合:  stacking 4-folds .    

## 经验

gbdt 有效   
模型融合有效   
特征工程决定最终效果上限 ；过多特征造成 模型调优困难   
NN 耗费资源，单NN效果可能略好于调好的 gbdt； relu造成节点萎缩，慎用 ； feature_size * nn_size 容易OOM ， nn_size 第一层一般为   512 256 

## 特征工程

输入： 原始数据  
输出： libsvm样本

一般步骤:  

1.  特征ID化   
* 朴素特征
* 交叉特征
* 统计特征 
这里需要注意，要能够方便的进行 高低频截断 ； 出现次数过少 或过多的特征 就需要截断去掉 
2. 基于ID化的特征，与 原始数据一起 生成libsvm 格式的 样本 

上述样本构造出来以后，可以作为baseline 训练，查验基本的auc，logloss等指标     


特征工程做不扎实，直接影响后续的 模型效果 

## scikit feature 支持
DictVectorizer可直接onehot 

### 难点

各项维度的 统计均值  是工程难点 


## spark code

spark.read.csv() 
write.parquet()  
> parquet 在windows下无法跑，需要额外的配置；在spark/hadoop集群环境可跑

collect_set()  之后 row.getAs[mutable.WrappedArray[String]](field_name)取出 

collect_map()似乎不太容易适用,采用collect_set(concat()) 代替

Window.partitionBy().orderBy().rangeBetween(left, Window.currentRow)  取到当前orderBy的字段值 v ，然后取  [v + left , v+Window.currentRow] 得值，两边都是 闭区间

##  针对历史序列的编程模式 
存储uid对应的所有 历史id值，  id:timestamp; 直接join到 dataframe中
ItemInfo(id, name, category, ....) ， iteminfo通常可载入内存 

后续处理时，*依据 history_timestamp 与当前timestamp 找出时间窗 ,构造特征. *

- 多时间窗
- 每个窗口内进行Rank/Count/Match/Rate(match_rate)等，可以对泛化特征也这么做
- 设定K类，每个类填充对应的idlist 等信息，再适用attn(current_item, (category_list_1, category_list_2, ...category_list_k)) 进行处理 

## 常规统计特征处理
单维度统计
组合维度统计
统计值采用 二维离散化处理

### spark 写TFRecord, 方便tensorflow训练
```
   df.rdd.map(row => {val example = buildExample(row)
            (new BytesWritable(example.toByteArray), NullWritable.get())
          })
            .coalesce(10)
            .saveAsNewAPIHadoopFile(
              trainOutput,
              classOf[NullWritable],
              classOf[BytesWritable],
              classOf[TFRecordFileOutputFormat],
              sc.hadoopConfiguration
              ) 
```

##  tensorflow训练要点 
- early_stopping
- decay learning rate (LearningRateScheduler) 
- Best Export 


## GBDT /  NN 

赛题数据规模大，则NN占优
赛题数据规模小，最好是GBDT和NN都做一份 

# 单独App Retention类问题的特征构建
- 状态转移序列 直接当特征用  s1:s2:s3..:sn直接当特征 

