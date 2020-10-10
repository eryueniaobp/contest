# encoding=utf-8
from typing import List
import numpy as np
import xgboost as xgb
import logging
import re
import tensorflow as tf
import tensorflow_addons as tfa


def build_xgb_nn_model(tree_num=10):
    """
    use xgb leaf features inside.
    To have a different model space in moe, just use xgb leaf features here.
    :return:
    """
    inputs = []

    embeddings = []
    for i in range(tree_num):
        inputs.append(tf.keras.Input(shape=(1,), name="tree_" + str(i)))
        embeddings.append(
            tf.keras.layers.Embedding(128, 32)(inputs[-1])
        )
    # leaf = tf.keras.Input(shape=(10,), name="tree_leaf")
    # inputs.append(leaf)
    leaf_context = tf.keras.layers.concatenate(embeddings, axis=-1)
    flatten_context = tf.keras.layers.Flatten()(leaf_context)
    fusion_context = tf.keras.layers.Dense(32, 'relu')(flatten_context)
    output = tf.keras.layers.Dense(1, 'sigmoid')(fusion_context)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output], name='xgb_nn')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(),
        tfa.metrics.F1Score(num_classes=1, threshold=0.5)
    ])
    model.summary()
    return model
class XGBLeafDataSource(object):
    """
    read the files and parse out the "app_rate" feature and use the xgb to get the leafs.
    """
    xgb_modelfile = 'xgb.fornn.model'
    xgb_param_dist = {'objective': 'binary:logistic', 'n_estimators': 10, 'max_depth': 6, 'subsample': 0.8,
                      'learning_rate': 0.1}
    xgbclf = xgb.XGBClassifier(**xgb_param_dist)
    try:
        xgbclf.load_model(xgb_modelfile)
        logging.info('load xgb model ok')
    except:
        pass

    def __init__(self, path, batch_size, span):
        self.path = path
        self.batch_size = batch_size
        self.span = span # feature span
        pass
    def iter_test(self):
        while True:
            # features_size = 406
            features, label = np.random.random((self.batch_size, 406)), np.random.randint(0, 2, (self.batch_size, 1))
            # print(label)
            yield {'app_rate': features}, label

    def iter(self, epoch=None):
        k = 0
        s, e = self.span
        while True:
            # print(k)
            if k == epoch:
                # k = 0
                break
            k += 1

            with open(self.path, 'r') as f:
                batch = []
                ys = []
                for line in f:
                    us = line.split(" ")
                    y = int(us[1])
                    app_rate = [float(i) for i in us[s:e]]
                    # print(app_rate)
                    batch.append(app_rate)
                    ys.append(y)
                    if len(batch) == self.batch_size:
                        leafs = XGBLeafDataSource.xgbclf.apply(np.array(batch))

                        fs = {}
                        t = 0
                        for tree_leaf in np.transpose(leafs):
                            fs['tree_' + str(t)] =  tree_leaf
                            t += 1
                        yield fs, np.array(ys)
                        batch = []
                        ys = []



def xgbnn_train():
    train_path = "./data_store/train_data"
    valid_path = "./data_store/valid_data"


    model = build_xgb_nn_model(10)
    model_path = './xgbnn_model/weights'
    csvfile = './xgbnn.csv'
    submit = './submit.xgbnn.csv'

    batch_size = 10
    phone_source = XGBLeafDataSource(train_path, batch_size, (47, 453))
    valid_phone_source = XGBLeafDataSource(valid_path, batch_size, (47, 453))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)

    csvlogger = tf.keras.callbacks.CSVLogger(filename=csvfile)

    model.fit(phone_source.iter(),
              validation_data=valid_phone_source.iter(),
              steps_per_epoch=100, epochs=20,
              validation_steps=20,
              callbacks=[model_checkpoint_callback,csvlogger ]
              )
#     model.save_weights(model_path)

def main():
    train_path = "./data_store/train_data"
    valid_path = "./data_store/valid_data"
    test_path = "./data_store/test_data"
    batch_size = 10
    phone_source = XGBLeafDataSource(train_path, batch_size, (47, 453))

    for leafs in phone_source.iter(1):
        # pass
        # leafs = xgbclf.apply(features['app_rate'])
        print(type(leafs))
        print(leafs)
        input("Press any key .. ")


    pass
if __name__ == '__main__':
    xgbnn_train()
    # main()
