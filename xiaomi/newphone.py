# encoding=utf-8
import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from typing import List
import tensorflow as tf

import numpy as np

from model import NewPhoneModel
appis = list(range(1, 100))
# model = NewPhoneModel(100).build_model()
appid_data = np.random.choice(appis, (100, 3))
energe_data = np.random.uniform(0.1, 1.0, size=(100,))
targets = np.random.choice([0,1 ], (100,))

CSV_COLUMN_DEFAULTS = [[0], [-1], [0], [0.0] , ['UNK'], [0], ]
CSV_COLUMNS = ['label', 'a', 'b', 'vitality','appid','phone']

def parse_line(line):
    columns = tf.io.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS, field_delim=';')
    features = dict(zip(CSV_COLUMNS, columns))
    # features['appid'] = tf.strings.to_number(tf.strings.split(features['appid'], sep=' '),tf.int32)
    appid = tf.strings.to_number(tf.strings.split(features['appid'], sep=' '),tf.int32)
    # app_size = 4
    # appid = tf.cond(tf.greater(tf.shape(appid), app_size), lambda: tf.slice(appid, begin=[0], size=[app_size]), lambda: appid)
    # appid = tf.cond(tf.less(tf.shape(appid), app_size), lambda: tf.pad(appid, [0, app_size - tf.shape(appid).numpy()[0]]), lambda: appid)

    # appid = tf.keras.preprocessing.sequence.pad_sequences(appid, maxlen= app_size, value=0)
    features['appid'] = appid

    # if  :  # only slice if longer
    #     appid = tf.slice(appid, begin=[0], size=[3])
    # features['appid'] = appid
    label = features.pop('label')
    # label  =features.pop('label')
    return features, label
def build_dataset():
    file_paths = tf.data.Dataset.list_files('./data/train*')
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda line: parse_line(line))
    return dataset
# def build_functional_compiled_model():
#     feature = tf.keras.layers.Input(shape=(4,) , name='feature')
#     feature2 = tf.keras.layers.Input(shape=(6,) , name='feature2')
#     context1 = tf.keras.layers.Dense(10)(feature)
#     context2 = tf.keras.layers.Dense(8)(feature2)
#     context = tf.keras.layers.concatenate([context1, context2])
#
#     output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(context)
#     model = tf.keras.models.Model(inputs=[feature, feature2], outputs=[output])
#     model.compile(loss='binary_crossentropy', optimizer='adam')
#     return model
"""
todo: add Attention ! 
"""
def build_functional_compiled_model2():
    a = tf.keras.layers.Input(shape=(1,) , name='a')
    b = tf.keras.layers.Input(shape=(1,) , name='b')
    appid = tf.keras.layers.Input(shape=(4,) , name='appid')


    vitality = tf.keras.layers.Input(shape=(1,) , name='vitality')
    phone = tf.keras.layers.Input(shape=(1,) , name='phone')
    """
    下面要用attention，这里三个embedding的 dimension要一致.
    todo: 采用multihead-Attention
    """
    embedding_phone = tf.keras.layers.Embedding(100, 10)(phone)
    embedding_vitality = tf.keras.layers.Embedding(100, 10)(vitality)

    embedding_appid= tf.keras.layers.Embedding(100, 10)(appid)

    attention_phone_appid = tf.keras.layers.Attention()([embedding_phone, embedding_appid])
    attention_vitality_appid = tf.keras.layers.Attention()([embedding_vitality, embedding_appid])

    flatten_phone_appid = tf.keras.layers.Flatten()(attention_phone_appid)
    flatten_vitality_appid = tf.keras.layers.Flatten()(attention_vitality_appid)
    # flatten_embedding = tf.keras.layers.Flatten()(embedembedding_appidding_d)

    input = tf.keras.layers.concatenate([a, b, vitality, flatten_phone_appid, flatten_vitality_appid ])
    context = tf.keras.layers.Dense(10, activation='relu')(input)
    # context2 = tf.keras.layers.Dense(8)(feature2)
    # context = tf.keras.layers.concatenate([context1, context2])

    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(context)
    model = tf.keras.models.Model(inputs=[a, b,appid, vitality,  phone], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    return model
model = build_functional_compiled_model2()
dataset = build_dataset()
dataset = dataset.shuffle(buffer_size=1024).batch(64).repeat()
# model.fit({'feature': X, 'feature2': X2}, y, batch_size=20, epochs=2)
model.fit(dataset, steps_per_epoch=100, epochs=2)


