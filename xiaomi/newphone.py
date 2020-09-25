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

CSV_COLUMN_DEFAULTS = [[0], [-1], [0], [0.0] , ['0'], ['0'], [0], ['0'], ['0']]
CSV_COLUMNS = ['label', 'a', 'b', 'vitality','appid_day1',"appid_day2", 'phone', 'vitality_seq', 'appid_weight']

def parse_line(line):
    columns = tf.io.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS, field_delim=';')
    features = dict(zip(CSV_COLUMNS, columns))
    # tf.print(features)
    # features['appid'] = tf.strings.to_number(tf.strings.split(features['appid'], sep=' '),tf.int32)
    appid1 = tf.strings.to_number(tf.strings.split(features['appid_day1'], sep=' '), tf.int32)
    appid2 = tf.strings.to_number(tf.strings.split(features['appid_day2'], sep=' '), tf.int32)
    # appid2 = tf.strings.to_number(tf.strings.split(features['appid_day2'], sep=' '), tf.int32)
    appid_size = 4
    days = 2
    appid_count = tf.stack([appid1, appid2], axis=0)  # shape :  2, appid_size
    # appid_weight = tf.reshape(tf.strings.to_number(tf.strings.split(features['appid_weight'], sep=' '), tf.float32), [appid_size, 1])


    vitality_seq = tf.strings.to_number(tf.strings.split(features['vitality_seq'], sep=' '),tf.int32)

    # app_size = 4
    # appid = tf.cond(tf.greater(tf.shape(appid), app_size), lambda: tf.slice(appid, begin=[0], size=[app_size]), lambda: appid)
    # appid = tf.cond(tf.less(tf.shape(appid), app_size), lambda: tf.pad(appid, [0, app_size - tf.shape(appid).numpy()[0]]), lambda: appid)

    # appid = tf.keras.preprocessing.sequence.pad_sequences(appid, maxlen= app_size, value=0)
    features['appid_count'] = appid_count
    features['vitality_seq'] = vitality_seq
    # features['appid_weight'] = appid_weight

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


def build_functional_complied_model_with_cnn():
    """
    使用cnn进行序列型特征处理
    假设有200个app  ; lstm可以同时使用 次数和时间，可以使用log(x)进行简单归一化
    :return:
    """

    phone = tf.keras.layers.Input(shape=(1,), name='phone')
    embedding_phone = tf.keras.layers.Embedding(100, 10)(phone)
    flatted_phone = tf.keras.layers.Flatten()(embedding_phone)

    appid_size = 4
    days = 2
    # 30: 30 days
    appid_count = tf.keras.layers.Input(shape=(days, appid_size), name='appid_count')

    #output shape: (batch,  units=100)
    # appid_context_value = tf.keras.layers.LSTM(units=100, return_sequences=False, return_state=False)(appid_count)

    #outputshape : (batch, days - kernel_size + 1,  filters)
    appid_context_value =  tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(days, appid_size))(appid_count)
    pooling_context_value = tf.keras.layers.GlobalAveragePooling1D()(appid_context_value)



    context = tf.keras.layers.concatenate([flatted_phone, pooling_context_value])

    fusion_context = tf.keras.layers.Dense(10, activation='relu', name='context_layer')(context)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(fusion_context)

    model = tf.keras.models.Model(inputs=[appid_count,  phone], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    return model


def build_functional_complied_model_with_lstm():
    """
    使用app 次数或者时间 来构造 序列型特征，输入到lstm进行处理.
    假设有200个app  ; lstm可以同时使用 次数和时间，可以使用log(x)进行简单归一化
    :return:
    """

    phone = tf.keras.layers.Input(shape=(1,), name='phone')
    embedding_phone = tf.keras.layers.Embedding(100, 10)(phone)
    flatted_phone = tf.keras.layers.Flatten()(embedding_phone)

    appid_size = 4
    days = 2
    # 30: 30 days
    appid_count = tf.keras.layers.Input(shape=(days, appid_size), name='appid_count')

    #output shape: (batch,  units=100)
    appid_context_value = tf.keras.layers.LSTM(units=100, return_sequences=False, return_state=False)(appid_count)

    context = tf.keras.layers.concatenate([flatted_phone, appid_context_value])

    fusion_context = tf.keras.layers.Dense(10, activation='relu', name='context_layer')(context)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(fusion_context)

    model = tf.keras.models.Model(inputs=[appid_count,  phone], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    return model


"""
todo: add Attention ! 
"""
def build_functional_compiled_model2():
    vitality_seq_size = 5
    appid_size = 4
    a = tf.keras.layers.Input(shape=(1,) , name='a')
    b = tf.keras.layers.Input(shape=(1,) , name='b')
    appid = tf.keras.layers.Input(shape=(appid_size,) , name='appid')
    appid_weight = tf.keras.layers.Input(shape=(appid_size, 1 ) , name='appid_weight')


    vitality = tf.keras.layers.Input(shape=(1,) , name='vitality')
    vitality_seq = tf.keras.layers.Input(shape=(vitality_seq_size,) , name='vitality_seq')

    phone = tf.keras.layers.Input(shape=(1,) , name='phone')
    """
    下面要用attention，这里三个embedding的 dimension要一致.
    todo: 采用multihead-Attention
    """
    embedding_phone = tf.keras.layers.Embedding(100, 10)(phone)
    embedding_vitality = tf.keras.layers.Embedding(100, 10)(vitality)

    embedding_appid= tf.keras.layers.Embedding(100, 10)(appid)

    # transpose_appid_weight = tf.reshape(appid_weight, [4,1])

    # print(tf.shape(transpose_appid_weight))
    app_id_weight2 = tf.tile(appid_weight, tf.constant([1, 1, 10], tf.int32))


    weighted_embedding_appid = tf.keras.layers.multiply([app_id_weight2, embedding_appid ])

    attention_phone_appid = tf.keras.layers.Attention()([embedding_phone, weighted_embedding_appid]) # 1, |app_embeding size|
    attention_vitality_appid = tf.keras.layers.Attention()([embedding_vitality, weighted_embedding_appid]) # 1, |app_embedding_size|

    # flatten_phone_appid = tf.keras.layers.Flatten()(attention_phone_appid)
    # flatten_vitality_appid = tf.keras.layers.Flatten()(attention_vitality_appid)
    # flatten_embedding = tf.keras.layers.Flatten()(embedembedding_appidding_d)
    phone_appid_score = tf.keras.layers.Dense(1, activation='relu')(attention_phone_appid)

    # 1, 1
    # weighted_phone_appid_score = tf.keras.layers.multiply([appid_weight, phone_appid_score])

    vitality_appid_score = tf.keras.layers.Dense(1, activation='relu')(attention_vitality_appid) # 1, 1
    # weighted_vitality_appid_score = tf.keras.layers.multiply([appid_weight, vitality_appid_score])

    vitality_score = tf.keras.layers.Dense(6, activation='relu', name='vitality_score')(vitality_seq)
    input = tf.keras.layers.concatenate([a, b, vitality, tf.keras.layers.Flatten()(phone_appid_score), tf.keras.layers.Flatten()(vitality_appid_score), vitality_score ])
    context = tf.keras.layers.Dense(10, activation='relu', name='context_layer')(input)
    # context2 = tf.keras.layers.Dense(8)(feature2)
    # context = tf.keras.layers.concatenate([context1, context2])

    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(context)
    model = tf.keras.models.Model(inputs=[a, b,appid, appid_weight,  vitality,  phone, vitality_seq ], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    return model
if __name__ == '__main__':
    # model = build_functional_compiled_model2()
    # model = build_functional_complied_model_with_lstm()
    model = build_functional_complied_model_with_cnn()
    dataset = build_dataset()

    # dataset = dataset.shuffle(buffer_size=1024).padded_batch(1, padded_shapes=({'appid': 4 }, None))
    dataset = dataset.shuffle(buffer_size=1024).batch(8).repeat()
    # for x in dataset.take(1):
    #     print(x)
    # model.fit({'feature': X, 'feature2': X2}, y, batch_size=20, epochs=2)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/weights',
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    model.fit(dataset, steps_per_epoch=100, epochs=10, callbacks=[model_checkpoint_callback])

    # model.load_weights('./model/weights')
    #
    # test_dataset  = build_dataset()
    # test_dataset = test_dataset.map(lambda x, y : x )
    # test_dataset = test_dataset.batch(10)
    # for test_sample in test_dataset:
    #     # print(test_sample)
    #     result = model.predict(test_sample)
    #     print(result)

