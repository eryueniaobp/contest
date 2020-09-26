# encoding=utf-8
import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from typing import List
import tensorflow as tf

import numpy as np
import argparse


class Constant:
    prefix = "/home/baiyun/PycharmProjects/miwin/data_store/feature"
    nagtive_count = 392972.0
    positive_count = 41326.0
    brand_count = 50
    model_name_count = 114
    version_count = 2230
    user_age_count = 9
    user_sex_count = 3
    user_degree_count = 8
    resident_province_count = 35
    resident_city_count = 377
    resident_city_type_count = 8
    phone_log_model_count = 184
    phone_raw_model_count = 135
    sale_channel_1_count = 13
    sale_channel_2_count = 34
    train_base_feature_vatality_header = [
        "uid", "label", "brand", "modelname", "version", "total_use_days", "user_age", "user_sex", "age", "user_degree",
        "resident_province", "resident_city",
        "resident_city_type", "phone_log_model", "phone_raw_model", "sale_channel_1", "sale_channel_2"]
    vatality_headers = ["day{}".format(i) for i in range(30)]
    app_count_inversion_rate_headers = ["appcount{}".format(i) for i in range(203)]
    app_duration_inversion_rate_headers = ["appduration{}".format(i) for i in range(203)]
    base_feature_rate_headers = list(map(lambda x: "{}_rate".format(x), train_base_feature_vatality_header[2:]))
    train_base_feature_recode_default = ["unk", 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0]
    train_base_feature_vatality = "{}/train_base_feature_vatality/part-00000".format(prefix)
    test_base_feature_vatality = "{}/test_base_feature_vatality/part-00000".format(prefix)
    train_valid_app_inversion_rate_dataset = "{}/train_label_valid_app_inversion_rate_csv/part-00000".format(prefix)
    test_valid_app_inversion_rate_dataset = "{}/test_valid_app_inversion_rate_csv/part-00000".format(prefix)
    train_base_and_app_rate = "{}/train_base_and_app_rate/part-00000".format(prefix)
    test_base_and_app_rate = "{}/test_base_and_app_rate/part-00000".format(prefix)
    train_file_path = "./data_store/train_data"



def parse_train_base_and_app_rate(line):
    deafult_value = Constant.train_base_feature_recode_default + \
                    [0 for _ in range(30)] + \
                    [0.0 for _ in range(406)] + \
                    [0.0 for _ in range(15)] + \
                    [0.0, "unk", "unk", "unk"] + \
                    ["unk" for _ in range(90)]
    columns = tf.io.decode_csv(line, record_defaults=deafult_value, field_delim=' ')
    features = dict(zip(Constant.train_base_feature_vatality_header, columns[:17]))
    features["vatality"] = columns[17:47]
    features["app_rate"] = columns[47:453]
    features["base_rate"] = columns[453:468]
    features["app_count"] = columns[468]
    features["day_app_use_count"] = tf.strings.to_number(tf.strings.split(columns[469], sep=','), tf.int32)
    features["app_caton_sum"] = tf.strings.to_number(tf.strings.split(columns[470], sep=','), tf.int64)
    features["all_app_use_info"] = tf.strings.to_number(tf.strings.split(columns[471], sep=','), tf.int32)

    appidcnt0 = tf.strings.to_number(tf.strings.split(columns[472], sep=','), tf.float32)
    appid_duration0 = tf.strings.to_number(tf.strings.split(columns[502], sep=','), tf.float32)
    appid_avg_duration0 = tf.strings.to_number(tf.strings.split(columns[532], sep=','), tf.float32)

    appid0 = tf.concat([appidcnt0, appid_duration0, appid_avg_duration0], axis=0)

    appidcnt1 = tf.strings.to_number(tf.strings.split(columns[473], sep=','), tf.float32)
    appid_duration1 = tf.strings.to_number(tf.strings.split(columns[503], sep=','), tf.float32)
    appid_avg_duration1 = tf.strings.to_number(tf.strings.split(columns[533], sep=','), tf.float32)

    appid1 = tf.concat([appidcnt1, appid_duration1, appid_avg_duration1], axis=0)

    appidcnt2 = tf.strings.to_number(tf.strings.split(columns[474], sep=','), tf.float32)
    appid_duration2 = tf.strings.to_number(tf.strings.split(columns[504], sep=','), tf.float32)
    appid_avg_duration2 = tf.strings.to_number(tf.strings.split(columns[534], sep=','), tf.float32)

    appid2 = tf.concat([appidcnt2, appid_duration2, appid_avg_duration2], axis=0)

    appidcnt3 = tf.strings.to_number(tf.strings.split(columns[475], sep=','), tf.float32)
    appid_duration3 = tf.strings.to_number(tf.strings.split(columns[505], sep=','), tf.float32)
    appid_avg_duration3 = tf.strings.to_number(tf.strings.split(columns[535], sep=','), tf.float32)

    appid3 = tf.concat([appidcnt3, appid_duration3, appid_avg_duration3], axis=0)

    appidcnt4 = tf.strings.to_number(tf.strings.split(columns[476], sep=','), tf.float32)
    appid_duration4 = tf.strings.to_number(tf.strings.split(columns[506], sep=','), tf.float32)
    appid_avg_duration4 = tf.strings.to_number(tf.strings.split(columns[536], sep=','), tf.float32)

    appid4 = tf.concat([appidcnt4, appid_duration4, appid_avg_duration4], axis=0)

    appidcnt5 = tf.strings.to_number(tf.strings.split(columns[477], sep=','), tf.float32)
    appid_duration5 = tf.strings.to_number(tf.strings.split(columns[507], sep=','), tf.float32)
    appid_avg_duration5 = tf.strings.to_number(tf.strings.split(columns[537], sep=','), tf.float32)

    appid5 = tf.concat([appidcnt5, appid_duration5, appid_avg_duration5], axis=0)

    appidcnt6 = tf.strings.to_number(tf.strings.split(columns[478], sep=','), tf.float32)
    appid_duration6 = tf.strings.to_number(tf.strings.split(columns[508], sep=','), tf.float32)
    appid_avg_duration6 = tf.strings.to_number(tf.strings.split(columns[538], sep=','), tf.float32)

    appid6 = tf.concat([appidcnt6, appid_duration6, appid_avg_duration6], axis=0)

    appidcnt7 = tf.strings.to_number(tf.strings.split(columns[479], sep=','), tf.float32)
    appid_duration7 = tf.strings.to_number(tf.strings.split(columns[509], sep=','), tf.float32)
    appid_avg_duration7 = tf.strings.to_number(tf.strings.split(columns[539], sep=','), tf.float32)

    appid7 = tf.concat([appidcnt7, appid_duration7, appid_avg_duration7], axis=0)

    appidcnt8 = tf.strings.to_number(tf.strings.split(columns[480], sep=','), tf.float32)
    appid_duration8 = tf.strings.to_number(tf.strings.split(columns[510], sep=','), tf.float32)
    appid_avg_duration8 = tf.strings.to_number(tf.strings.split(columns[540], sep=','), tf.float32)

    appid8 = tf.concat([appidcnt8, appid_duration8, appid_avg_duration8], axis=0)

    appidcnt9 = tf.strings.to_number(tf.strings.split(columns[481], sep=','), tf.float32)
    appid_duration9 = tf.strings.to_number(tf.strings.split(columns[511], sep=','), tf.float32)
    appid_avg_duration9 = tf.strings.to_number(tf.strings.split(columns[541], sep=','), tf.float32)

    appid9 = tf.concat([appidcnt9, appid_duration9, appid_avg_duration9], axis=0)

    appidcnt10 = tf.strings.to_number(tf.strings.split(columns[482], sep=','), tf.float32)
    appid_duration10 = tf.strings.to_number(tf.strings.split(columns[512], sep=','), tf.float32)
    appid_avg_duration10 = tf.strings.to_number(tf.strings.split(columns[542], sep=','), tf.float32)

    appid10 = tf.concat([appidcnt10, appid_duration10, appid_avg_duration10], axis=0)

    appidcnt11 = tf.strings.to_number(tf.strings.split(columns[483], sep=','), tf.float32)
    appid_duration11 = tf.strings.to_number(tf.strings.split(columns[513], sep=','), tf.float32)
    appid_avg_duration11 = tf.strings.to_number(tf.strings.split(columns[543], sep=','), tf.float32)

    appid11 = tf.concat([appidcnt11, appid_duration11, appid_avg_duration11], axis=0)

    appidcnt12 = tf.strings.to_number(tf.strings.split(columns[484], sep=','), tf.float32)
    appid_duration12 = tf.strings.to_number(tf.strings.split(columns[514], sep=','), tf.float32)
    appid_avg_duration12 = tf.strings.to_number(tf.strings.split(columns[544], sep=','), tf.float32)

    appid12 = tf.concat([appidcnt12, appid_duration12, appid_avg_duration12], axis=0)

    appidcnt13 = tf.strings.to_number(tf.strings.split(columns[485], sep=','), tf.float32)
    appid_duration13 = tf.strings.to_number(tf.strings.split(columns[515], sep=','), tf.float32)
    appid_avg_duration13 = tf.strings.to_number(tf.strings.split(columns[545], sep=','), tf.float32)

    appid13 = tf.concat([appidcnt13, appid_duration13, appid_avg_duration13], axis=0)

    appidcnt14 = tf.strings.to_number(tf.strings.split(columns[486], sep=','), tf.float32)
    appid_duration14 = tf.strings.to_number(tf.strings.split(columns[516], sep=','), tf.float32)
    appid_avg_duration14 = tf.strings.to_number(tf.strings.split(columns[546], sep=','), tf.float32)

    appid14 = tf.concat([appidcnt14, appid_duration14, appid_avg_duration14], axis=0)

    appidcnt15 = tf.strings.to_number(tf.strings.split(columns[487], sep=','), tf.float32)
    appid_duration15 = tf.strings.to_number(tf.strings.split(columns[517], sep=','), tf.float32)
    appid_avg_duration15 = tf.strings.to_number(tf.strings.split(columns[547], sep=','), tf.float32)

    appid15 = tf.concat([appidcnt15, appid_duration15, appid_avg_duration15], axis=0)

    appidcnt16 = tf.strings.to_number(tf.strings.split(columns[488], sep=','), tf.float32)
    appid_duration16 = tf.strings.to_number(tf.strings.split(columns[518], sep=','), tf.float32)
    appid_avg_duration16 = tf.strings.to_number(tf.strings.split(columns[548], sep=','), tf.float32)

    appid16 = tf.concat([appidcnt16, appid_duration16, appid_avg_duration16], axis=0)

    appidcnt17 = tf.strings.to_number(tf.strings.split(columns[489], sep=','), tf.float32)
    appid_duration17 = tf.strings.to_number(tf.strings.split(columns[519], sep=','), tf.float32)
    appid_avg_duration17 = tf.strings.to_number(tf.strings.split(columns[549], sep=','), tf.float32)

    appid17 = tf.concat([appidcnt17, appid_duration17, appid_avg_duration17], axis=0)

    appidcnt18 = tf.strings.to_number(tf.strings.split(columns[490], sep=','), tf.float32)
    appid_duration18 = tf.strings.to_number(tf.strings.split(columns[520], sep=','), tf.float32)
    appid_avg_duration18 = tf.strings.to_number(tf.strings.split(columns[550], sep=','), tf.float32)

    appid18 = tf.concat([appidcnt18, appid_duration18, appid_avg_duration18], axis=0)

    appidcnt19 = tf.strings.to_number(tf.strings.split(columns[491], sep=','), tf.float32)
    appid_duration19 = tf.strings.to_number(tf.strings.split(columns[521], sep=','), tf.float32)
    appid_avg_duration19 = tf.strings.to_number(tf.strings.split(columns[551], sep=','), tf.float32)

    appid19 = tf.concat([appidcnt19, appid_duration19, appid_avg_duration19], axis=0)

    appidcnt20 = tf.strings.to_number(tf.strings.split(columns[492], sep=','), tf.float32)
    appid_duration20 = tf.strings.to_number(tf.strings.split(columns[522], sep=','), tf.float32)
    appid_avg_duration20 = tf.strings.to_number(tf.strings.split(columns[552], sep=','), tf.float32)

    appid20 = tf.concat([appidcnt20, appid_duration20, appid_avg_duration20], axis=0)

    appidcnt21 = tf.strings.to_number(tf.strings.split(columns[493], sep=','), tf.float32)
    appid_duration21 = tf.strings.to_number(tf.strings.split(columns[523], sep=','), tf.float32)
    appid_avg_duration21 = tf.strings.to_number(tf.strings.split(columns[553], sep=','), tf.float32)

    appid21 = tf.concat([appidcnt21, appid_duration21, appid_avg_duration21], axis=0)

    appidcnt22 = tf.strings.to_number(tf.strings.split(columns[494], sep=','), tf.float32)
    appid_duration22 = tf.strings.to_number(tf.strings.split(columns[524], sep=','), tf.float32)
    appid_avg_duration22 = tf.strings.to_number(tf.strings.split(columns[554], sep=','), tf.float32)

    appid22 = tf.concat([appidcnt22, appid_duration22, appid_avg_duration22], axis=0)

    appidcnt23 = tf.strings.to_number(tf.strings.split(columns[495], sep=','), tf.float32)
    appid_duration23 = tf.strings.to_number(tf.strings.split(columns[525], sep=','), tf.float32)
    appid_avg_duration23 = tf.strings.to_number(tf.strings.split(columns[555], sep=','), tf.float32)

    appid23 = tf.concat([appidcnt23, appid_duration23, appid_avg_duration23], axis=0)

    appidcnt24 = tf.strings.to_number(tf.strings.split(columns[496], sep=','), tf.float32)
    appid_duration24 = tf.strings.to_number(tf.strings.split(columns[526], sep=','), tf.float32)
    appid_avg_duration24 = tf.strings.to_number(tf.strings.split(columns[556], sep=','), tf.float32)

    appid24 = tf.concat([appidcnt24, appid_duration24, appid_avg_duration24], axis=0)

    appidcnt25 = tf.strings.to_number(tf.strings.split(columns[497], sep=','), tf.float32)
    appid_duration25 = tf.strings.to_number(tf.strings.split(columns[527], sep=','), tf.float32)
    appid_avg_duration25 = tf.strings.to_number(tf.strings.split(columns[557], sep=','), tf.float32)

    appid25 = tf.concat([appidcnt25, appid_duration25, appid_avg_duration25], axis=0)

    appidcnt26 = tf.strings.to_number(tf.strings.split(columns[498], sep=','), tf.float32)
    appid_duration26 = tf.strings.to_number(tf.strings.split(columns[528], sep=','), tf.float32)
    appid_avg_duration26 = tf.strings.to_number(tf.strings.split(columns[558], sep=','), tf.float32)

    appid26 = tf.concat([appidcnt26, appid_duration26, appid_avg_duration26], axis=0)

    appidcnt27 = tf.strings.to_number(tf.strings.split(columns[499], sep=','), tf.float32)
    appid_duration27 = tf.strings.to_number(tf.strings.split(columns[529], sep=','), tf.float32)
    appid_avg_duration27 = tf.strings.to_number(tf.strings.split(columns[559], sep=','), tf.float32)

    appid27 = tf.concat([appidcnt27, appid_duration27, appid_avg_duration27], axis=0)

    appidcnt28 = tf.strings.to_number(tf.strings.split(columns[500], sep=','), tf.float32)
    appid_duration28 = tf.strings.to_number(tf.strings.split(columns[530], sep=','), tf.float32)
    appid_avg_duration28 = tf.strings.to_number(tf.strings.split(columns[560], sep=','), tf.float32)

    appid28 = tf.concat([appidcnt28, appid_duration28, appid_avg_duration28], axis=0)

    appidcnt29 = tf.strings.to_number(tf.strings.split(columns[501], sep=','), tf.float32)
    appid_duration29 = tf.strings.to_number(tf.strings.split(columns[531], sep=','), tf.float32)
    appid_avg_duration29 = tf.strings.to_number(tf.strings.split(columns[561], sep=','), tf.float32)

    appid29 = tf.concat([appidcnt29, appid_duration29, appid_avg_duration29], axis=0)

    appid_count = tf.stack(
        [appid0, appid1, appid2, appid3, appid4, appid5, appid6, appid7, appid8, appid9, appid10, appid11, appid12,
         appid13, appid14, appid15, appid16, appid17, appid18, appid19, appid20, appid21, appid22, appid23, appid24,
         appid25, appid26, appid27, appid28, appid29], axis=0)

    features['app_all'] = appid_count
    labels = features.pop("label")
    features.pop("uid")
    return features, labels


def build_dataset(files, infer=False):
    train_size = int(0.7 * (Constant.positive_count + Constant.nagtive_count))
    # validation_size = int(0.3 * (Constant.positive_count + Constant.nagtive_count))
    file_paths = tf.data.Dataset.list_files(files)
    dataset = tf.data.TextLineDataset(file_paths)
    if not infer:
        # dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.map(lambda line: parse_train_base_and_app_rate(line))
        # train_dataset = dataset.take(train_size)
        # validation_dataset = dataset.skip(train_size)
        return dataset
    else:
        return None
        # dataset = dataset.map(lambda line: parse_test_base_and_app_rate(line))
        # return dataset

def build_functional_complied_model_with_cnn():
    brand_input = tf.keras.Input(shape=(1,), name="brand")
    model_name_input = tf.keras.Input(shape=(1,), name="modelname")
    version_input = tf.keras.Input(shape=(1,), name="version")
    total_use_days_input = tf.keras.Input(shape=(1,), name="total_use_days")
    user_age_input = tf.keras.Input(shape=(1,), name="user_age")
    user_sex_input = tf.keras.Input(shape=(1,), name="user_sex")
    # age_input = tf.keras.Input(shape=(1,), name="age")
    user_degree_input = tf.keras.Input(shape=(1,), name="user_degree")
    resident_province_input = tf.keras.Input(shape=(1,), name="resident_province")
    resident_city_input = tf.keras.Input(shape=(1,), name="resident_city")
    resident_city_type_input = tf.keras.Input(shape=(1,), name="resident_city_type")
    phone_log_input = tf.keras.Input(shape=(1,), name="phone_log_model")
    phone_raw_input = tf.keras.Input(shape=(1,), name="phone_raw_model")
    sale_channel_1_input = tf.keras.Input(shape=(1,), name="sale_channel_1")
    sale_channel_2_input = tf.keras.Input(shape=(1,), name="sale_channel_2")

    brand_embedding = tf.keras.layers.Embedding(Constant.brand_count, 10)(brand_input)
    model_name_embedding = tf.keras.layers.Embedding(Constant.model_name_count, 10)(model_name_input)
    version_embedding = tf.keras.layers.Embedding(Constant.version_count, 10)(version_input)
    user_age_embedding = tf.keras.layers.Embedding(Constant.user_age_count, 10)(user_age_input)
    user_sex_embedding = tf.keras.layers.Embedding(Constant.user_sex_count, 10)(user_sex_input)
    user_degree_embedding = tf.keras.layers.Embedding(Constant.user_degree_count, 10)(user_degree_input)
    resident_province_embedding = tf.keras.layers.Embedding(Constant.resident_province_count, 10)(
        resident_province_input)
    resident_city_embedding = tf.keras.layers.Embedding(Constant.resident_city_count, 10)(resident_city_input)
    resident_city_type_embedding = tf.keras.layers.Embedding(Constant.resident_city_type_count, 10)(
        resident_city_type_input)
    phone_log_embedding = tf.keras.layers.Embedding(Constant.phone_log_model_count, 10)(phone_log_input)
    phone_raw_embedding = tf.keras.layers.Embedding(Constant.phone_raw_model_count, 10)(phone_raw_input)
    sale_channel_1_embedding = tf.keras.layers.Embedding(Constant.sale_channel_1_count, 10)(sale_channel_1_input)
    sale_channel_2_embedding = tf.keras.layers.Embedding(Constant.sale_channel_2_count, 10)(sale_channel_2_input)

    total_use_days_input_embedding = tf.keras.layers.Embedding(5000, 100)(total_use_days_input)

    brand_flatten = tf.keras.layers.Flatten()(brand_embedding)
    model_name_flatten = tf.keras.layers.Flatten()(model_name_embedding)
    version_flatten = tf.keras.layers.Flatten()(version_embedding)
    user_age_flatten = tf.keras.layers.Flatten()(user_age_embedding)
    user_sex_flatten = tf.keras.layers.Flatten()(user_sex_embedding)
    user_degree_flatten = tf.keras.layers.Flatten()(user_degree_embedding)
    resident_province_flatten = tf.keras.layers.Flatten()(resident_province_embedding)
    resident_city_flatten = tf.keras.layers.Flatten()(resident_city_embedding)
    resident_city_type_flatten = tf.keras.layers.Flatten()(resident_city_type_embedding)
    phone_log_flatten = tf.keras.layers.Flatten()(phone_log_embedding)
    phone_raw_flatten = tf.keras.layers.Flatten()(phone_raw_embedding)
    sale_channel_1_flatten = tf.keras.layers.Flatten()(sale_channel_1_embedding)
    sale_channel_2_flatten = tf.keras.layers.Flatten()(sale_channel_2_embedding)
    total_use_days_input_flatten = tf.keras.layers.Flatten()(total_use_days_input_embedding)

    vitality_days = 30
    vitality_dimension = 10
    vitality_seq = tf.keras.layers.Input(shape=(vitality_days,), name='vatality')
    # shape  : (batch, vitality_days, 10 )
    embedding_vitality = tf.keras.layers.Embedding(3, vitality_dimension)(vitality_seq)
    # vitality_context_value = tf.keras.layers.LSTM(units=32, return_sequences=False, return_state=False)(embedding_vitality)

    vitality_conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(vitality_days, vitality_dimension))(embedding_vitality)
    vitality_pooling = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(vitality_conv1d)
    vitality_context_value = tf.keras.layers.Flatten()(vitality_pooling)





    appid_size = 203
    days = 30
    app_dimension = appid_size * 3
    # 30: 30 days
    app_all = tf.keras.layers.Input(shape=(days, app_dimension), name='app_all')

    appid_conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                             input_shape=(days, app_dimension ))(app_all)

    appid_pooling = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(appid_conv1d)
    appid_context_value = tf.keras.layers.Flatten()(appid_pooling)



    context = tf.keras.layers.concatenate([
        brand_flatten, model_name_flatten, version_flatten,
        total_use_days_input, user_age_flatten, user_sex_flatten,
        user_degree_flatten, resident_province_flatten, resident_city_flatten, resident_city_type_flatten,
        phone_log_flatten,
        phone_raw_flatten, sale_channel_1_flatten, sale_channel_2_flatten,
        total_use_days_input_flatten,
        appid_context_value, vitality_context_value])

    fusion_context = tf.keras.layers.Dense(32, activation='relu', name='context_layer')(context)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(fusion_context)

    model = tf.keras.models.Model(inputs=[
        brand_input, model_name_input, version_input, total_use_days_input, user_age_input, user_sex_input,
        user_degree_input, resident_province_input, resident_city_input, resident_city_type_input, phone_log_input,
        phone_raw_input, sale_channel_1_input, sale_channel_2_input,
        vitality_seq, app_all
    ], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])
    model.summary()

    return model

def build_functional_complied_model_with_lstm():
    """
    使用app 次数或者时间 来构造 序列型特征，输入到lstm进行处理.
    假设有200个app  ; lstm可以同时使用 次数和时间，可以使用log(x)进行简单归一化
    :return:
    """

    brand_input = tf.keras.Input(shape=(1,), name="brand")
    model_name_input = tf.keras.Input(shape=(1,), name="modelname")
    version_input = tf.keras.Input(shape=(1,), name="version")
    total_use_days_input = tf.keras.Input(shape=(1,), name="total_use_days")
    user_age_input = tf.keras.Input(shape=(1,), name="user_age")
    user_sex_input = tf.keras.Input(shape=(1,), name="user_sex")
    # age_input = tf.keras.Input(shape=(1,), name="age")
    user_degree_input = tf.keras.Input(shape=(1,), name="user_degree")
    resident_province_input = tf.keras.Input(shape=(1,), name="resident_province")
    resident_city_input = tf.keras.Input(shape=(1,), name="resident_city")
    resident_city_type_input = tf.keras.Input(shape=(1,), name="resident_city_type")
    phone_log_input = tf.keras.Input(shape=(1,), name="phone_log_model")
    phone_raw_input = tf.keras.Input(shape=(1,), name="phone_raw_model")
    sale_channel_1_input = tf.keras.Input(shape=(1,), name="sale_channel_1")
    sale_channel_2_input = tf.keras.Input(shape=(1,), name="sale_channel_2")

    brand_embedding = tf.keras.layers.Embedding(Constant.brand_count, 10)(brand_input)
    model_name_embedding = tf.keras.layers.Embedding(Constant.model_name_count, 10)(model_name_input)
    version_embedding = tf.keras.layers.Embedding(Constant.version_count, 10)(version_input)
    user_age_embedding = tf.keras.layers.Embedding(Constant.user_age_count, 10)(user_age_input)
    user_sex_embedding = tf.keras.layers.Embedding(Constant.user_sex_count, 10)(user_sex_input)
    user_degree_embedding = tf.keras.layers.Embedding(Constant.user_degree_count, 10)(user_degree_input)
    resident_province_embedding = tf.keras.layers.Embedding(Constant.resident_province_count, 10)(
        resident_province_input)
    resident_city_embedding = tf.keras.layers.Embedding(Constant.resident_city_count, 10)(resident_city_input)
    resident_city_type_embedding = tf.keras.layers.Embedding(Constant.resident_city_type_count, 10)(
        resident_city_type_input)
    phone_log_embedding = tf.keras.layers.Embedding(Constant.phone_log_model_count, 10)(phone_log_input)
    phone_raw_embedding = tf.keras.layers.Embedding(Constant.phone_raw_model_count, 10)(phone_raw_input)
    sale_channel_1_embedding = tf.keras.layers.Embedding(Constant.sale_channel_1_count, 10)(sale_channel_1_input)
    sale_channel_2_embedding = tf.keras.layers.Embedding(Constant.sale_channel_2_count, 10)(sale_channel_2_input)

    total_use_days_input_embedding = tf.keras.layers.Embedding(5000, 100)(total_use_days_input)

    brand_flatten = tf.keras.layers.Flatten()(brand_embedding)
    model_name_flatten = tf.keras.layers.Flatten()(model_name_embedding)
    version_flatten = tf.keras.layers.Flatten()(version_embedding)
    user_age_flatten = tf.keras.layers.Flatten()(user_age_embedding)
    user_sex_flatten = tf.keras.layers.Flatten()(user_sex_embedding)
    user_degree_flatten = tf.keras.layers.Flatten()(user_degree_embedding)
    resident_province_flatten = tf.keras.layers.Flatten()(resident_province_embedding)
    resident_city_flatten = tf.keras.layers.Flatten()(resident_city_embedding)
    resident_city_type_flatten = tf.keras.layers.Flatten()(resident_city_type_embedding)
    phone_log_flatten = tf.keras.layers.Flatten()(phone_log_embedding)
    phone_raw_flatten = tf.keras.layers.Flatten()(phone_raw_embedding)
    sale_channel_1_flatten = tf.keras.layers.Flatten()(sale_channel_1_embedding)
    sale_channel_2_flatten = tf.keras.layers.Flatten()(sale_channel_2_embedding)
    total_use_days_input_flatten = tf.keras.layers.Flatten()(total_use_days_input_embedding)

    vitality_days = 30
    vitality_seq = tf.keras.layers.Input(shape=(vitality_days, ), name='vatality')
    # shape  : (batch, vitality_days, 10 )
    embedding_vitality = tf.keras.layers.Embedding(3, 10)(vitality_seq)
    vitality_context_value = tf.keras.layers.LSTM(units=32, return_sequences=False, return_state=False)(
        embedding_vitality)


    appid_size = 203
    days = 30
    # 30: 30 days
    app_all = tf.keras.layers.Input(shape=(days, appid_size * 3 ), name='app_all')

    #output shape: (batch,  units=100)
    appid_context_value = tf.keras.layers.LSTM(units=32, return_sequences=False, return_state=False)(app_all)

    context = tf.keras.layers.concatenate([
        brand_flatten, model_name_flatten, version_flatten,
        total_use_days_input, user_age_flatten, user_sex_flatten,
        user_degree_flatten, resident_province_flatten, resident_city_flatten, resident_city_type_flatten,
        phone_log_flatten,
        phone_raw_flatten, sale_channel_1_flatten, sale_channel_2_flatten,
        total_use_days_input_flatten,
        appid_context_value, vitality_context_value])

    fusion_context = tf.keras.layers.Dense(32, activation='relu', name='context_layer')(context)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(fusion_context)

    model = tf.keras.models.Model(inputs=[
        brand_input, model_name_input, version_input, total_use_days_input, user_age_input, user_sex_input,
        user_degree_input, resident_province_input, resident_city_input, resident_city_type_input, phone_log_input,
        phone_raw_input, sale_channel_1_input, sale_channel_2_input,
        vitality_seq,  app_all
    ], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    model.summary()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='task')
    trainparser = subparser.add_parser('train')
    trainparser.add_argument('--model', choices=['lstm', 'cnn'], required=True)

    testparser = subparser.add_parser('test')
    testparser.add_argument('--model', choices=['lstm', 'cnn'], required=True)

    args = parser.parse_args()
    if args.task == 'train':
        if args.model == 'lstm':
            model = build_functional_complied_model_with_lstm()
            model_path = './lstm_model/weights'
            csvfile = './lstm.csv'
        elif args.model == 'cnn':
            model = build_functional_complied_model_with_cnn()
            model_path = './cnn_model/weights'
            csvfile = './cnn.csv'
        else:
            raise NotImplementedError('check your model')

        # dataset = build_dataset()
        train_dataset = build_dataset("./train_data")
        valid_dataset = build_dataset("./valid_data")
        # dataset = dataset.shuffle(buffer_size=1024).padded_batch(1, padded_shapes=({'appid': 4 }, None))
        dataset = train_dataset.shuffle(buffer_size=1024).prefetch(tf.data.experimental.AUTOTUNE).batch(256).repeat()
        valid_dataset = valid_dataset.batch(64)
        # for x in dataset.take(4):
        #     print(x)
        # model.fit({'feature': X, 'feature2': X2}, y, batch_size=20, epochs=2)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
        import datetime
        ts = datetime.datetime.now().isoformat()
        csvlogger = tf.keras.callbacks.CSVLogger(filename=csvfile)

        class_weight = {1: Constant.nagtive_count / Constant.positive_count, 0: 1.0}
        model.fit(dataset, validation_data=valid_dataset,
                  class_weight= class_weight,
                  steps_per_epoch=2, epochs=2, callbacks=[model_checkpoint_callback, csvlogger])

