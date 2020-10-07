# encoding=utf-8
from typing import List
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_addons as tfa

from typing import Dict, Text
from phone_constant import Constant

# class UserModel(tf.keras.Model):
#     def __init__(self):
#         pass
#     def call(self):

class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.phone_embeddings = self.build_phone_tower(32)

        self.user_embeddings = self.build_user_tower(32)


        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            # tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1, 'sigmoid')
        ])
    def build_phone_tower(self, embedding_size):

        brand_input = tf.keras.Input(shape=(1,), name="brand")
        model_name_input = tf.keras.Input(shape=(1,), name="modelname")
        version_input = tf.keras.Input(shape=(1,), name="version")
        phone_log_input = tf.keras.Input(shape=(1,), name="phone_log_model")
        phone_raw_input = tf.keras.Input(shape=(1,), name="phone_raw_model")

        brand_embedding = tf.keras.layers.Embedding(Constant.brand_count, 10)(brand_input)
        model_name_embedding = tf.keras.layers.Embedding(Constant.model_name_count, 10)(model_name_input)
        version_embedding = tf.keras.layers.Embedding(Constant.version_count, 10)(version_input)
        phone_log_embedding = tf.keras.layers.Embedding(Constant.phone_log_model_count, 10)(phone_log_input)
        phone_raw_embedding = tf.keras.layers.Embedding(Constant.phone_raw_model_count, 10)(phone_raw_input)

        brand_flatten = tf.keras.layers.Flatten()(brand_embedding)
        model_name_flatten = tf.keras.layers.Flatten()(model_name_embedding)
        version_flatten = tf.keras.layers.Flatten()(version_embedding)
        phone_log_flatten = tf.keras.layers.Flatten()(phone_log_embedding)
        phone_raw_flatten = tf.keras.layers.Flatten()(phone_raw_embedding)

        context = tf.keras.layers.concatenate([
            brand_flatten,
            model_name_flatten,
            version_flatten,
            phone_log_flatten,
            phone_raw_flatten,
          ], name='phone_tower_context_layer')

        full_fusion_context = tf.keras.layers.Dense(128, activation='relu', name='phone_tower_fusion_layer')(context)
        phone_tower_embedding = tf.keras.layers.Dense(embedding_size, activation='relu', name='phone_tower_layer')(
            full_fusion_context)

        model = tf.keras.models.Model(inputs=[
            brand_input,
            model_name_input,
            version_input,
            phone_log_input ,
            phone_raw_input
        ], outputs=[phone_tower_embedding], name='phone_tower')

        return model
    def build_user_tower(self, embedding_size):
        # brand_input = tf.keras.Input(shape=(1,), name="brand")
        # model_name_input = tf.keras.Input(shape=(1,), name="modelname")
        # version_input = tf.keras.Input(shape=(1,), name="version")
        # phone_log_input = tf.keras.Input(shape=(1,), name="phone_log_model")
        # phone_raw_input = tf.keras.Input(shape=(1,), name="phone_raw_model")
        #

        total_use_days_input = tf.keras.Input(shape=(1,), name="total_use_days")
        user_age_input = tf.keras.Input(shape=(1,), name="user_age")
        user_sex_input = tf.keras.Input(shape=(1,), name="user_sex")
        # age_input = tf.keras.Input(shape=(1,), name="age")
        user_degree_input = tf.keras.Input(shape=(1,), name="user_degree")
        resident_province_input = tf.keras.Input(shape=(1,), name="resident_province")
        resident_city_input = tf.keras.Input(shape=(1,), name="resident_city")
        resident_city_type_input = tf.keras.Input(shape=(1,), name="resident_city_type")

        sale_channel_1_input = tf.keras.Input(shape=(1,), name="sale_channel_1")
        sale_channel_2_input = tf.keras.Input(shape=(1,), name="sale_channel_2")

        user_age_embedding = tf.keras.layers.Embedding(Constant.user_age_count, 10, name='age_emb')(user_age_input)
        user_sex_embedding = tf.keras.layers.Embedding(Constant.user_sex_count, 10, name='sex_emb')(user_sex_input)
        user_degree_embedding = tf.keras.layers.Embedding(Constant.user_degree_count, 10, name='degree_emb')(user_degree_input)
        resident_province_embedding = tf.keras.layers.Embedding(Constant.resident_province_count, 10, name='province_emb')(
            resident_province_input)
        resident_city_embedding = tf.keras.layers.Embedding(Constant.resident_city_count, 10, name='city_emb')(resident_city_input)
        resident_city_type_embedding = tf.keras.layers.Embedding(Constant.resident_city_type_count , 10, name='city_type_emb')(
            resident_city_type_input)
        # model_name_embedding = tf.keras.layers.Embedding(Constant.model_name_count, 10)(model_name_input)
        # phone_log_embedding = tf.keras.layers.Embedding(Constant.phone_log_model_count, 10)(phone_log_input)
        # phone_raw_embedding = tf.keras.layers.Embedding(Constant.phone_raw_model_count, 10)(phone_raw_input)
        sale_channel_1_embedding = tf.keras.layers.Embedding(Constant.sale_channel_1_count, 10, name='sale_ch1_emb')(sale_channel_1_input)
        sale_channel_2_embedding = tf.keras.layers.Embedding(Constant.sale_channel_2_count, 10, name='sale_ch2_emb')(sale_channel_2_input)

        total_use_days_input_embedding = tf.keras.layers.Embedding(5000, 100, name='use_days_emb')(total_use_days_input)

        # brand_flatten = tf.keras.layers.Flatten()(brand_embedding)
        # model_name_flatten = tf.keras.layers.Flatten()(model_name_embedding)
        # version_flatten = tf.keras.layers.Flatten()(version_embedding)
        user_age_flatten = tf.keras.layers.Flatten()(user_age_embedding)
        user_sex_flatten = tf.keras.layers.Flatten()(user_sex_embedding)
        user_degree_flatten = tf.keras.layers.Flatten()(user_degree_embedding)
        resident_province_flatten = tf.keras.layers.Flatten()(resident_province_embedding)
        resident_city_flatten = tf.keras.layers.Flatten()(resident_city_embedding)
        resident_city_type_flatten = tf.keras.layers.Flatten()(resident_city_type_embedding)
        # phone_log_flatten = tf.keras.layers.Flatten()(phone_log_embedding)
        # phone_raw_flatten = tf.keras.layers.Flatten()(phone_raw_embedding)
        sale_channel_1_flatten = tf.keras.layers.Flatten()(sale_channel_1_embedding)
        sale_channel_2_flatten = tf.keras.layers.Flatten()(sale_channel_2_embedding)
        total_use_days_input_flatten = tf.keras.layers.Flatten()(total_use_days_input_embedding)

        vitality_days = 30
        vitality_dimension = 10
        vitality_seq = tf.keras.layers.Input(shape=(vitality_days,), name='vatality')
        embedding_vitality = tf.keras.layers.Embedding(3, vitality_dimension)(vitality_seq)

        appid_size = 203
        days = 30
        app_dimension = appid_size * 3
        # 30: 30 days
        app_all = tf.keras.layers.Input(shape=(days, app_dimension), name='all_app')





        vitality_conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                             input_shape=(vitality_days, vitality_dimension))(embedding_vitality)
        vitality_pooling = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(vitality_conv1d)
        vitality_context_value = tf.keras.layers.Flatten()(vitality_pooling)

        appid_conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                              input_shape=(days, app_dimension))(app_all)
        appid_pooling = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(appid_conv1d)
        appid_context_value = tf.keras.layers.Flatten()(appid_pooling)

        context = tf.keras.layers.concatenate([
            # brand_flatten,
            # model_name_flatten,
            # version_flatten,
            # total_use_days_input,
            user_age_flatten, user_sex_flatten,
            user_degree_flatten, resident_province_flatten,
            resident_city_flatten, resident_city_type_flatten,
            # phone_log_flatten,
            # phone_raw_flatten,
            sale_channel_1_flatten, sale_channel_2_flatten,
            total_use_days_input_flatten,

            appid_context_value, vitality_context_value
        ], name='user_tower_context_layer')

        full_fusion_context = tf.keras.layers.Dense(128, activation='relu', name='user_tower_fusion_layer')(context)
        user_tower_embedding = tf.keras.layers.Dense(embedding_size, activation='relu', name='user_tower_layer')(full_fusion_context)
        # fusion_context = tf.keras.layers.Dropout(0.2)(full_fusion_context)
        # output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(fusion_context)


        model = tf.keras.models.Model(inputs=[
            # model_name_input,
            total_use_days_input, user_age_input, user_sex_input,
            user_degree_input, resident_province_input,
            resident_city_input, resident_city_type_input,
            sale_channel_1_input, sale_channel_2_input,
            vitality_seq, app_all
        ], outputs=user_tower_embedding, name='user_tower')
        model.summary()
        return model



    # def call(self, brand,
    #          modelname,
    #          version,
    #          total_use_days, #user
    #          user_age,  #user
    #          user_sex,  #user
    #          user_degree, #user
    #          resident_province, #user
    #          resident_city, #user
    #          resident_city_type, #user
    #          phone_log_model,
    #          phone_raw_model,
    #          sale_channel_1,  #user
    #          sale_channel_2,   #use
    #          vitality,  #user
    #          appall  # user
    #          ):
    def call(self, features):
        user_embedding = self.user_embeddings(features)
        phone_embedding = self.phone_embeddings(features)
        # user_embedding = self.user_embeddings({
        #     "user_age": user_age,
        #     "user_sex": user_sex,
        #     "user_degree": user_degree,
        #     "resident_province": resident_province,
        #     "resident_city": resident_city,
        #     "resident_city_type": resident_city_type,
        #     "sale_channel_1": sale_channel_1,
        #     "sale_channel_2": sale_channel_2,
        #     "vatality": vitality,
        #     "all_app":  appall
        # })
        # phone_embedding = self.phone_embeddings({
        #     "brand" : brand,
        #     "modelname": modelname,
        #     "version": version,
        #     "phone_log_model": phone_log_model,
        #     "phone_raw_model": phone_raw_model
        # })

        return self.ratings(tf.concat([user_embedding, phone_embedding], axis=1))


class NewPhoneModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()


        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                    # tfrs.metrics.FactorizedTopK
                    tf.keras.metrics.Accuracy(),
                    tf.keras.metrics.RootMeanSquaredError(),
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC(),
                     # tfa.metrics.F1Score(num_classes=1, threshold=0.5)
                     ]
        )

    def compute_loss(self, inputs, training=False) -> tf.Tensor:
        print(type(inputs))
        features, label = inputs[0] , inputs[1]
        rating_predictions = self.ranking_model(features)
                                                # features['brand'],
                                                # features["modelname"],
                                                # features["version"],
                                                # features["total_use_days"],
                                                # features["user_age"],
                                                # features["user_sex"],
                                                # features["user_degree"],
                                                # features["resident_province"],
                                                # features["resident_city"],
                                                # features["resident_city_type"],
                                                # features["phone_log_model"],
                                                # features["phone_raw_model"],
                                                # features["sale_channel_1"],
                                                # features["sale_channel_2"],
                                                # features["vatality"],
                                                # features["all_app"],
                                                # )

        # The task computes the loss and the metrics.
        return self.task(labels=label, predictions=rating_predictions)
