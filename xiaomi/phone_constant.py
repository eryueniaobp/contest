# encoding=utf-8
from typing import List


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

