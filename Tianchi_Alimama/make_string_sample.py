#encoding=utf-8
__author__ = 'peng'



import fire, logging
import  pandas as pd
import time ,datetime
class Predict_Category_Property(object):
    def __init__(self, line):
        units  = line.split(';')

        buf = []
        for u in units:
            cate, ps = u.split(':')
            pss = ps.split(',')

            for p in pss:
                buf.append( (cate , p ))

        self.buf = buf

    def __iter__(self):

        for cate , property in self.buf:
            yield cate, property


def convdate(t):

    return time.strftime('%Y-%m-%d', time.localtime(t))
def conv_item_property_list(key, cl):
    us = cl.split(";")
    buf = []
    for u in us:
        buf.append('{}={}'.format(key, u))
    return buf

def conv_item_category_list(key,  cl):

    us = cl.split(";")
    buf = []
    for u in us:
        buf.append('{}={}'.format(key, u))
    return buf
def conv_predict_category_property(line):
    pcp = Predict_Category_Property(line)
    buf = []

    pc = []
    for cate, prop in pcp:
        pc.append(cate)
        buf.append('predict_prop={}'.format(prop))

    for c in pc:
        buf.append('predict_cate={}'.format(c))


    return buf


def main():
    """
    Index([u'instance_id', u'item_id', u'item_category_list',
       u'item_property_list', u'item_brand_id', u'item_city_id',
       u'item_price_level', u'item_sales_level', u'item_collected_level',
       u'item_pv_level', u'user_id', u'user_gender_id', u'user_age_level',
       u'user_occupation_id', u'user_star_level', u'context_id',
       u'context_timestamp', u'context_page_id', u'predict_category_property',
       u'shop_id', u'shop_review_num_level', u'shop_review_positive_rate',
       u'shop_star_level', u'shop_score_service', u'shop_score_delivery',
       u'shop_score_description', u'is_trade'],
      dtype='object')



    数据拼接格式为 "category_0;category_1;category_2"，其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目

    :return:
    """
    input = '/Users/baipeng/contest/alimama/round1_ijcai_18_train_20180301.txt'
    input = '/Users/baipeng/contest/alimama/h1000.train'
    output = './data/string.sample'

    logging.info('begin to make string sample')
    df = pd.read_csv(input,sep=' ',header=0)
    with open(output, 'w') as f:
        for row in df.itertuples():
            buf = []

            date=convdate(row.context_timestamp)
            buf.append('2 insid={} date={} {}'.format(row.instance_id, date, row.is_trade ))

            buf.append('item_id={}'.format(row.item_id))

            buf.extend( conv_item_category_list('item_category', row.item_category_list))

            buf.extend( conv_item_property_list('item_property', row.item_property_list))

            buf.append('item_brand_id={}'.format(row.item_brand_id))
            buf.append('item_city_id={}'.format(row.item_city_id))
            buf.append('item_price_level={}'.format(row.item_price_level))
            buf.append('item_sales_level={}'.format(row.item_sales_level))
            buf.append('item_collected_level={}'.format(row.item_collected_level))
            buf.append('item_pv_level={}'.format(row.item_pv_level))

            # ================= 覆盖度过滤之后 才能直接使用user_id
            buf.append('user_id={}'.format(row.user_id))
            buf.append('user_gender_id={}'.format(row.user_gender_id))
            buf.append('user_age_level={}'.format(row.user_age_level))
            buf.append('user_occupation_id={}'.format(row.user_occupation_id))
            buf.append('user_star_level={}'.format(row.user_star_level))
            buf.append('context_page_id={}'.format(row.context_page_id))

            buf.extend(conv_predict_category_property(row.predict_category_property))


            buf.append('shop_id={}'.format(row.shop_id))
            buf.append(u'shop_review_num_level={}'.format(row.shop_review_num_level))
            buf.append(u'shop_review_positive_rate={}'.format(row.shop_review_positive_rate))
            buf.append(u'shop_star_level={}'.format(row.shop_star_level))
            buf.append(u'shop_score_service={}'.format(row.shop_score_service))
            buf.append(u'shop_score_delivery={}'.format(row.shop_score_delivery))
            buf.append(u'shop_score_description={}'.format(row.shop_score_description))




            f.write(' '.join(buf) + '\n')



    logging.info('make string sample done')









if __name__ == '__main__':
    fire.Fire()
