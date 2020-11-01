# encoding=utf-8
from preprocess import build_dataset
from kashgari.corpus import ChineseDailyNerCorpus
import kashgari
from kashgari.tasks.labeling import BiLSTM_CRF_Model
import os
import keras
from kashgari.embeddings import BertEmbedding
from kashgari.callbacks import EvalCallBack
import datetime
import numpy as np
import sys
import argparse
import tensorflow as tf

## roberta
bert_path = '/root/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12'
bert_embed = BertEmbedding(bert_path, trainable=True)


class BIODataGenerator:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        pass

    def forfit(self):
        while True:
            batch_X = []
            batch_y = []
            with open(self.data_path, 'r') as f:
                X, Y = [], []
                for line in f:
                    if len(line.strip()) == 0:
                        batch_X.append(X)
                        batch_y.append(Y)
                        X, Y = [], []
                        if len(batch_X) == self.batch_size:
                            yield batch_X, batch_y
                            batch_X, batch_y = [], []
                        pass
                    else:
                        x, y = line.strip().split('\t', 1)
                        X.append(x)
                        Y.append(y)
                        if len(batch_X) == self.batch_size:
                            yield batch_X, batch_y
                            batch_X = []
                            batch_y = []
                if len(batch_X) > 0:
                    yield batch_X, batch_y
                    batch_X = []
                    batch_y = []


class DataGenerator:
    def __init__(self, data_path, batch_size, start, span):
        self.batch_size = batch_size
        self.start = start
        self.span = span
        self.data_path = data_path

    def forfit(self):
        while True:
            batch_X = []
            batch_y = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    rnum = np.random.random()
                    if rnum >= self.start and rnum < self.start + self.span:
                        X, y = line.strip().split('^')
                        xs = X.split('@')
                        ys = y.split('\t')
                        batch_X.append(xs)
                        batch_y.append(ys)

                        if len(batch_X) == self.batch_size:
                            yield batch_X, batch_y
                            batch_X = []
                            batch_y = []
                if len(batch_X) > 0:
                    yield batch_X, batch_y
                    batch_X = []
                    batch_y = []


def parse_line(line):
    # tf.print(line)
    bs = tf.strings.split(line, sep='^')
    Ys = tf.strings.split(bs[1], sep='\t')
    # tf.while_loop
    Xs = tf.strings.split(bs[0], sep='@')
    # tf.print(Xs.shape, Ys.shape)
    # assert Xs.shape[0] == Ys.shape[0]
    # return Xs, Ys
    return Xs, Ys


def build_dataset(path):
    file_paths = tf.data.Dataset.list_files(path)
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda line: parse_line(line))
    return dataset


def train_ten_fold(data_path):
    """

    :param train_path:
    :param model_path:
    :return:
    """

    base_checkpoint_filepath = 'checkpoint_fold'
    base_model_path = 'model_fold'
    for i in range(10):
        start = 0.1 * i
        span = 0.1
        checkpoint_filepath = base_checkpoint_filepath + '{}/ner'.format(i)
        model_path = base_model_path + '{}'.format(i)
        train_it2(data_path, checkpoint_filepath, model_path, start, span)


from collections import Counter


class Evaluator(tf.keras.callbacks.Callback):
    def __init__(self, kash_model, path, valid_x, valid_y, step=5):
        super(Evaluator, self).__init__()
        self.best_val_f1 = 0
        self.kash_model = kash_model
        self.x_data = valid_x
        self.y_data = valid_y
        self.path = path
        self.step = step

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.step == 0:
            report = self.kash_model.evaluate(self.x_data,  # type: ignore
                                              self.y_data)
            f1, precision, recall = report['f1-score'], report['precision'], report['recall']
            if f1 >= self.best_val_f1:
                self.best_val_f1 = f1
                self.model.save_weights('{}/best_model.weights'.format(self.path))
            print(
                'in Evaluator, epoch= %s , valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                (epoch, f1, precision, recall, self.best_val_f1)
            )


def train_it2(train_path, checkpoint_filepath, model_path, start, span):
    data_generator = BIODataGenerator(train_path, 100000000)
    Xs, ys = data_generator.forfit().__next__()

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    rng = np.random.RandomState(0)
    k = 0
    for x, y in zip(Xs, ys):
        # x = [str(i, 'utf-8') for i in x]
        # y = [str(i, 'utf-8') for i in y]
        rnum = rng.rand()
        k += 1
        if rnum < start or rnum >= start + span:
            train_x += [x]
            train_y += [y]
        else:
            valid_x += [x]
            valid_y += [y]
    # dataset = dataset.batch(32)
    print('====' * 8)
    print('total = ', k)
    print('start , span = ', (start, span))
    print('len train = ', len(train_x))
    # checkpoint_filepath = './checkpoint'
    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.mkdir(os.path.dirname(checkpoint_filepath))

    # train_x, train_y = ChineseDailyNerCorpus.load_data('train')
    # test_x, test_y = ChineseDailyNerCorpus.load_data('test')
    # valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)
    #train_x, train_y = train_x[:1000], train_y[:1000]
    #valid_x, valid_y = valid_x[:200], valid_y[:200]

    model = BiLSTM_CRF_Model(bert_embed, sequence_length=128)
    eval_callback = Evaluator(model, checkpoint_filepath, valid_x, valid_y)
    early_stop = keras.callbacks.EarlyStopping(patience=10)
    reduse_lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    # eval_callback = EvalCallBack(kash_model=model,
    #                              x_data=valid_x,
    #                              y_data=valid_y,
    #                              step=1)

    model.fit(train_x, train_y, valid_x, valid_y, batch_size=64, epochs=100,
              callbacks=[early_stop, eval_callback, reduse_lr_callback])
    model.save(model_path)


def train_it(train_path, checkpoint_filepath, model_path, start, span):
    dataset = build_dataset(train_path)
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    rng = np.random.RandomState(0)
    k = 0
    for x, y in dataset.as_numpy_iterator():
        x = [str(i, 'utf-8') for i in x]
        y = [str(i, 'utf-8') for i in y]
        rnum = rng.rand()
        k += 1
        if rnum < start or rnum >= start + span:
            train_x += [x]
            train_y += [y]
        else:
            valid_x += [x]
            valid_y += [y]
    # dataset = dataset.batch(32)
    print('====' * 8)
    print('total = ', k)
    print('start , span = ', (start, span))
    print('len train = ', len(train_x))
    # checkpoint_filepath = './checkpoint'
    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.mkdir(os.path.dirname(checkpoint_filepath))

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)

    model = BiLSTM_CRF_Model(bert_embed, sequence_length=100)
    evaluator = Evaluator(model, checkpoint_filepath, valid_x, valid_y)
    model.fit(train_x, train_y, valid_x, valid_y, batch_size=64, epochs=20, callbacks=[evaluator])
    model.save(model_path)


def predict_it(test_path, model_path, output_path):
    bert_embed = BertEmbedding(bert_path)
    dataset = build_dataset(test_path)
    test_x, test_y = [], []
    for x, y in dataset.as_numpy_iterator():
        x = [str(i, 'utf-8') for i in x]
        y = [str(i, 'utf-8') for i in y]
        test_x += [x]
        test_y += [y]

    # 加载保存模型
    loaded_model = kashgari.utils.load_model('saved_ner_model')
    # loaded_model = tf.keras.models.load_model(model_path)
    loaded_model.tf_model.load_weights(model_path)
    # 使用模型进行预测
    test_y = loaded_model.predict(test_x)
    with open(output_path, 'w') as f:
        for y in test_y:
            f.write('\t'.join(y) + '\n')
    print('predict_it done {} {} {}'.format(test_path, model_path, output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='task')
    train = subparser.add_parser('train')
    predict = subparser.add_parser('predict')
    predict = subparser.add_parser('tmp')
    args = parser.parse_args()

    model_path = 'saved_ner_model'
    checkpoint_filepath = './checkpoint3/ckpt'
    if args.task == 'train':
        # train_path = './data/train.clean/round1_train.*'
        train_path = './data/train.v2/round1_train.v2.txt'
        train_ten_fold(train_path)
    elif args.task == 'predict':
        test_path = './data/test.clean/round1_test.*'
        output_path = './data/submit/raw/round1_test.{}.ann'.format(datetime.datetime.now().strftime('%Y%m%d.%H%M'))

        import os

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predict_it(test_path, './checkpoint3/ckpt', output_path)
    else:
        test_path = './data/test.tmp/round1_test.*'
        output_path = './data/submit/tmp'
        import os

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        predict_it(test_path, model_path, output_path + '/round1_test.ann')

#
# # 验证模型，此方法将打印出详细的验证报告
# model.evaluate(train_x, train_y)
#
# # 保存模型到 `saved_ner_model` 目录下
# model.save('saved_ner_model')
