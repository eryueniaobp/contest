# encoding=utf-8
from typing import List
import tensorflow as tf
import os, sys, glob, re
from collections import namedtuple
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# print(tf.executing_eagerly())
ANN = namedtuple('ANN', ('id', 'entity_type', 's', 'e', 'span'))
SIMPLE_ANN = namedtuple('SIMPLE_ANN', ('entity_type','span'))
rng = np.random.RandomState(0)

def get_simple_anns(annfile):
    """
    sorted anns
    :param annfile:
    :return:
    """
    buf = []
    with open(annfile, 'r') as f:
        for line in f:
            try:
                id, entity_type, s, e, span = re.split('\\s+', line.strip(), maxsplit=4)
                s, e = int(s), int(e)
                anno = SIMPLE_ANN(entity_type=entity_type, span=span)
                buf += [anno]
            except:
                input("press any key")
    return buf
def extract_all_annotations(anns):
    s = set()
    for annfile in tqdm(anns):
        buf = get_simple_anns(annfile)
        s = s.union(set(buf))

    return s
def get_txt(txtfile):
    with open(txtfile, 'r') as f:
        txt = f.readlines()
        assert len(txt) == 1
        txt = txt[0]
    return txt
def build_ANN(ann):
    try:
        id, entity_type, s, e, span = re.split('\\s+', ann.strip(),maxsplit=4)
    except Exception as e:
        print(e)
        print(ann)
        input("Press any key to")

    s, e = int(s), int(e)
    return ANN(id=id, entity_type=entity_type, s=s, e=e, span=span)

def get_anns(annfile):
    """
    sorted anns
    :param annfile:
    :return:
    """
    buf = []
    with open(annfile, 'r') as f:
        for line in f:
            anno = build_ANN(line)
            buf += [anno]
    buf = sorted(buf, key=lambda x: x.s)
    return buf

def sample_from_simple_anns(entity_type, simple_anns):
    def is_ok(ann):
        if ann.entity_type == entity_type:
            return True
        else:
            return False
    buf = list(filter(is_ok, simple_anns))
    return buf[rng.randint(0, len(buf), 1)[0]]



def augment_data(txt_path, ann_path, output_path,
                augment_rate=0.3,
                shuffle_segment_rate=0.4,
                epochs=10):

    os.makedirs(output_path, exist_ok=True)

    anns = glob.glob('./data/train/*.ann')
    anns = sorted(anns, key=lambda x: int(x.split('/')[3][:-4]))

    last_num = int(anns[-1].split('/')[3][:-4])

    txts = glob.glob('./data/train/*.txt')
    txts = sorted(txts, key=lambda x: int(x.split('/')[3][:-4]))
    print(list(zip(anns, txts)))
    simple_anns = extract_all_annotations(anns)

    # print(simple_anns)
    # input("press any kety")



    # for i in tqdm(range(1000000)):
    #     pass
    # input('Press')

    for i in range(epochs):
        print('epochs = {} , last_num = {}'.format(i, last_num))
        for annfile, txtfile in tqdm(zip(anns, txts)):
            # print(annfile, txtfile)
            annbuf = get_anns(annfile)
            txt    = get_txt(txtfile)

            incr = [0]

            new_annbuf = []
            for ann in annbuf: #type: ANN
                if rng.random() < augment_rate:
                    if rng.random() < shuffle_segment_rate:
                        """
                        shuffle inside. 
                        """
                        span = list(ann.span)
                        np.random.shuffle(span)
                        new_span = ''.join(span)

                        # print(new_span, ann.span)
                        # input("Press nay ")
                        new_annbuf.append(ann._replace(span=new_span))
                        incr.append(incr[-1])
                    else:

                        new_simple_ann = sample_from_simple_anns(ann.entity_type, simple_anns)

                        new_annbuf.append(ann._replace(span=new_simple_ann.span))
                        diff = len(new_simple_ann.span)  - len(ann.span)
                        incr.append(incr[-1] + diff)
                else:
                    incr.append(incr[-1])
                    new_annbuf.append(ann._replace())
            # correction  by incr idx

            # print(annfile, incr[1:])
            # k = 0
            # for x, y in zip(annbuf, new_annbuf):
            #     print(x, y , incr[1:][k])
            #     k +=  1
            # print(new_annbuf)
            prev = 0

            buf2 = []
            for newann, incrdiff in zip(new_annbuf, incr[1:]):

                s = newann.s + prev
                e = newann.e + incrdiff
                buf2.append(newann._replace(s=s, e = e  ))
                prev = incrdiff

            new_annbuf = buf2

            # correction txgt .
            sbuf = []
            p = 0
            for newann, ann in zip(new_annbuf, annbuf):
                sbuf  += txt[p:ann.s]
                sbuf += newann.span
                p = ann.e

            newtxt = ''.join(sbuf)

            last_num += 1
            save_it(newtxt, new_annbuf,'{}/{}'.format(output_path, last_num))


def check_ann(txt, anns):
    for ann in anns:
        assert ann.span == txt[ann.s:ann.e]
def save_it(txt, anns, prefix):

    check_ann(txt, anns)

    with open(prefix + '.txt', 'w') as f :
        f.write(txt)
    with open(prefix +'.ann', 'w') as wf:
        for ann in anns:
            wf.write('{}\t{} {} {}\t{}\n'.format(ann.id, ann.entity_type, ann.s, ann.e, ann.span))


if __name__ == '__main__':
    ann_path, txt_path = './data/train/*.ann', './data/train/*.txt'
    output_path = './data/train_ext'
    augment_data(txt_path, ann_path, output_path,
                 augment_rate=0.3,
                 shuffle_segment_rate=0.4)
    pass
