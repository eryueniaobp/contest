# encoding=utf-8


import  pandas as pd
import  logging,argparse
import pickle
import os
import numpy as np

import scipy.stats as ss

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='fac.log',
                    filemode='a')


def build_map(df):
    toolmap_pickle = 'toolmap.pickle'
    if os.path.exists(toolmap_pickle):
        logging.info('load {0}'.format(toolmap_pickle))
        with open(toolmap_pickle, 'r') as f:
            map = pickle.load(f)

        return map
    else:
        cols = df.columns.values

        colbuf = []
        for i in cols:
            if 'tool' in i.lower():
                colbuf.append(i)


        logging.info(colbuf)

        map = {}
        for key in colbuf:
            tool_id = df[key].unique()

            toolidmap = {}
            for id in tool_id:
                if id not in toolidmap:
                    toolidmap[id] = len(toolidmap) + 1
            print toolidmap
            map[key] = toolidmap

        with open(toolmap_pickle,'w') as f:
            pickle.dump(map,f)
            logging.info('dump {0}'.format(toolmap_pickle))
        return map

def convert_with_col_cluster():
    """
    根据聚类的情况
    :return:
    """
    logging.info('conver_with_cluster begin')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        input = 'train.xlsx'
        output = 'train.cls.txt'
        hasY = True
    elif args.mode == 'testA':
        input = 'testA.xlsx'
        output = 'testA.cls.txt'
        hasY = False
    elif args.mode == 'testB':
        input = 'testB.xlsx'
        output = 'testB.cls.txt'
        hasY = False
    else:
        raise 'check your parameter'

    cluster = pd.read_csv('cluster.txt',header=0)
    dvals = cluster.values

    cmap  ={}
    for row in dvals:
        if row[0] in cmap:
            cmap[row[0]].append( row[1])
        else:
            cmap[row[0]] = [ row[1] ]

    df = pd.read_excel(input, header=0)  # type:pd.DataFrame
    df = df.fillna(0)

    r, c = df.shape
    logging.info('row:col = {0}:{1}'.format(r, c))

    dvals = df.values

    cols = df.columns.values
    colmap = {}
    for k in cols:
        colmap[k] = len(colmap)

    def make_stat(colbuf):
        return [
            np.min(colbuf),
            np.max(colbuf),
            np.sum(colbuf),
            np.median(colbuf),
            np.mean(colbuf),
            np.mean(np.absolute( colbuf - np.mean(colbuf))), #mad
            ss.skew(colbuf),
        ]
    def make_buf(buf, statbuf, start):
        for i in statbuf:
            buf.append('{0}:{1}'.format(start , i ))
            start +=1

        return start
    with open(output, 'w') as f:
        for i in range(r):

            # print i
            buf = []
            row =dvals[i]
            if hasY:
                y = row[c-1]
            else:
                y = 0.
            buf.append('{0}'.format(y))

            start = 1
            for cluster in sorted(cmap.keys()):

                colbuf = []
                colnames = cmap[cluster]
                for col in colnames:
                    icol = colmap[col]
                    colbuf.append(row[icol])

                colbuf =  np.array(colbuf)

                statbuf =  make_stat(colbuf)

                start =  make_buf(buf, statbuf, start)

            record = ' '.join(buf)
            f.write(record+'\n')
    logging.info('convert_with_cluster done')



def main():
    """

    TRAIN SAMPLE
    TOOL_ID
    Tool
    TOOL_ID (#1)
    TOOL_ID (#2)
    TOOL_ID (#3)
    Tool (#1)
    Tool (#2)
    tool
    tool (#1)
    TOOL
    TOOL (#1)
    Tool (#3)
    TOOL (#2)

    TEST SAMPLE   看起来是严格匹配的 .
    TOOL_ID
    Tool
    TOOL_ID (#1)
    TOOL_ID (#2)
    TOOL_ID (#3)
    Tool (#1)
    Tool (#2)
    tool
    tool (#1)
    TOOL
    TOOL (#1)
    Tool (#3)
    TOOL (#2)

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        input = 'train.xlsx'
        output = 'train.txt'
        hasY = True
    elif args.mode == 'testA':
        input = 'testA.xlsx'
        output = 'testA.txt'
        hasY = False
    elif args.mode == 'testB':
        input = 'testB.xlsx'
        output = 'testB.txt'
        hasY = False
    else:
        raise 'check your parameter'

    df = pd.read_excel(input, header=0) #type:pd.DataFrame

    toolmap = build_map(df)

    df = df.fillna(0) #fill error


    r, c = df.shape
    logging.info('row:col = {0}:{1}'.format(r,c))

    dvals = df.values

    cols = df.columns.values


    with open('daycol.pickle','r') as f:
        daycols = pickle.load(f)
        logging.info('daycols = {0}'.format(len(daycols)))

    fidbuf = []
    fidset = set()

    with open(output, 'w') as f:
        for i in range(r):
            buf = []
            row = dvals[i]

            if hasY:
                y = row[c - 1]

            else:
                y = 0.

            buf.append('{0}'.format(y))





            start = 0
            for key in toolmap:
                icol = cols.tolist().index(key)
                colval = row[icol]
                if colval not in toolmap[key]:
                    print 'key =' , key ,  '  colval=',colval
                else:
                    fid = start + toolmap[key][colval]
                    buf.append('{0}:1'.format(fid))
                start += len(toolmap[key])


            start +=1
            for j in range(2, c):
                if 'tool' not in cols[j].lower() and 'y' not in cols[j].lower() and cols[j] not in daycols:
                    val = row[j]

                    buf.append("{0}:{1}".format(start, val))
                    if start not in fidset:
                        fidbuf.append((cols[j], start))
                        fidset.add(start)
                    start +=1

            record = ' '.join(buf)

            f.write(record+'\n')

    pd.DataFrame({
        'col': [i[0] for i in fidbuf] ,
        'fid': [i[1] for i in fidbuf]
    }).to_csv('fid.map',index=False)

    logging.info('done')




def check_columns():

    train = pd.read_excel('train.xlsx', header=0)
    testA = pd.read_excel('testA.xlsx',header=0)
    testB = pd.read_excel('testB.xlsx',header=0)

    assert len(train.columns.values)-1  == len(testA.columns.values) and len(train.columns.values) -1  == len(testB.columns.values)

    col1 = train.columns.values
    colA = testA.columns.values
    colB = testB.columns.values

    a = [0 if col1[i] == v else 1 for i , v in enumerate(colA)]
    b = [0 if col1[i] == v else 1 for i, v in enumerate(colB)]
    assert sum(a) == 0 and sum(b) == 0


def check_time_columns():
    train = pd.read_excel('train.xlsx', header=0)
    dvals = train.values
    cols = train.columns.values
    buf = []
    for row in dvals:
        for i , v  in enumerate(row):
            if v > 1e6 and cols[i] not in buf:
                buf.append(cols[i])
    print len(buf)
    print buf

    with open('daycol.pickle','w') as f:
        pickle.dump(buf,f)


def convert():
    train = pd.read_excel('train.xlsx', header=0) #type: pd.DataFrame

    train.to_csv('train.csv',index=False)

if __name__ == '__main__':
    # main()

    convert_with_col_cluster()
    # check_time_columns()
    # check_columns()
    # convert()
