# encoding=utf-8
import scipy.sparse as sp
import numpy as np
def _load_svmlight_file(f, n_feature ,  mini_batch=200000):

    data = []
    indices = []
    indptr = [0]
    query = []
    aidbuf  =[]

    labels = []


    qid_prefix='qid'
    COLON = ':'

    lnum = 0

    for line in f:
        # skip comments
        line_parts = line.split()
        if len(line_parts) == 0:
            continue

        target, qid, aid, features = line_parts[0], line_parts[1], line_parts[2], line_parts[3:]

        labels.append (  float(target) )

        prev_idx = -1
        n_features = len(features)
        if n_features and features[0].startswith(qid_prefix):
            _, value = features[0].split(COLON, 1)


            query.append( int(value) )
            features.pop(0)
            n_features -= 1

        _, value = qid.split(COLON, 1)
        query.append(int(value))
        _, value = aid.split(COLON, 1)
        aidbuf.append(int(value))





        for i in xrange(0, n_features):
            idx_s, value = features[i].split(COLON, 1)
            idx = int(idx_s)

            if idx <= prev_idx:
                raise ValueError("Feature indices in SVMlight/LibSVM data "
                                 "file should be sorted and unique.")

            indices.append(idx)

            data.append(float(value))

            prev_idx = idx
        indptr.append(len(data))
        lnum +=1

        if lnum >= mini_batch :

            shape =(lnum, n_feature)

            X = sp.csr_matrix((data, indices, indptr), shape)
            X.sort_indices()

            yield ( X , np.array(labels), query, aidbuf)
            # clear data
            lnum = 0

            data = []
            indices = []
            indptr = [0]
            query = []
            labels = []
            aidbuf = []
    if lnum > 0:
        #尾巴上的数据共享.
        shape = (lnum, n_feature)

        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()

        yield (X, np.array(labels), query, aidbuf)

        lnum = 0

        data = []
        indices = []
        indptr = [0]
        query = []
        labels = []
        aidbuf = []





