#encoding=utf-8
from scipy import sparse
import re
class AppVectorEncoder(object):
    def __init__(self,filterBuf=None):
        self.n_size = 0
        self.idmap = {}
        self.filterBuf = filterBuf
    def _include(self, ent):
        if self.filterBuf is None or ent in self.filterBuf:
            return True
        return False
    def fit(self, X):
        for row in X:
            units = re.split("\\s+", row)
            for unit in units:
                if unit == '-1': unit = 'null:0:0'
                ent, v1, v2  = unit.split(":")
                if self._include(ent):
                    if ent not in self.idmap:
                        cur  = len(self.idmap) * 2
                        self.idmap[ent] =  ( 1 + cur , 2+cur)




    def size(self):
        return len(self.idmap) * 2


    def feature_names_(self,colname):
        buf = [(0,'{}=_holder'.format(colname))]
        for k in self.idmap:
            id1, id2 = self.idmap[k]
            buf.append((id1, '{}={}'.format(colname,k)))
            buf.append((id2, '{}={}'.format(colname,k)))
        buf = sorted(buf, key=lambda x: x[0])
        return [ i[1] for i in buf ]


    def transform(self, X):
        """

        :param X:
        :return:  sparse matrix.
        """
        data = []
        indices = []
        indptr= [0]  # row-i  indptr[i]:indptr[i+1]

        n_row = 0
        n_col = self.size() + 1
        for row in X:
            n_row += 1
            units = re.split("\\s+", row)
            buf = [(0,0)]
            for unit in units:
                if unit == '-1': unit = 'null:0:0'
                ent, v1,v2 = unit.split(":")
                v1,v2 = float(v1),float(v2)

                if ent  in self.idmap:
                    ind1,ind2 = self.idmap[ent]
                    buf.append((ind1, v1))
                    buf.append((ind2, v2))
            # a = (1,2)
            if len(buf) > 0 :
                buf = sorted(buf, key=lambda x : x[0]  )

                for ind, val in buf:
                    indices.append(ind)
                    data.append(val)

                indptr.append(len(data))

        return sparse.csr_matrix((data, indices, indptr),shape=(n_row,n_col), dtype=float)

if __name__ == '__main__':
    data = [
        "a:1:2 b:2:3",
        "a:3:4 c:4:5"
    ]
    enc = AppVectorEncoder()
    enc.fit(data)

    print(enc.transform(data).toarray())
