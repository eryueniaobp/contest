#encoding=utf-8
#一大类特征
from scipy import sparse
import re
class VectorEncoder(object):
    def __init__(self):
        self.n_size = 0
        self.idmap = {}
    def fit(self, X):
        for row in X:
            units = re.split("\\s+", row)
            for unit in units:
                if unit == '-1': unit = 'null:0'
                ent, value = unit.split(":")
                if ent not in self.idmap:
                    self.idmap[ent] =  1 + len(self.idmap)




    def size(self):
        return len(self.idmap)

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
            buf = []
            for unit in units:
                if unit == '-1': unit = 'null:0'
                ent, value = unit.split(":")
                value = float(value)

                if ent  in self.idmap:
                    ind = self.idmap[ent]
                    buf.append((ind, value))
            # a = (1,2)
            buf = sorted(buf, key=lambda x : x[0]  )

            for ind, val in buf:
                indices.append(ind)
                data.append(val)

            indptr.append(len(data))

        return sparse.csr_matrix((data, indices, indptr),shape=(n_row,n_col), dtype=float)

if __name__ == '__main__':
    data = [
        "a:1 b:2",
        "a:3 c:4"
    ]
    enc = VectorEncoder()
    enc.fit(data)

    print(enc.transform(data).toarray())
