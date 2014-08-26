import numpy as np
from itertools import izip
from collections import Counter


class StratifiedTrainIterator(object):
    def __init__(self, dataset, batch_size):
        self.rng = np.random.RandomState(4697596)
        x, y = dataset
        self.data = {}
        if y.ndim == 1:
            clazzes = np.unique(y)
            boolean_indexes = [y == clazz for clazz in clazzes]
        else:
            clazzes = range(y.shape[1])
            boolean_indexes = [y[:, clazz] == 1 for clazz in clazzes]

        self.idx = {}
        for clazz, boolean_idxs in izip(clazzes, boolean_indexes):
            self.data[clazz] = {'x': x[boolean_idxs],
                                'y': y[boolean_idxs],
                                'idx': np.arange(np.sum(boolean_idxs))
            }
        self.batch_size = batch_size
        self.__restart()

    def __iter__(self):
        return self

    def get_clazzes_batch(self, batch_size):
        clazzes_batch = []
        clazzes_stream = []
        for clazz in self.data:
            if self.idx[clazz] != len(self.data[clazz]['idx']):
                clazzes_batch.append(clazz)
                batch_size -= 1
                clazzes_stream += [clazz] * (len(self.data[clazz]['idx']) - self.idx[clazz] - 1)
            else:
                return []
        self.rng.shuffle(clazzes_stream)
        clazzes_batch += clazzes_stream[:batch_size]
        return clazzes_batch

    def next(self):
        clazzes = self.get_clazzes_batch(self.batch_size)
        if len(clazzes) == self.batch_size:
            x = None
            y = None
            for clazz, clazz_batch_size in Counter(clazzes).iteritems():
                idx = self.data[clazz]['idx'][self.idx[clazz]:self.idx[clazz] + clazz_batch_size]
                if x is None:
                    x = self.data[clazz]['x'][idx]
                    y = self.data[clazz]['y'][idx]
                else:
                    x = np.concatenate((x, self.data[clazz]['x'][idx]))
                    y = np.concatenate((y, self.data[clazz]['y'][idx]))
                self.idx[clazz] += clazz_batch_size
            return x, y
        else:
            self.__restart()
            raise StopIteration

    def __restart(self):
        self.idx = {}
        for clazz in self.data:
            self.rng.shuffle(self.data[clazz]['idx'])
            self.idx[clazz] = 0