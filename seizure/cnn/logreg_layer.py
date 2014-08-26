import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class LogisticRegressionLayer(object):
    def __init__(self, rng, input, n_in, training_mode, dropout_prob):
        print 'logistic regression'
        W_values = rng.normal(size=(n_in, 1), scale=0.1, loc=0)
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((1,), dtype='float32'), name='b', borrow=True)
        inv_dropout_prob = np.float32(1.0 - dropout_prob)
        self.p_y_given_x = ifelse(T.eq(training_mode, 1),
                                  self.logit(T.dot(self._dropout(rng, input, dropout_prob), self.W) + self.b),
                                  self.logit(T.dot(input, inv_dropout_prob * self.W) + self.b))
        self.p_y_given_x = self.p_y_given_x.T
        self.y_pred = T.round(self.p_y_given_x)
        self.params = [self.W, self.b]

    def cross_entropy_cost(self, y):
        return -T.mean(y * T.log(self.p_y_given_x + 10e-6) + (1 - y) * T.log(1 - self.p_y_given_x + 10e-6))

    def auc_cost(self, y, kappa=0.9, tau=2):
        f_pos = T.nonzero_values(y * self.p_y_given_x)
        f_neg = T.nonzero_values((1 - y) * self.p_y_given_x)
        diff = f_pos.T.dimshuffle(0, 'x') - f_neg.T.dimshuffle('x', 0)
        r = (-(diff - kappa)) ** tau * (diff < kappa)
        auc = T.mean(r)
        return auc

    def confusion_matrix(self, y):
        tp = T.and_(T.eq(y, 1), T.eq(self.y_pred, 1)).sum()
        tn = T.and_(T.eq(y, 0), T.eq(self.y_pred, 0)).sum()
        fp = T.and_(T.eq(y, 0), T.eq(self.y_pred, 1)).sum()
        fn = T.and_(T.eq(y, 1), T.eq(self.y_pred, 0)).sum()
        return [tp, tn, fp, fn]

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output

    @staticmethod
    def logit(x):
        return 1.0 / (1.0 + T.exp(-x))
