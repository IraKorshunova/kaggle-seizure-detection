import theano
import theano.tensor as T


class GlobalPoolLayer(object):
    def __init__(self, input):

        avg_out = input.mean(3)
        max_out = input.max(3)
        l2_out = T.sqrt((input ** 2).mean(3))
        self.output = T.concatenate([avg_out, max_out, l2_out], axis=2)
