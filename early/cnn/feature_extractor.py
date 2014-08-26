import theano
from early.cnn.hidden_layer import HiddenLayer
from early.cnn.conv_layer import ConvPoolLayer
from global_pooling_layer import GlobalPoolLayer

class FeatureExtractor(object):
    def __init__(self, rng, input, nkerns, recept_width, pool_width, stride,  training_mode, dropout_prob, activation,
                 weights_variance, n_timesteps, dim):
        n_layers = len(nkerns)
        if n_layers == 2:
            self.layer0 = ConvPoolLayer(rng=rng, input=input,
                                        image_shape=(None, 1, dim, n_timesteps),
                                        poolsize=pool_width,
                                        filter_shape=(nkerns[0], 1, 1, recept_width),
                                        activation=activation,
                                        weights_variance=weights_variance)

            layer1_input = self.layer0.output.flatten(2)
            self.layer1 = HiddenLayer(rng=rng, input=layer1_input,
                                      n_in=nkerns[0] * dim, n_out=nkerns[1],
                                      training_mode=training_mode,
                                      dropout_prob=dropout_prob, activation=activation,
                                      weights_variance=weights_variance)

            self.output = self.layer1.output
            self.params = self.layer0.params + self.layer1.params

        elif n_layers == 3:
            self.layer0 = ConvPoolLayer(rng, input=input,
                                        image_shape=(None, 1, dim, n_timesteps),
                                        filter_shape=(nkerns[0], 1, 1, recept_width[0]),
                                        poolsize=(1, pool_width[0]), activation=activation,
                                        weights_variance=weights_variance, subsample=(1, stride[0]))

            input_layer1_width = ((n_timesteps - recept_width[0]) / stride[0] + 1) / pool_width[0]
            self.layer1 = ConvPoolLayer(rng, input=self.layer0.output,
                                        image_shape=(None, nkerns[0], dim, input_layer1_width),
                                        filter_shape=(nkerns[1], nkerns[0], 1, recept_width[1]),
                                        poolsize=(1, pool_width[1]), activation=activation,
                                        weights_variance=weights_variance, subsample=(1, stride[1]))

            self.glob_pool = GlobalPoolLayer(input=self.layer1.output)
            layer2_input = self.glob_pool.output.flatten(2)
            input_layer2_size = 3
            self.layer2 = HiddenLayer(rng=rng, input=layer2_input,
                                      n_in=nkerns[1] * dim * input_layer2_size, n_out=nkerns[2],
                                      training_mode=training_mode,
                                      dropout_prob=dropout_prob, activation=activation,
                                      weights_variance=weights_variance)

            # layer2_input = self.layer1.output.flatten(2)
            #
            # input_layer2_size = ((input_layer1_width - recept_width[1]) / stride[1] + 1) / pool_width[1]
            # self.layer2 = HiddenLayer(rng=rng, input=layer2_input,
            #                           n_in=nkerns[1] * dim * input_layer2_size, n_out=nkerns[2],
            #                           training_mode=training_mode,
            #                           dropout_prob=dropout_prob, activation=activation,
            #                           weights_variance=weights_variance)

            self.output = self.layer2.output
            self.params = self.layer0.params + self.layer1.params + self.layer2.params
