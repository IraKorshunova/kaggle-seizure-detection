import numpy as np
import theano
import theano.tensor as T
from theano import Param
from early.cnn_trainer.random_train_iterator import RandomTrainIterator
from early.cnn.softmax_layer import SoftmaxLayer
from early.cnn.feature_extractor import FeatureExtractor
from sklearn.metrics import roc_curve, auc


class ConvNet(object):
    def __init__(self, nkerns, recept_width, pool_width, stride, dropout_prob, l2_reg, training_batch_size, activation,
                 weights_variance, n_timesteps, dim):

        self.training_batch_size = training_batch_size
        rng = np.random.RandomState(23455)

        self.training_mode = T.iscalar('training_mode')
        self.x = T.matrix('x')
        self.y = T.bvector('y')
        self.batch_size = theano.shared(training_batch_size)

        self.input = self.x.reshape((self.batch_size, 1, dim, n_timesteps))

        self.feature_extractor = FeatureExtractor(rng, self.input, nkerns, recept_width, pool_width, stride,
                                                  self.training_mode,
                                                  dropout_prob[0],
                                                  activation, weights_variance, n_timesteps, dim)

        self.classifier = SoftmaxLayer(rng=rng, input=self.feature_extractor.output,
                                       n_in=nkerns[-1], n_out=3,
                                       training_mode=self.training_mode, dropout_prob=dropout_prob[1])

        self.params = self.feature_extractor.params + self.classifier.params

        # ---------------------- BACKPROP
        self.cost = self.classifier.cross_entropy_cost(self.y)
        L2_sqr = sum((param ** 2).sum() for param in self.params[::2])
        self.grads = T.grad(self.cost + l2_reg * L2_sqr, self.params)
        self.updates = self._adadelta_updates(self.grads)

        # --------------------- FUNCTIONS
        self.train_model = theano.function([self.x, self.y, Param(self.training_mode, default=1)],
                                           self.cost,
                                           updates=self.updates)

        self.validate_model = theano.function([self.x, self.y, Param(self.training_mode, default=0)],
                                              self.cost)

        self.test_model = theano.function([self.x, Param(self.training_mode, default=0)],
                                          self.classifier.p_y_given_x)

    def train(self, train_set, max_iter):
        print 'training for', max_iter, 'iterations'
        self.batch_size.set_value(self.training_batch_size)
        train_set_iterator = RandomTrainIterator(train_set, self.training_batch_size)
        done_looping = False
        iter = 0
        while not done_looping:
            for train_x, train_y in train_set_iterator:
                self.train_model(train_x, train_y)
                if iter > max_iter:
                    done_looping = True
                    break
                iter += 1

    def valid_roc_auc(self, valid_set, prediction_target):
        x, y = valid_set[0], valid_set[1]
        p_y_given_x = self.get_test_proba(x, prediction_target)
        print zip(y,p_y_given_x)
        fpr, tpr, thresholds = roc_curve(y, p_y_given_x, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_test_proba(self, x_test, prediction_target):
        self.batch_size.set_value(len(x_test))
        p_y_given_x = self.test_model(x_test)
        if prediction_target == 'seizure':
            p_y_given_x = p_y_given_x[:, 1] + p_y_given_x[:, 2]
        elif prediction_target == 'early':
            p_y_given_x = p_y_given_x[:, 1]
        else:
            raise ValueError('wrong prediction target')
        return p_y_given_x

    def validate(self, train_set, valid_set, valid_freq, max_iter, prediction_target):
        train_set_iterator = RandomTrainIterator(train_set, self.training_batch_size)

        valid_y_bin = np.copy(valid_set[1])
        if prediction_target == 'seizure':
            valid_y_bin[np.where(valid_y_bin == 2)[0]] = 1
        elif prediction_target == 'early':
            valid_y_bin[np.where(valid_y_bin == 2)[0]] = 0
        else:
            raise ValueError('wrong prediction target')

        valid_set_size = len(valid_set[1])

        # ------------------------------  TRAINING
        epoch = 0
        iter = 0
        best_cost = np.inf
        best_iter_cost = 0
        done_looping = False

        patience = 5000
        patience_increase = 2
        improvement_threshold = 0.995

        while iter < max_iter and not done_looping:
            epoch += 1
            for train_x, train_y in train_set_iterator:
                self.train_model(train_x, train_y)
                iter += 1

                # ------------------------ VALIDATION
                if iter % valid_freq == 0:
                    self.batch_size.set_value(valid_set_size)
                    cost = self.validate_model(valid_set[0], valid_set[1])
                    auc = self.valid_roc_auc((valid_set[0], valid_y_bin), prediction_target)
                    print "%5s %4s %7s %2s %15s %15s %10s " % ('valid', epoch, iter, '|', auc, cost, patience)
                    self.batch_size.set_value(self.training_batch_size)

                    if cost <= best_cost:
                        if cost < best_cost * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_iter_cost = iter
                        best_cost = cost

                if patience <= iter:
                    done_looping = True
        print 'best_iter:', best_iter_cost, 'best_cost:', best_cost
        return best_iter_cost

    def _adadelta_updates(self, grads, learning_rate=0.1, rho=0.95, epsilon=1e-6):
        print 'adadelta'
        accumulators = [theano.shared(np.zeros_like(param_i.get_value())) for param_i in self.params]
        delta_accumulators = [theano.shared(np.zeros_like(param_i.get_value())) for param_i in self.params]

        updates = []
        for param_i, grad_i, acc_i, acc_delta_i in zip(self.params, grads, accumulators, delta_accumulators):
            acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
            updates.append((acc_i, acc_i_new))

            update_i = grad_i * T.sqrt(acc_delta_i + epsilon) / T.sqrt(acc_i_new + epsilon)
            updates.append((param_i, param_i - learning_rate * update_i))

            acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i ** 2
            updates.append((acc_delta_i, acc_delta_i_new))

        return updates
