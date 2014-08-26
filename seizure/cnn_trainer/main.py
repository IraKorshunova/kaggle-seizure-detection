import os
import sys
import numpy as np

from theano import config
from pandas import DataFrame

from seizure.cnn.conv_net import ConvNet
from seizure.cnn_trainer.loader import load_train, load_test


config.floatX = 'float32'


def train_and_test(patient_name, prediction_target, root_path, csv_path):
    path = root_path + '/' + patient_name
    d = load_train(path)
    X = d['x']
    Y = d['y']
    scaler = d['scaler']
    x_train, y_train = d['x_train'], d['y_train']
    x_valid, y_valid = d['x_valid'], d['y_valid']

    if prediction_target == 'seizure':
        Y[np.where(Y == 2)[0]] = 1
        y_train[np.where(y_train == 2)[0]] = 1
        y_valid[np.where(y_valid == 2)[0]] = 1

    print '============ dataset'
    print 'train:', x_train.shape
    print 'n_pos:', np.sum(y_train), 'n_neg:', len(y_train) - np.sum(y_train)
    print 'valid:', x_valid.shape
    print 'n_pos:', np.sum(y_valid), 'n_neg:', len(y_valid) - np.sum(y_valid)
    print '===================='

    # ----------- PARAMETERS 1+2
    n_timesteps = d['n_timesteps']
    dim = d['n_channels']
    dropout_prob = [0.2, 0.5]  # on 2 last layers
    batch_size = 10
    max_iter = 40000
    valid_freq = 50
    activation = 'tanh'
    weights_variance = 0.1
    l2_reg = 0.001
    objective_function = 'cross_entropy'

    # ----------- PARAMETERS 2
    # recept_width = [25, 26]
    # pool_width = [1, 1]
    # nkerns = [20, 40, 128]
    # stride = [1, 1]

    # ----------- PARAMETERS 3
    recept_width = [5, 1]
    pool_width = [2, 1]
    nkerns = [32, 16, 128]
    stride = [5, 1]

    # ----------- PARAMETERS 4
    # recept_width = [15, 4]
    # pool_width = [2, 1]
    # nkerns = [32, 64, 128]
    # stride = [5, 1]

    print '======== parameters'
    print 'n_timesteps:', n_timesteps
    print 'n_channels:', dim
    print 'max_epoch: ', max_iter
    print 'valid_freq', valid_freq
    print 'nkerns: ', nkerns
    print 'receptive width: ', recept_width
    print 'pool_width:', pool_width
    print 'strides:', stride
    print 'dropout_prob: ', dropout_prob
    print 'batch_size:', batch_size
    print 'activation:', activation
    print 'L2_reg:', l2_reg
    print 'weights_variance:', weights_variance
    print 'objective function:', objective_function
    print '===================='

    cnn = ConvNet(nkerns=nkerns,
                  recept_width=recept_width,
                  pool_width=pool_width,
                  stride=stride,
                  dropout_prob=dropout_prob,
                  l2_reg=l2_reg,
                  training_batch_size=batch_size,
                  activation=activation,
                  weights_variance=weights_variance,
                  n_timesteps=n_timesteps,
                  dim=dim,
                  objective_function=objective_function)

    best_iter_cost = cnn.validate(train_set=(x_train, y_train),
                                  valid_set=(x_valid, y_valid),
                                  valid_freq=valid_freq,
                                  max_iter=max_iter)

    cnn = ConvNet(nkerns=nkerns,
                  recept_width=recept_width,
                  pool_width=pool_width,
                  stride=stride,
                  dropout_prob=dropout_prob,
                  l2_reg=l2_reg,
                  training_batch_size=batch_size,
                  activation=activation,
                  weights_variance=weights_variance,
                  n_timesteps=n_timesteps,
                  dim=dim,
                  objective_function=objective_function)

    cnn.train(train_set=(X, Y), max_iter=max(500, best_iter_cost))

    # test data
    d = load_test(path, scaler)
    x_test = d['x']
    id = d['id']

    test_proba = cnn.get_test_proba(x_test)
    ans = zip(id, test_proba)
    df = DataFrame(data=ans, columns=['clip', prediction_target])
    df.to_csv(csv_path + '/' + patient_name +
              prediction_target + '.csv', index=False, header=True)


if __name__ == '__main__':
    root_path = sys.argv[1]
    csv_path = sys.argv[2]
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3',
             'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']
    for patient_name in names:
        print '***********************', patient_name, '***************************'
        train_and_test(patient_name, 'seizure', root_path, csv_path)
