import sys
import os
from theano import config
from early.cnn.conv_net import ConvNet
from early.cnn_trainer.loader import load_train, load_test
from pandas import DataFrame

config.floatX = 'float32'


def train_and_test(patient_name, prediction_target, root_path, csv_path):
    path = root_path + '/' + patient_name
    d = load_train(path)
    X = d['x']
    Y = d['y']
    scaler = d['scaler']
    x_train, y_train = d['x_train'], d['y_train']
    x_valid, y_valid = d['x_valid'], d['y_valid']
    print 'sfsfsfsfsfsfs', y_valid

    print '============ dataset'
    print X.shape, Y.shape
    print 'train:', x_train.shape
    print 'valid:', x_valid.shape
    print '===================='

    # ----------- PARAMETERS
    n_timesteps = d['n_timesteps']
    dim = d['n_channels']
    dropout_prob = [0.3, 0.5]
    batch_size = 10
    max_iter = 50000
    valid_freq = 50
    activation = 'tanh'
    weights_variance = 0.1
    l2_reg = 0.0001

    # #----------- PARAMETERS 1
    # recept_width = n_timesteps
    # pool_width = 1
    # nkerns = [40, 100]

    # ----------- PARAMETERS 2
    recept_width = [10, 3]
    pool_width = [1, 1]
    nkerns = [32, 64, 128]
    stride = [5, 1]

    #--------------
    recept_width = [5, 3]
    pool_width = [1, 1]
    nkerns = [16, 32, 64]
    stride = [3, 1]

    print '======== parameters'
    print 'target:', prediction_target
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
                  dim=dim)

    best_iter_cost = cnn.validate(train_set=(x_train, y_train),
                                  valid_set=(x_valid, y_valid),
                                  valid_freq=valid_freq,
                                  max_iter=max_iter,
                                  prediction_target=prediction_target)

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
                  dim=dim)

    cnn.train(train_set=(X, Y), max_iter=max(200, best_iter_cost))

    # test data
    d = load_test(path, scaler)
    x_test = d['x']
    id = d['id']

    test_proba = cnn.get_test_proba(x_test, prediction_target)
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
    for patient_name in ['Patient_3']:
        print '***********************', patient_name, '***************************'
        train_and_test(patient_name, 'seizure', root_path, csv_path)