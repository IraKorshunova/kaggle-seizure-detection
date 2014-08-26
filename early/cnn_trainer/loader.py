import os
import re
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def get_outliers(x, thresh=10):
    x_sorted = -np.sort(-x)[:10]
    arg_sorted = np.argsort(-x)[:10]
    if len(x_sorted.shape) == 1:
        x_sorted = x_sorted[:, np.newaxis]
    median = np.median(x_sorted, axis=0)
    diff = np.sum((x_sorted - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    print zip(x_sorted, arg_sorted, modified_z_score > thresh)
    return arg_sorted[modified_z_score > thresh]


def load_train(path):
    data = []
    filenames = sorted(os.listdir(path))
    for filename in sorted(filenames, key=lambda x: int(re.search(r'(\d+).mat', x).group(1))):
        if 'test' not in filename:
            data.append(loadmat(path + '/' + filename, squeeze_me=True))

    n_train = len(data)
    n_channels, n_timesteps = data[0]['data'].shape
    n_features = n_channels * n_timesteps
    x = np.zeros((n_train, n_features), dtype='float32')
    y = np.zeros(n_train, dtype='int8')

    seizures_list = []
    seizure_idx = []
    for i, datum in enumerate(data):
        x[i, :] = np.reshape(datum['data'], n_features)
        if 'latency' in datum:
            latency = datum['latency']
            y[i] = 1 if latency <= 15 else 2
            if latency == 0:
                if len(seizure_idx) > 0:
                    seizures_list.append(seizure_idx)
                seizure_idx = []
            seizure_idx.append(i)
        else:
            y[i] = 0
    seizures_list.append(seizure_idx)

    # PRINT STATS
    neg_idx = np.where(y == 0)[0]
    early_idx = np.where(y == 1)[0]
    late_idx = np.where(y == 2)[0]

    print 'train set statistics:'
    print 'n_channles:', n_channels, 'n_timesteps:', n_timesteps
    print 'train set size:', n_train
    print 'negative:', x[neg_idx, :].shape
    print 'early_seizure:', x[early_idx, :].shape
    print 'late_seizure:', x[late_idx, :].shape

    # separate ictal to train/valid
    n_seizures = len(seizures_list)
    n_valid_seizures = int(max(1, np.round(0.2 * n_seizures)))
    seizures_valid_idx = sum(seizures_list[-n_valid_seizures:], [])
    seizures_train_idx = sum(seizures_list[:-n_valid_seizures], [])
    # seizures_valid_idx = sum(seizures_list[:n_valid_seizures], [])
    # seizures_train_idx = sum(seizures_list[n_valid_seizures:], [])

    for i in seizures_list:
        print 'seizure', i
    print 'valid seizures', seizures_valid_idx
    print 'train ', seizures_train_idx

    # separate interictal to train/ valid
    neg_valid_len = int(0.2 * len(neg_idx))
    # neg_valid_idx = neg_idx[-neg_valid_len:]
    # neg_train_idx = neg_idx[:-neg_valid_len]
    neg_valid_idx = neg_idx[:neg_valid_len]
    neg_train_idx = neg_idx[neg_valid_len:]

    # combine ictal and interictal
    train_idx = neg_train_idx.tolist() + seizures_train_idx
    valid_idx = neg_valid_idx.tolist() + seizures_valid_idx

    # remove outliers from training data indices
    outliers_idx = []
    outliers_idx.extend(get_outliers(np.std(x[neg_idx, :], axis=1)))
    outliers_idx.extend(get_outliers(np.std(x[early_idx, :], axis=1)))
    outliers_idx.extend(get_outliers(np.std(x[late_idx, :], axis=1)))
    print 'outliers indices:', outliers_idx
    for i in outliers_idx:
        if i in train_idx:
            train_idx.remove(i)
        if i in valid_idx:
            valid_idx.remove(i)

    # train and validation data
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]

    # all data
    x = np.vstack((x_train, x_valid))
    y = np.concatenate((y_train, y_valid))

    # scale train and valid
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)

    # scale whole data
    scaler_all = StandardScaler()
    x = scaler_all.fit_transform(x)

    return {'x': x, 'y': y, 'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid,
            'n_channels': n_channels,
            'n_timesteps': n_timesteps, 'scaler': scaler_all}


def load_test(path, scaler):
    data, id = [], []
    for filename in sorted(os.listdir(path), key=lambda x: int(re.search(r'(\d+).mat', x).group(1))):
        if 'test' in filename:
            data.append(loadmat(path + '/' + filename, squeeze_me=True))
            id.append(filename)

    n_test = len(data)
    n_channels, n_timesteps = data[0]['data'].shape
    n_features = n_channels * n_timesteps
    x = np.zeros((n_test, n_features), dtype='float32')

    for i, datum in enumerate(data):
        x[i, :] = np.reshape(datum['data'], n_features)

    x = scaler.transform(x)

    print 'test set statistics:'
    print 'n_channles:', n_channels, 'n_timesteps:', n_timesteps
    print 'test set size:', x.shape

    return {'x': x, 'id': id}