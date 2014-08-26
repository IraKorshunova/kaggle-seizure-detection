import numpy as np
from scipy.io import loadmat, savemat
import scipy as sc
import scipy.signal
from scipy import interpolate

filename = 'clips/Patient_2/Patient_2_ictal_segment_1.mat'
d = loadmat(filename, squeeze_me=True)
x = d['data']
latency = d['latency']
sampling_frequency = d['freq']
n_channels = d['channels']

sampling_frequency = x.shape[1]
lowcut = 0
highcut = 25
nyq = 0.5 * sampling_frequency
high = highcut / nyq
b, a = sc.signal.butter(3, high)
x_filt = sc.signal.lfilter(b, a, x, axis=1)

t = np.linspace(0, x.shape[1] - 1, sampling_frequency)
sampling_frequency2 = 2 * highcut
t2 = np.linspace(0, x.shape[1], sampling_frequency2, endpoint=False)
f = interpolate.interp1d(t, x_filt, axis=1)
x2 = f(t2)
