import os
import numpy as np
import scipy as sc
import scipy.signal
from scipy import interpolate
from scipy.io import loadmat, savemat


def get_files_paths(directory, extension):
    files_with_extension = list()
    for root, dirs, files in os.walk(directory):
        files_with_extension += [root + '/' + file_name for file_name in files if
                                 file_name.endswith(extension) and not file_name.startswith('.')]
    return files_with_extension


if __name__ == '__main__':
    read_dir = 'clips'
    write_dir = 'xclips'

    for raw_file_path in get_files_paths('../EEG/Volumes/Seagate/seizure_detection/competition_data/clips/Patient_4',
                                         '.mat'):
        print raw_file_path
        preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)

        dir_path = os.path.dirname(preprocessed_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # =========================You should split this processing into reusable functions============================
        d = loadmat(raw_file_path, squeeze_me=True)
        x = d['data']
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

        if '_ictal_' in raw_file_path:
            d2 = {'data': x2, 'latency': d['latency'], 'freq': sampling_frequency, 'channels': d['channels']}
        else:
            d2 = {'data': x2, 'freq': sampling_frequency, 'channels': d['channels']}

        # =============================================================================================================

        savemat(preprocessed_file_path, d2)
        print raw_file_path
