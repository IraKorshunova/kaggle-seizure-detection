from pandas import read_csv, merge
import csv
import sys
from pandas.core.frame import DataFrame

def average_csv_data(patients, filename, target, *data_path):
    data_path = data_path[0]
    df_list = []
    for p in data_path:
        df = DataFrame(columns=['clip',target])
        for patient in patients:
            d = read_csv(p + '/' + patient + target + '.csv')
            df = df.append(d)
        df_list.append(df)

    avg_df = DataFrame(columns=['clip', target])
    avg_df['clip'] = df_list[0]['clip']
    avg_df[target] = 0
    for df in df_list:
        avg_df[target] += df[target]

    avg_df[target] /= 1.0 * len(df_list)

    with open(filename+'.csv', 'wb') as f:
        avg_df.to_csv(f, header=True, index=False)

if __name__ == '__main__':
    path = sys.argv[1:]
    print path
    patients = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5',
                'Patient_6', 'Patient_7', 'Patient_8']
    average_csv_data(patients, 'submission_early', 'early', path)




