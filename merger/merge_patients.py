from pandas import read_csv, merge
import csv
import sys


def merge_csv_data(seizure_path, early_path, patients, submission_name):
    with open('submission_'+submission_name+'.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'seizure', 'early'])

    for patient in patients:
        df_seizure = read_csv(seizure_path + '/' + patient + 'seizure.csv')
        df_early = read_csv(early_path + '/' + patient + 'early.csv')
        df = merge(df_seizure, df_early, on='clip')
        with open('submission_'+submission_name+'.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


if __name__ == '__main__':
    seizure_path = sys.argv[1]
    early_path = sys.argv[2]
    patients = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5',
                'Patient_6', 'Patient_7', 'Patient_8']
    merge_csv_data(seizure_path, early_path, patients, 'late_loader_newa')

