from pandas import read_csv, merge
import sys


def merge_csv_data(seizure_csv, early_csv, filename):
        df_seizure = read_csv(seizure_csv)
        df_early = read_csv(early_csv)
        df = merge(df_seizure, df_early, on='clip')

        with open(filename + '.csv', 'wb') as f:
            df.to_csv(f, header=True, index=False)

if __name__ == '__main__':
    seizure_csv = sys.argv[1]
    early_csv = sys.argv[2]
    merge_csv_data(seizure_csv, early_csv, 'submission')
