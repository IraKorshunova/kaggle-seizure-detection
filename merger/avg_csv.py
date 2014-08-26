from pandas import read_csv, merge
import csv
import sys
from pandas.core.frame import DataFrame

def average_submissions():
    with open('submission_avg_13Aug.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'seizure', 'early'])

        df1 = read_csv('submission_late_loader_newa.csv')
        df2 = read_csv('submission_newa_all.csv')
        df = DataFrame(columns=['clip', 'seizure', 'early'])
        df['clip'] = df1['clip']
        df['seizure'] = (df1['seizure'] + df2['seizure'])/2.0
        df['early'] = (df1['early'] + df2['early'])/2.0
        with open('submission_avg_13Aug.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


def average_csv_data(filename, target, *data_path):
    data_path = data_path[0]
    df_list = []
    for p in data_path:
        d = read_csv(p)
        df_list.append(d)

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
    average_csv_data('submission_seizure', 'seizure', path)




