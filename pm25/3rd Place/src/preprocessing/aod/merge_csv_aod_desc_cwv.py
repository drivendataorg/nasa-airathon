"""
to run the script:
    python merge_csv.py <path_to_the_assembled_csv> <path_train_label.csv> <save_path> <save_filename>
args:
    <path_to_the_assembled_csv>: folder with separete csvs for each var and loc
    <path_train_label.csv>: in the format path/filename.csv 
    <save_path>: in the format path
    <save_filename>: name for the merged csv

"""

from genericpath import exists
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import sys


def join_labels_aod_cwv(path_to_the_assembled_csv, path_train_label, save_path, save_filename):

    assembled_csv = list(Path(path_to_the_assembled_csv).glob('*'))
    assembled_csv = list(filter(lambda x: '_'.join(x.name.split('.')[0].split('_')[1:]) in ('Optical_Depth_055_descriptive', 'Column_WV'), assembled_csv))
    lb = pd.read_csv(path_train_label)
    lb['date'] = pd.to_datetime(lb['datetime'].apply(lambda x: x[:10]))

    hmp = {}
    for csv_file in assembled_csv:
        temp = csv_file.name.split('.')[0]
        start = temp.index('_')
        var = temp[start+1:]

        hmp.setdefault(var, [])
        hmp[var].append(csv_file)


    for k,v in hmp.items():
    
        res = []
        for file in v:
            temp = pd.read_csv(file)
            res.append(temp)

        df = pd.concat(res)

        df['date'] = pd.to_datetime(df['filename'].apply(lambda x: f'{x[:4]}/{x[4:6]}/{x[6:8]}'))
        df['time'] = df['filename'].apply(lambda x: x[9:])

        df = df[df['file_n'] == 0]
        df.drop_duplicates(subset=['date', 'grid_id'], inplace=True)

        if k == 'Optical_Depth_055_descriptive':
            var_of_interest = 'Optical_Depth_055'
            interest_cols = [f'{var_of_interest}_avg', f'{var_of_interest}_min', f'{var_of_interest}_max', 
            f'{var_of_interest}_var', f'{var_of_interest}_std', f'{var_of_interest}_p95']
        else:
            interest_cols = [k]

        lb = pd.merge(lb, df[['date', 'grid_id'] + interest_cols], on = ['date', 'grid_id'], how = 'left', suffixes=('', f':{k}'))
       
    os.makedirs(save_path, exist_ok = True)
    lb.to_csv(f'{save_path}/{save_filename}', index=False)


if __name__ == '__main__':
    
    path_to_the_assembled_csv = sys.argv[1]
    path_train_label = sys.argv[2]
    save_path = sys.argv[3]
    save_filename = sys.argv[4]

    join_labels_aod_cwv(path_to_the_assembled_csv, path_train_label, save_path, save_filename)

    
    # assembled_csv = list(Path(sys.argv[1]).glob('*'))
    # assembled_csv = list(filter(lambda x: '_'.join(x.name.split('.')[0].split('_')[1:]) in ('Optical_Depth_055_descriptive', 'Column_WV'), assembled_csv))
    # lb = pd.read_csv(sys.argv[2])
    # lb['date'] = pd.to_datetime(lb['datetime'].apply(lambda x: x[:10]))

    # hmp = {}
    # for csv_file in assembled_csv:
    #     temp = csv_file.name.split('.')[0]
    #     start = temp.index('_')
    #     var = temp[start+1:]

    #     hmp.setdefault(var, [])
    #     hmp[var].append(csv_file)


    # for k,v in hmp.items():
    
    #     res = []
    #     for file in v:
    #         temp = pd.read_csv(file)
    #         res.append(temp)

    #     df = pd.concat(res)

    #     df['date'] = pd.to_datetime(df['filename'].apply(lambda x: f'{x[:4]}/{x[4:6]}/{x[6:8]}'))
    #     df['time'] = df['filename'].apply(lambda x: x[9:])

    #     df = df[df['file_n'] == 0]
    #     df.drop_duplicates(subset=['date', 'grid_id'], inplace=True)

    #     if k == 'Optical_Depth_055_descriptive':
    #         var_of_interest = 'Optical_Depth_055'
    #         interest_cols = [f'{var_of_interest}_avg', f'{var_of_interest}_min', f'{var_of_interest}_max', 
    #         f'{var_of_interest}_var', f'{var_of_interest}_std', f'{var_of_interest}_p95']
    #     else:
    #         interest_cols = [k]

    #     lb = pd.merge(lb, df[['date', 'grid_id'] + interest_cols], on = ['date', 'grid_id'], how = 'left', suffixes=('', f':{k}'))
       
    # os.makedirs(sys.argv[3], exist_ok = True)
    # lb.to_csv(f'{sys.argv[3]}/{sys.argv[4]}', index=False)