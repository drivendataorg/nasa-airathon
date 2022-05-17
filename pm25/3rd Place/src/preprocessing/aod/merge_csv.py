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


if __name__ == '__main__':
    assembled_csv = list(Path(sys.argv[1]).glob('*'))
    lb = pd.read_csv(sys.argv[2])
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

        lb = pd.merge(lb, df[['date', 'grid_id', k]], on = ['date', 'grid_id'], how = 'left', suffixes=('', f':{k}'))
       
    os.makedirs(sys.argv[3], exist_ok = True)
    lb.to_csv(f'{sys.argv[3]}/{sys.argv[4]}', index=False)