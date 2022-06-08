import pandas as pd
import numpy as np
import os
from pathlib import Path 
import glob

def get_group_csv(group_name, read_path, locations, period_name):
        
    res = []
    for loc in locations:
        g_path = f'{read_path}/{loc}/{period_name}/{loc}_gfs_{group_name}.csv'
        
        g_temp = pd.read_csv(g_path)
        res.append(g_temp) # O(1)

    g_df = pd.concat(res)
    return g_df


def form_final_view(aod_desc_path, read_path, save_path, save_name, locations, period_name):

    df = pd.read_csv(aod_desc_path)

    g1_df = get_group_csv('group_1', read_path, locations, period_name)
    g2_df = get_group_csv('group_2', read_path, locations, period_name)

    g1_df.rename(columns={'time':'date'}, inplace=True)
    df = pd.merge(df, g1_df, on=['date', 'grid_id'], how='left')

    g2_df.rename(columns={'time_date':'date'}, inplace=True)
    df = pd.merge(df, g2_df, on=['date', 'grid_id'], how='left')

    df = df.drop('location_y', axis=1).rename(columns={'location_x':'location'})  

    df.to_csv(f'{save_path}/{save_name}', index=False) # train_desc_aod_and_meteo_vars_11.03.csv


if __name__ == "__main__":

    read_path = '../data/gfs/merged_csv'
    save_path = '../merged_csv'
    train_aod_desc_path = '../merged_csv/merged_aod_desc_cwv_train_10.03.csv'
    test_aod_desc_path = '../merged_csv/merged_aod_desc_cwv_test_10.03.csv'
    locations = ['la', 'tp', 'dl']    

    form_final_view(train_aod_desc_path, read_path, save_path, 'train_desc_aod_and_meteo_vars_11.03.csv', locations)



