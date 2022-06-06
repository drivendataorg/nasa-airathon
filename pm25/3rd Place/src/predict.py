import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import shapely.wkt
import pickle
import joblib
import yaml
import sys

def prep(df, le_mappings, sort=False):
    df['grid_id_encoded'] = df['grid_id'].map(le_mappings)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df.drop(cols_to_drop, inplace=True, axis=1)

    if sort:
        df.sort_values(by='date', inplace=True)

    return df

def fillna(df, method='locf_nocb'):

    if method == 'locf_nocb':
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
    
    elif method == 'interp_linear':
        df = df.interpolate(method='linear', axis=0)
    
    elif method == 'interp_spline_2':
        df = df.interpolate(method='spline', order=2, axis=0)

    elif method == 'interp_spline_3':
        df = df.interpolate(method='spline', order=3, axis=0)
        
    return df


def fillna_by_grid(df, method):
    
    grid_ids = df.grid_id.unique()
    res = []
    for id in grid_ids:
        mask = df['grid_id'] == id
        subset = df[mask].copy()

        subset.sort_values(by='date', inplace=True, ascending=True)
        
        if method in ('locf_nocb', 'interp_linear', 'interp_spline_2', 'interp_spline_3'):
            date_temp = subset['date'].copy()
            subset = fillna(subset.drop('date', axis=1), method=method)
            subset['date'] = date_temp
        
        res.append(subset)
    
    return pd.concat(res).sort_values(by='date', ascending=True)


def fillna_by_city(df, method):
    
    locations = df.location.unique()
    res = []
    for loc in locations:
        mask = df['location'] == loc
        subset = df[mask].copy()

        subset.sort_values(by='date', inplace=True, ascending=True)
        
        if method in ('locf_nocb', 'interp_linear', 'interp_spline_2', 'interp_spline_3'):
            date_temp = subset['date'].copy()
            subset = fillna(subset.drop('date', axis=1), method=method)
            subset['date'] = date_temp
        
        res.append(subset)
    
    return pd.concat(res).sort_values(by='date', ascending=True)


def getBounds(shape):
    x1 = []
    y1 = []
      
    x = shape.exterior.coords.xy[0]
    y = shape.exterior.coords.xy[1]
    x1.append(min(x))
    x1.append(max(x))
    y1.append(min(y))
    y1.append(max(y))

    return x1,y1

def get_geo_coords(hmp):

    for i in range(grid_metadata.shape[0]):
        grid_id = grid_metadata.iloc[i].grid_id
        grid_shape = shapely.wkt.loads(grid_metadata.iloc[i]['wkt'])

        lon, lat = getBounds(grid_shape)        
        hmp[grid_id] = (lon, lat)

    return hmp 


def add_geo_coords(df_inp):
    df = df_inp.copy()
    hmp = {}
    hmp = get_geo_coords(hmp)

    df['min_lon'] = df['grid_id'].map(hmp).apply(lambda x: x[0][0])
    df['max_lon'] = df['grid_id'].map(hmp).apply(lambda x: x[0][1])
    df['min_lat'] = df['grid_id'].map(hmp).apply(lambda x: x[1][0])
    df['max_lat'] = df['grid_id'].map(hmp).apply(lambda x: x[1][1])

    df = df.drop(['longitude', 'latitude'], axis=1)
    return df

def get_train_test_split(df, split_ratio = 0.8):
    y = df['value']
    x = df.drop(columns=['value', 'date', 'grid_id', 'datetime', 'location'])

    # Only future values should go to test
    split_idx = int(df.shape[0]*split_ratio)
    train_x, test_x = x.iloc[:split_idx], x.iloc[split_idx:]
    train_y, test_y = y.iloc[:split_idx], y.iloc[split_idx:]

    return train_x, test_x, train_y, test_y


grid_metadata = None

if __name__ == '__main__':

    cfg_name = sys.argv[1]

    with open(f'cfg/{cfg_name}') as f:
        # use safe_load instead load
        hmp = yaml.safe_load(f)

    save_path = hmp['path_save_final_view']
    final_view_save_name = hmp['aod_and_gfs_filename']
    grid_metadata = pd.read_csv(hmp['path_grid_metadata'])

    test_df = pd.read_csv(f'{save_path}/{final_view_save_name}')
    test_df['date'] = pd.to_datetime(test_df['date'])

    cols_to_drop = ['surface', 'level', 'heightAboveGround', 'Optical_Depth_055_var', 'Optical_Depth_055_std']

    test_df = add_geo_coords(test_df)

    save_path = 'models'
    mean_enc_file = f'models/saved_encodings/mean_enc_mappings_02.04_joblib.pkl' # hmp['mean_enc_mappings']
    mean_enc_file_city = f'models/saved_encodings/mean_enc_mappings_02.04_joblib.pkl' # hmp['city_mean_enc_mappings'] #  
    le_file = 'models/saved_encodings/le_mappings_02.04_joblib.pkl' # hmp['le_mappings']   
    global_mean_val_path = 'models/saved_encodings/global_mean_enc.txt' # hmp['global_mean_enc']

    le_enc_hmp = joblib.load(le_file)

    # Accounting for unseen grid ids
    max_val = len(le_enc_hmp.keys())
    for grid_id in test_df.grid_id.unique():
        if grid_id not in le_enc_hmp:
            le_enc_hmp[grid_id] = max_val
            max_val += 1

    test_df = prep(test_df, le_enc_hmp)

    imputing_method = 'interp_linear'
    test_filled = fillna_by_grid(test_df, imputing_method)

    # Can't drop test records, interpolating left ones
    imputing_method = 'locf_nocb'
    test_filled = fillna_by_grid(test_filled, imputing_method)

    if test_filled.isnull().any().any():
        imputing_method = 'interp_linear'
        test_filled = fillna_by_city(test_filled, imputing_method)

        imputing_method = 'locf_nocb'
        test_filled = fillna_by_city(test_filled, imputing_method)

    if test_filled.isnull().any().any():
        print('Some columns are fully missing for each grid id for the given location. (Probably, Aerosol Optical Depth values)')
        print('Filling in with global mean of the dataset or precomputed statistics from train')
        
        
        precomputed_stats = pd.read_csv('models/precomputed_fill_in_values.csv')
        for i in range(test_filled.shape[0]):
            if test_filled.iloc[i].isnull().any(axis=0):
                mask = test_filled.iloc[i].isnull()
                location = test_filled.iloc[i]['location']
                temp = precomputed_stats[precomputed_stats['location'] == location] 
                columns = test_filled.iloc[i][mask].index.values
                for col in columns:
                    test_filled.at[i, col] = temp[col]

        
        for i in test_filled.columns[test_filled.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values    
            try:
                test_filled[i].fillna(precomputed_stats[i].mean(), inplace=True)
            except:
                pass


    test_filled['wind_magnitude'] = np.sqrt(test_filled.u ** 2 + test_filled.v ** 2)
    
    # Mean encoding
    mean_enc_hmp = joblib.load(mean_enc_file)
    mean_enc_hmp_city = joblib.load(mean_enc_file_city)

    with open(global_mean_val_path, 'r') as f:
        global_mean = f.read().splitlines()[0]
        global_mean = np.float64(global_mean)

    mask_unseen = ~test_filled.grid_id.isin(mean_enc_hmp.keys())
    
    unseen_grid_city_pairs = {i[0]:i[1] for i in test_filled[mask_unseen].groupby('grid_id')['location'].first().reset_index().values}

    for grid_id in unseen_grid_city_pairs:
        loc = unseen_grid_city_pairs[grid_id] 
        if loc in mean_enc_hmp_city:
            mean_enc_hmp[grid_id] = mean_enc_hmp_city[loc]
        else:
            mean_enc_hmp[grid_id] = global_mean


    # Need to account for the new grids here later
    test_filled['mean_encode_grid'] = test_filled['grid_id'].map(mean_enc_hmp)

    y = test_filled['value']
    x = test_filled.drop(columns=['value', 'date', 'grid_id', 'datetime', 'location'])

    # Check order of columns maybe

    filename_rf = 'models/rf_winning_02.04_joblib.pkl' # hmp['saved_final_rf_model']  #'rf_winning_02.04_joblib.pkl'  
    filename_grb = 'models/grb_winning_02.04_joblib.pkl' # hmp['saved_final_gbr_model']  

    rf_joblib = joblib.load(filename_rf)
    gr_joblib = joblib.load(filename_grb)

    #test_filled.to_csv('test_www.csv')

    preds_rf = rf_joblib.predict(x)
    preds_gr = gr_joblib.predict(x)
    
    preds = (preds_gr + preds_rf) / 2
    test_filled['value'] = preds

    # Further just return date, value, grid_id
    # Or match with subm

    subm = pd.read_csv(hmp['path_labels'])
    subm = pd.merge(subm[['datetime', 'grid_id']], test_filled[['datetime', 'grid_id', 'value']], on=['datetime', 'grid_id'])
    subm.to_csv(f"models/predictions/pred_{hmp['period_name']}.csv", index=False) # subm_02.04_winning.csv
    print(f'Predictions are available in models/predictions/pred_{hmp["period_name"]}.csv')