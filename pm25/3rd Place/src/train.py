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

def prep(df, sort=False):
    le = LabelEncoder()
    df['grid_id_encoded'] = le.fit_transform(df['grid_id'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df.drop(cols_to_drop, inplace=True, axis=1)

    if sort:
        df.sort_values(by='date', inplace=True)

    hmp_le_enc = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, hmp_le_enc

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
    
    df = pd.read_csv(f'{save_path}/{final_view_save_name}') #'../merged_csv/train_desc_aod_and_meteo_vars_11.03.csv')
    df['date'] = pd.to_datetime(df['date'])

    cols_to_drop = ['surface', 'level', 'heightAboveGround', 'Optical_Depth_055_var', 'Optical_Depth_055_std']

    df = add_geo_coords(df)
    df_p, hmp_le_enc = prep(df)

    imputing_method = 'interp_linear'
    df_filled = fillna_by_grid(df_p, imputing_method)
    df_filled.dropna(inplace=True)

    df_filled['wind_magnitude'] = np.sqrt(df_filled.u ** 2 + df_filled.v ** 2)
    df_filled['mean_encode_grid'] = df_filled.groupby('grid_id')['value'].transform('mean')
    df_filled['mean_encode_city'] = df_filled.groupby('location')['value'].transform('mean')

    hmp_mean_enc = {i[0]:i[1] for i in df_filled.groupby('grid_id')['mean_encode_grid'].first().reset_index().values}
    hmp_mean_enc_city = {i[0]:i[1] for i in df_filled.groupby('location')['mean_encode_city'].first().reset_index().values}
    global_mean_enc = df_filled['mean_encode_grid'].mean()

    df_filled.drop('mean_encode_city', axis=1, inplace=True)

    df_filled = df_filled.sort_values(by='date', ascending=True)

    params = {'max_depth':41, 'max_features':'log2', 'min_samples_leaf':25, 'n_estimators':502}
    rf = RandomForestRegressor(random_state=17, **params)

    train_x, test_x, train_y, test_y = get_train_test_split(df_filled)

    rf.fit(train_x, train_y)
    # With mean encoding, wind magnitude
    preds = rf.predict(test_x)
    rmse = math.sqrt(mean_squared_error(test_y, preds))
    r2 = r2_score(test_y, preds)
    print('RF: Results on validation: rmse: {} r2: {}'.format(rmse, r2))

    gr = GradientBoostingRegressor()
    gr.fit(train_x, train_y)

    preds_gr = gr.predict(test_x)
    rmse = math.sqrt(mean_squared_error(test_y, preds_gr))
    r2 = r2_score(test_y, preds_gr)
    print('GBR: Results on validation: rmse: {} r2: {}'.format(rmse, r2))

    # Saving model, encodings
    save_path = 'models'
    # os.makedirs('saved_models', exist_ok=True)

    # Pickle
    # filename_rf = 'rf_winning_02.04.pkl'  

    # with open(f'{save_path}/{filename_rf}', 'wb') as file:  
    #     pickle.dump(rf, file)

    # filename_grb = 'grb_winning_02.04.pkl'  

    # with open(f'{save_path}/{filename_grb}', 'wb') as file:  
    #     pickle.dump(gr, file)


    print('Training on full data')
    y = df_filled['value']
    x = df_filled.drop(columns=['value', 'date', 'grid_id', 'datetime', 'location'])

    params = {'max_depth':41, 'max_features':'log2', 'min_samples_leaf':25, 'n_estimators':502}
    rf = RandomForestRegressor(random_state=17, **params)
    rf.fit(x, y)

    gr = GradientBoostingRegressor()
    gr.fit(x, y)

    # Joblib
    filename_rf = hmp['saved_final_rf_model']  # 'rf_winning_02.04_joblib.pkl'  
    filename_grb = hmp['saved_final_gbr_model'] # 'grb_winning_02.04_joblib.pkl'  

    joblib.dump(rf, filename_rf)
    joblib.dump(gr, filename_grb)

    # Saving Encodings
    mean_enc_file = hmp['mean_enc_mappings']  # 'mean_enc_mappings_02.04_joblib.pkl'  
    le_file = hmp['le_mappings']   # 'le_mappings_02.04_joblib.pkl'  
    mean_enc_file_city = hmp['city_mean_enc_mappings']
    global_mean_enc_path = hmp['global_mean_enc']

    joblib.dump(hmp_mean_enc, mean_enc_file)
    joblib.dump(hmp_le_enc, le_file)
    joblib.dump(hmp_mean_enc_city, mean_enc_file_city)

    with open(global_mean_enc_path, 'w') as f:
        f.write(str(global_mean_enc))
        f.write('\n')



