import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.utils import check_existance
from typing import Tuple
import datetime
from loguru import logger

META_COLUMNS = ['filename']

def _load_features(path_dir: str, total: int) -> pd.DataFrame:
    """
    Load features from .npz files.
    """
    features = defaultdict(lambda:[])
    for idx in tqdm(range(total)):
        filename = os.path.join(path_dir, f"{idx}.npz")
        if not os.path.exists(filename):
            continue
        data = np.load(filename)
        for key in data.keys():
            if key == 'filename':
                continue
            if key == 'label':
                if data[key] >= 0 or data[key] < 0:
                    features[key].append(data[key])
                else:
                    features[key].append(np.nan)
                continue
            _band = data[key].ravel()
            _band = np.concatenate((
                _band[_band >= 0], _band[_band < 0]
            )) # removing nan values
            mean, var = _band.mean(), _band.std() ** 2
            features[key + '_mean'].append(mean)
            features[key + '_var'].append(var)
    k = len(features[list(features.keys())[0]])
    for _ in range(total - k):
        for k in features.keys():
            features[k].append(np.nan)
    return pd.DataFrame(features)

def _temporal_impute(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in train and test data using temporal grid wise data.
    """
    logger.info("Performing temporal grid wise imputation...")
    train_df['datetime_p'] = train_metadata['datetime'].apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
    )
    test_df['datetime_p'] = test_metadata['datetime'].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
    )
    for idx in tqdm(range(train_df.shape[0])):
        el = train_df.iloc[idx]
        grid_id = train_metadata.iloc[idx]['grid_id']
        indices = train_metadata[train_metadata['grid_id'] == grid_id].index
        cur_train_df = train_df.loc[indices]
        dt = el['datetime_p']
        
        possible = cur_train_df[cur_train_df['datetime_p'] < dt].sort_values('datetime_p', ascending=False).reset_index()
        if len(possible) == 0:
            continue
        el_ = possible.iloc[0]        
        train_df.iloc[idx] = el.fillna(el_)    
        
    for idx in tqdm(range(test_df.shape[0])):
        el = test_df.iloc[idx]
        grid_id = test_metadata.iloc[idx]['grid_id']
        indices = test_metadata[test_metadata['grid_id'] == grid_id].index
        cur_test_df = test_df.loc[indices]
        dt = el['datetime_p']
        
        possible = cur_test_df[cur_test_df['datetime_p'] < dt].sort_values('datetime_p', ascending=False).reset_index()
        if len(possible) == 0:
            continue
        el_ = possible.iloc[0]        
        test_df.iloc[idx] = el.fillna(el_)      
        
    train_df = train_df.drop(columns=['datetime_p'])
    test_df = test_df.drop(columns=['datetime_p'])
    
    return train_df, test_df

def _impute(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in train and test data using grid wise mean data.
    """
    logger.info("Performing grid wise mean imputation...")
    feat_columns = [col for col in train_df.columns if col != 'grid_id']

    for grid_id in train_metadata['grid_id'].unique():
        for col in feat_columns:
            indices = train_df[train_df['grid_id'] == grid_id].index
            mean_val = train_df.loc[indices, col].mean()
            train_df.loc[indices, col] = train_df.loc[indices, col].fillna(mean_val)
            
            indices = test_df[test_df['grid_id'] == grid_id].index
            test_df.loc[indices, col] = test_df.loc[indices, col].fillna(mean_val)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    return train_df, test_df

def _load_metadata(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Loading metadata...")
    grid_ids = grid_df['grid_id'].values
    tzs = grid_df['tz'].values
    elevation_means = grid_df['elevation_mean'].values
    elevation_vars = grid_df['elevation_var'].values

    train_dts = pd.to_datetime(
        train_metadata['datetime'], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    test_dts = pd.to_datetime(
        test_metadata['datetime'], format="%Y-%m-%dT%H:%M:%SZ", utc=True)

    for i in range(len(grid_ids)):
        indices = train_df[train_df['grid_id'] == grid_ids[i]].index
        train_df.loc[indices, 'elevation_mean'] = elevation_means[i]
        train_df.loc[indices, 'elevation_var'] = elevation_vars[i]
        
        loc_train_dts = train_dts.loc[indices]
        loc_train_dts = loc_train_dts.dt.tz_convert(tzs[i])
        train_df.loc[indices, 'month'] = loc_train_dts.apply(lambda x: x.month)
        train_df.loc[indices, 'day'] = loc_train_dts.apply(lambda x: x.day)

        indices = test_df[test_df['grid_id'] == grid_ids[i]].index
        test_df.loc[indices, 'elevation_mean'] = elevation_means[i]
        test_df.loc[indices, 'elevation_var'] = elevation_vars[i]

        loc_test_dts = test_dts.loc[indices]
        loc_test_dts = loc_test_dts.dt.tz_convert(tzs[i])
        test_df.loc[indices, 'month'] = loc_test_dts.apply(lambda x: x.month)
        test_df.loc[indices, 'day'] = loc_test_dts.apply(lambda x: x.day)

    train_df['location'] = train_metadata['grid_id'].apply(
        lambda x: grid_df[grid_df['grid_id'] == x]['location'].values[0])
    test_df['location'] = test_metadata['grid_id'].apply(
        lambda x: grid_df[grid_df['grid_id'] == x]['location'].values[0])
    location_encoding = {
        x: i for i, x in enumerate(grid_df['location'].unique())
    }
    train_df['location'] = train_df['location'].apply(
        lambda x: location_encoding[x])
    test_df['location'] = test_df['location'].apply(
        lambda x: location_encoding[x])

    return train_df, test_df

def _load_temporal_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
):
    grid_ins_count = defaultdict(lambda: 0)
    temporal_data = defaultdict(lambda: [])
    temporal_columns = [col for col in train_df.columns if col.endswith('_mean')]

    logger.info("Loading temporal features...")
    for idx in tqdm(range(train_df.shape[0])):
        grid_id = train_metadata.iloc[idx]['grid_id']
        indices = train_metadata[train_metadata['grid_id'] == grid_id].index
        if grid_ins_count[grid_id] == 0:
            for col in temporal_columns:
                temporal_data[col + "_temporal_diff"].append(0)
        else:
            prev_idx = indices[grid_ins_count[grid_id] - 1]
            el1 = train_df.iloc[idx]
            el2 = train_df.iloc[prev_idx]
            for col in temporal_columns:
                temporal_data[col + "_temporal_diff"].append(
                    el1[col] - el2[col]
                )
        grid_ins_count[grid_id] += 1
    train_df = pd.concat((train_df, pd.DataFrame(temporal_data)), axis=1)

    grid_ins_count = defaultdict(lambda: 0)
    temporal_data = defaultdict(lambda: [])
    for idx in tqdm(range(test_df.shape[0])):
        grid_id = test_metadata.iloc[idx]['grid_id']
        indices = test_metadata[test_metadata['grid_id'] == grid_id].index
        if grid_ins_count[grid_id] == 0:
            for col in temporal_columns:
                temporal_data[col + "_temporal_diff"].append(0)
        else:
            prev_idx = indices[grid_ins_count[grid_id] - 1]
            el1 = test_df.iloc[idx]
            el2 = test_df.iloc[prev_idx]
            for col in temporal_columns:
                temporal_data[col + "_temporal_diff"].append(
                    el1[col] - el2[col]
                )
        grid_ins_count[grid_id] += 1
    test_df = pd.concat((test_df, pd.DataFrame(temporal_data)), axis=1)

    return train_df, test_df

def prepare_dataset(
    data_products: dict,
    train_metafile: str, test_metafile: str, grid_metafile: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features from data.
    """

    check_existance(train_metafile)
    check_existance(test_metafile)
    check_existance(grid_metafile)

    train_dfs = []
    test_dfs = []
    for product in data_products.keys():
        check_existance(data_products[product]['train_file'])
        check_existance(data_products[product]['test_file'])

        cur_train_df = pd.read_csv(data_products[product]['train_file'])
        cur_test_df = pd.read_csv(data_products[product]['test_file'])

        if data_products[product]['drop']:
            cur_train_df.drop(columns=data_products[product]['drop'], inplace=True)
            cur_test_df.drop(columns=data_products[product]['drop'], inplace=True)

        columns = list(cur_train_df.columns)
        columns = [f"{product}_{col}" for col in columns]

        cur_train_df.columns = columns
        cur_test_df.columns = columns

        train_dfs.append(cur_train_df)
        test_dfs.append(cur_test_df)
    
    train_df = pd.concat(train_dfs, axis=1)
    test_df = pd.concat(test_dfs, axis=1)
    train_metadata = pd.read_csv(train_metafile)
    test_metadata = pd.read_csv(test_metafile)

    train_df['grid_id'] = train_metadata['grid_id']
    test_df['grid_id'] = test_metadata['grid_id']

    # print(f"Train Nan values: \n{train_df.isna().sum()}")
    # print(f"Test Nan values: \n{test_df.isna().sum()}")

    train_df['row_nan_count'] = train_df.isna().sum(axis=1)
    test_df['row_nan_count'] = test_df.isna().sum(axis=1)

    grid_data = pd.read_csv(grid_metafile)

    train_df, test_df = _impute(train_df, test_df, train_metadata, test_metadata)
    train_df, test_df = _load_temporal_features(train_df, test_df, grid_data, train_metadata, test_metadata)

    train_df['mean_value'] = train_metadata['grid_id'].apply(
    lambda x: train_metadata[train_metadata['grid_id'] == x]['value'].mean()
    )
    test_df['mean_value'] = test_metadata['grid_id'].apply(
        lambda x: train_metadata[train_metadata['grid_id'] == x]['value'].mean()
    )

    train_df, test_df = _load_metadata(train_df, test_df, grid_data, train_metadata, test_metadata)
    train_df = train_df.drop(columns=['grid_id'])
    test_df = test_df.drop(columns=['grid_id'])

    train_df['label'] = train_metadata['value']
    test_df['label'] = test_metadata['value']

    train_df.to_csv('tmptrain_features.csv', index=False)
    test_df.to_csv('tmptest_features.csv', index=False)

    return train_df, test_df