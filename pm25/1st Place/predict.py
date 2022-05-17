# General imports
import yaml
import pickle
import warnings
warnings.filterwarnings('ignore')
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from src.inference import (
    get_maiac_data, get_misr_data, get_gfs_data
)

from src.utils import *
from loguru import logger

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', dest='config', type=str, help='Path to the config file', default='configs/predict.yml')
parser.add_argument(
    '--observation_start_time', dest='obs_start_time', type=str, help="Observation start time in YYYY-MM-DDTHH:mm:ssZ format", default="2021-03-02T18:30:00Z")
parser.add_argument(
    '--grid_id', dest='grid_id', type=str, help='Grid ID', default="C7PGV")
parser.add_argument(
    '--satdata_file', dest='satdata_file', type=str, help='Path to the satellite data file which contains maiac and misr satellite data information', default="data/pm25_satellite_metadata.csv")
parser.add_argument(
    '--ncar_email', dest='ncar_email', type=str, help='Email address to login to NCAR website to download GFS data', required=True)
parser.add_argument(
    '--ncar_pswd', dest='ncar_pswd', type=str, help='Password to login to NCAR website to download GFS data', required=True)
args = parser.parse_args()

def load_temporal_features(features):
    new_features = defaultdict(lambda: [])
    for col in features.columns:
        new_features[col].append(
            features.iloc[1][col]
        )
    for col in features.columns:
        if col.endswith('_mean'):
            new_features[col + "_temporal_diff"].append(
                features.iloc[1][col] - features.iloc[0][col] 
            )
    return pd.DataFrame(new_features)

def load_metadata(config, features, grid_metadata):
    features['datetime'] = [config.OBS_START_TIME]
    features['datetime'] = pd.to_datetime(
        features['datetime'], utc=True)
    tz = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['tz'].values[0]
    features['datetime'] = features['datetime'].dt.tz_convert(tz)
    features['elevation_mean'] = [
        grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['elevation_mean'].values[0]]
    features['elevation_var'] = [
        grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['elevation_var'].values[0]]
    features['month'] = features['datetime'].apply(lambda x: x.month)
    features['day'] = features['datetime'].apply(lambda x: x.day)
    return features.drop(columns=['datetime'])   

if __name__ == '__main__':
	# Config
    with open(args.config, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    config.OBS_START_TIME = datetime.datetime.strptime(args.obs_start_time, '%Y-%m-%dT%H:%M:%SZ')
    config.GRID_ID = args.grid_id
    config.SATELLITE_FILE = args.satdata_file
    config.NCAR_EMAIL = args.ncar_email
    config.NCAR_PSWD = args.ncar_pswd
    # logger.info(f"Config:{str(config)}")
    # logger.info(f"Observation start time: {config.OBS_START_TIME}")
    # logger.info(f"Grid ID: {config.GRID_ID}")

    # ============================== L O A D I N G  D A T A ============================== #
    grid_metadata = pd.read_csv(config.GRID_METAFILE)
    sat_data = pd.read_csv(config.SATELLITE_FILE)
    sat_data['time_end'] = sat_data['time_end'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00')
    )
    assert config.GRID_ID in grid_metadata['grid_id'].values, f"Grid ID {config.GRID_ID} not found in grid metadata"

    maiac_data = get_maiac_data(config, sat_data, grid_metadata)
    maiac_data.columns = map(lambda x: f"maiac_{x}", maiac_data.columns)
    misr_data = get_misr_data(config, sat_data, grid_metadata)
    misr_data.columns = map(lambda x: f"misr_{x}", misr_data.columns)
    gfs_data = get_gfs_data(config, grid_metadata)
    gfs_data.columns = map(lambda x: f"gfs_{x}", gfs_data.columns)

    df = pd.concat([maiac_data, misr_data, gfs_data], axis=1)

    # ============================== P R E P A R I N G  D A T A ============================== #
    df['row_nan_count'] = df.isna().sum(axis=1)
    for col in df.columns:
        _fill_value = config.GRID_WISE_MEAN_IMPUTATION[config.GRID_ID][col]
        if _fill_value != "nan":
            df[col] = df[col].fillna(_fill_value)
    df = df.fillna(0)
    
    df = load_temporal_features(df)
    df['mean_value'] = [config.GRID_MEAN_VALUES[config.GRID_ID]]
    df = load_metadata(config, df, grid_metadata)
    df['location'] = [config.LOCATION_ENCODING[grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['location'].values[0]]]
    df = df.to_numpy()

    # ============================== P R E D I C T I N G ============================== #

    # P I P E L I N E - 1
    LOG_DIR = config.PIPE1_DIR
    assert os.path.exists(LOG_DIR), f"{LOG_DIR} does not exist, please make sure you downloaded the weights."
    models = [
        'xgb_tuned_None',
        'catb_tuned_None',
        'lgbm_tuned_None',
    ]
    features = defaultdict(lambda: np.zeros(len(df)))
    for name in models:
        for fold in range(1, 6):
            model_file = f"{LOG_DIR}/{name}_fold-{fold}.pkl"
            reg = pickle.load(open(model_file, "rb"))
            features[name] += (reg.predict(df) / 5)
    features = pd.DataFrame(features).to_numpy()
    # print(features)

    pipeline_1_preds = 0
    for fold in range(1, 6):
        model_file = f"{LOG_DIR}/linreg_42_fold-{fold}.pkl"
        reg = pickle.load(open(model_file, "rb"))
        pipeline_1_preds += (reg.predict(features) / 5)
    
    # P I P E L I N E - 2
    LOG_DIR = config.PIPE2_DIR
    assert os.path.exists(LOG_DIR), f"{LOG_DIR} does not exist, please make sure you downloaded the weights."
    location = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['location'].values[0]
    models = [
        f'xgb_tuned_42_{location}_42',
        f'catb_tuned_42_{location}_42',
        f'lgbm_tuned_42_{location}_42'
    ]
    features = defaultdict(lambda: np.zeros(len(df)))
    for name in models:
        for fold in range(1, 6):
            model_file = f"{LOG_DIR}/{name}_fold-{fold}.pkl"
            reg = pickle.load(open(model_file, "rb"))
            features[name] += (reg.predict(df) / 5)
    features = pd.DataFrame(features).to_numpy()
    # print(features)

    pipeline_2_preds = 0
    for fold in range(1, 6):
        model_file = f"{LOG_DIR}/linreg_42_fold-{fold}.pkl"
        reg = pickle.load(open(model_file, "rb"))
        pipeline_2_preds += (reg.predict(features) / 5)
    
    pipeline_1_preds = pipeline_1_preds.squeeze()
    pipeline_2_preds = pipeline_2_preds.squeeze()
    # print(f"Pipeline 1: {pipeline_1_preds}")
    # print(f"Pipeline 2: {pipeline_2_preds}")
    print(f"\nFinal prediction: {(pipeline_1_preds + pipeline_2_preds) / 2}")
    