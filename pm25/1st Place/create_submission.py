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

from src.data import prepare_dataset

from src.utils import *
from loguru import logger

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', dest='config', type=str, help='Path to the config file', default='configs/pipeline_0.yml')
args = parser.parse_args()

if __name__ == '__main__':
	# Config
    with open(args.config, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    # ============================== L O A D I N G  D A T A ============================== #
    logger.info("Loading data...")
    _, df = prepare_dataset(
        config.DATA_PRODUCTS,
        config.TRAIN_METAFILE, config.TEST_METAFILE, config.GRID_METAFILE
    )
    df = df.drop(columns=['label'])
    df = df.to_numpy()

    grid_metadata = pd.read_csv(config.GRID_METAFILE)
    sub_df = pd.read_csv(config.TEST_METAFILE)
    sub_df['location'] = sub_df['grid_id'].apply(
        lambda x: grid_metadata[grid_metadata['grid_id'] == x]['location'].values[0]
    )

    # ============================== P R E D I C T I N G ============================== #
    # P I P E L I N E - 1
    LOG_DIR = "models/pipe_1"
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
    LOG_DIR = "models/pipe_2"
    features = defaultdict(lambda: np.zeros(len(df)))
    models = [
        f'xgb_tuned_42',
        f'catb_tuned_42',
        f'lgbm_tuned_42'
    ]

    for location in grid_metadata['location'].unique():
        indices = sub_df[sub_df['location'] == location].index
        cur_df = df[indices]
        for name in models:
            for fold in range(1, 6):
                model_file = f"{LOG_DIR}/{name}_{location}_42_fold-{fold}.pkl"
                reg = pickle.load(open(model_file, "rb"))
                features[name][indices] += (reg.predict(cur_df) / 5)
    features = pd.DataFrame(features).to_numpy()
    # print(features)

    pipeline_2_preds = 0
    for fold in range(1, 6):
        model_file = f"{LOG_DIR}/linreg_42_fold-{fold}.pkl"
        reg = pickle.load(open(model_file, "rb"))
        pipeline_2_preds += (reg.predict(features) / 5)

    sub_df['value'] = (pipeline_1_preds + pipeline_2_preds) / 2
    sub_df.drop(columns=['location'], inplace=True)
    sub_df.to_csv('data/proc/submission.csv', index=False)
    
