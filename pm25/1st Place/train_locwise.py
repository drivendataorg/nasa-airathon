# General imports
import os
import sys
import cv2
import time
import glob
import json
import yaml
import scipy
import random
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
from src.models import run_kfold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.utils import *
from loguru import logger
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str, help='Path to the config file', default='configs/model_0.yml')
args = parser.parse_args()

if __name__ == '__main__':
	# Config
    with open(args.config, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, TIMESTAMP)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    logger.add(os.path.join(config.OUTPUT_PATH, 'logs.log'))
    logger.info(f"Config:{str(config)}")
    
    train_df, test_df = prepare_dataset(
        config.DATA_PRODUCTS,
        config.TRAIN_METAFILE, config.TEST_METAFILE, config.GRID_METAFILE
    )
    
    train_metadata = pd.read_csv(config.TRAIN_METAFILE)
    test_metadata = pd.read_csv(config.TEST_METAFILE)
    grid_metadata = pd.read_csv(config.GRID_METAFILE)

    train_metadata['location'] = train_metadata['grid_id'].apply(
        lambda x: grid_metadata[grid_metadata['grid_id'] == x]['location'].values[0]
    )
    test_metadata['location'] = test_metadata['grid_id'].apply(
        lambda x: grid_metadata[grid_metadata['grid_id'] == x]['location'].values[0]
    )

    train_labels = train_df['label'].to_numpy()
    train_df = train_df.drop(['label'], axis=1)
    test_df = test_df.drop(['label'], axis=1)

    train_features = train_df.to_numpy()
    test_features = test_df.to_numpy()

    logger.info(f"Using following features: {train_df.columns}")
    logger.info(f"Found {len(train_features)} training instances")
    if config.MODEL == 'xgboost':
        model = XGBRegressor
        model_params = config.XGB_PARAMS
    elif config.MODEL == 'catboost':
        model = CatBoostRegressor
        model_params = config.CATB_PARAMS
    elif config.MODEL == 'lightgbm':
        model = LGBMRegressor
        model_params = config.LGBM_PARAMS
    else:
        raise ValueError(f"Model {config.MODEL} not supported")

    logger.info(f"Training {config.MODEL} model")
    all_feat_importances = np.zeros((len(train_features[0])))
    all_train_preds = np.zeros((len(train_features)))
    all_test_preds = np.zeros((len(test_features)))
    all_oof_preds = np.zeros((len(train_features)))

    for loc in train_metadata['location'].unique():

        train_indices = train_metadata[train_metadata['location'] == loc].index
        test_indices = test_metadata[test_metadata['location'] == loc].index

        logger.info(f"Training model for location {loc} with {len(train_indices)} train instances and {len(test_indices)} test instances...")

        train_preds, test_preds, oof_preds, feat_importances = run_kfold(
            train_features[train_indices], 
            train_labels[train_indices], 
            test_features[test_indices], config.N_FOLDS,
            model, model_params, config.OUTPUT_PATH, name=config.MODEL + '_' + loc
        )
        all_train_preds[train_indices] = train_preds
        all_test_preds[test_indices] = test_preds
        all_oof_preds[train_indices] = oof_preds
        all_feat_importances += feat_importances

        metrics = compute_metrics(train_preds, train_labels[train_indices])
        for k, v in metrics.items():
            logger.info(f"Average train_{k}: {np.mean(v)}")
        metrics = compute_metrics(oof_preds, train_labels[train_indices])
        for k, v in metrics.items():
            logger.info(f"Average eval_{k}: {np.mean(v)}")

    metrics = compute_metrics(all_train_preds, train_labels)
    for k, v in metrics.items():
        logger.info(f"Overall Average train_{k}: {np.mean(v)}")
    metrics = compute_metrics(all_oof_preds, train_labels)
    for k, v in metrics.items():
        logger.info(f"Overall Average eval_{k}: {np.mean(v)}")
    
    submission = pd.read_csv(config.TEST_METAFILE)
    submission['value'] = all_test_preds
    submission.to_csv(
        os.path.join(config.OUTPUT_PATH, f'submission_{TIMESTAMP}.csv'), index=False
    )
    submission.head()

    all_feat_importances = [
        [train_df.columns[i], all_feat_importances[i]] for i in range(len(train_df.columns))
    ]
    all_feat_importances.sort(key=lambda x: x[1])
    level0_feat_importance = np.array(all_feat_importances)
    plt.barh(level0_feat_importance[:, 0], level0_feat_importance[:, 1])
    plt.yticks(fontsize='xx-small')
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'feature_importance.png'))
    plt.show()
