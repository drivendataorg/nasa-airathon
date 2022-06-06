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
from sklearn.linear_model import LinearRegression

from src.data import prepare_dataset
from src.models import run_kfold, feature_selection
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.utils import *
from loguru import logger
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str, help='Path to the config file', default='configs/pipeline_0.yml')
args = parser.parse_args()

def get_model(model):
    if 'xgb' in model:
        return XGBRegressor
    elif 'catb' in model:
        return CatBoostRegressor
    elif 'lgbm' in model:
        return LGBMRegressor
    elif 'linreg' in model:
        return LinearRegression
    else:
        raise ValueError(f'Model {model} not supported')

def get_params(model, config):
    if model == 'xgb':
        params = {}
    elif model == 'catb':
        params = {'verbose': 0}
    elif model == 'lgbm':
        params = {}
    elif model == 'xgb_tuned':
        params = config.XGB_PARAMS
    elif model == 'catb_tuned':
        params = config.CATB_PARAMS
    elif model == 'lgbm_tuned':
        params = config.LGBM_PARAMS
    elif model == 'linreg':
        params = {}
    else:
        raise ValueError(f'Model {model} not supported for params')
    return params

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
    # train_df = pd.read_csv('train_features.csv')
    # test_df = pd.read_csv('test_features.csv')

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
    features = train_df.columns.to_list()

    # ============================== F E A T U R E  S E L E C T I O N ============================== #
    if config.FEATURE_SELECTION:
        logger.info(f"Running feature selection on {config.FS_SAMPLE_PERCENTAGE * 100}% of the data...")
        config.FS_SAMPLE_PERCENTAGE = int(config.FS_SAMPLE_PERCENTAGE * len(train_df))
        fs_indices = train_df.sample(config.FS_SAMPLE_PERCENTAGE).index
        features = feature_selection(
            train_df.loc[fs_indices], 
            train_labels[fs_indices], 
            [get_model(model) for model in config.FS_MODELS],
            config.FS_SEEDS,
            [get_params(model, config) for model in config.FS_MODELS],
            config.N_FOLDS,
            features_threshold=config.FEATURES_THRESHOLD,
            topk_features=config.TOPK_FEATURES
        )

    train_features = train_df[features].to_numpy()
    test_features = test_df[features].to_numpy()

    logger.info(f"Using following features: {features}")
    logger.info(f"Found {len(train_features)} training instances")

    # ============================== L E V E L - 0  T R A I N I N G ============================== #
    train_level0_oof = defaultdict()
    test_level0_oof = defaultdict()
    level0_feat_importance = np.zeros((len(features)))
    logger.info(f"Starting level 0 training...")
    tim = Timer()

    for i, model in enumerate(config.MODELS):
        for j, seed in enumerate(config.SEEDS):
            feat_importances = np.zeros((len(train_features[0])))
            train_preds = np.zeros((len(train_features)))
            test_preds = np.zeros((len(test_features)))
            oof_preds = np.zeros((len(train_features)))

            for loc in train_metadata['location'].unique():
                logger.info(f"Training {model} model with {seed} seed on {loc} location...")

                train_indices = train_metadata[train_metadata['location'] == loc].index
                test_indices = test_metadata[test_metadata['location'] == loc].index

                cur_train_preds, cur_test_preds, cur_oof_preds, cur_feat_importances = run_kfold(
                    train_features[train_indices], 
                    train_labels[train_indices],
                    test_features[test_indices], config.N_FOLDS,
                    get_model(model), get_params(model, config), config.OUTPUT_PATH, name=f"{model}_{seed}_{loc}",
                    seed=seed
                )

                train_preds[train_indices] = cur_train_preds
                test_preds[test_indices] = cur_test_preds
                oof_preds[train_indices] = cur_oof_preds
                feat_importances += cur_feat_importances

            level0_feat_importance += feat_importances
            train_level0_oof['preds_' + model + '_' + str(seed)] = oof_preds
            test_level0_oof['preds_' + model + '_' + str(seed)] = test_preds

            metrics = compute_metrics(train_preds, train_labels)
            for k, v in metrics.items():
                logger.info(f"Average train_{k}: {np.mean(v)}")
            metrics = compute_metrics(oof_preds, train_labels)
            for k, v in metrics.items():
                logger.info(f"Average eval_{k}: {np.mean(v)}")

    level0_feat_importance = [
        [features[i], level0_feat_importance[i]] for i in range(len(features))
    ]
    level0_feat_importance.sort(key=lambda x: x[1])
    level0_feat_importance = np.array(level0_feat_importance)
    plt.barh(level0_feat_importance[:, 0], level0_feat_importance[:, 1])
    plt.yticks(fontsize='xx-small')
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'l0_feature_importance.png'))
    # plt.show()

    logger.info(tim.beep("Level 0 training finished in "))
    
    train_level0_oof = pd.DataFrame(train_level0_oof)
    test_level0_oof = pd.DataFrame(test_level0_oof)
    train_level0_oof.to_csv(os.path.join(config.OUTPUT_PATH, 'train_level0_oof.csv'), index=False)
    test_level0_oof.to_csv(os.path.join(config.OUTPUT_PATH, 'test_level0_oof.csv'), index=False)

    submission = pd.read_csv(config.TEST_METAFILE)
    submission['value'] = test_level0_oof.mean(axis=1)
    submission.to_csv(os.path.join(config.OUTPUT_PATH, f'l0avg_submission_{TIMESTAMP}.csv'), index=False)

    # ============================== L E V E L - 1  T R A I N I N G ============================== #
    logger.info(f"Starting level 1 training...")

    train_preds, test_preds, oof_preds, feat_importances = run_kfold(
        train_level0_oof.to_numpy(), train_labels, 
        test_level0_oof.to_numpy(), config.N_FOLDS,
        get_model(config.BLENDER), {}, config.OUTPUT_PATH, name=config.BLENDER,
        seed=config.BLENDER_SEED
    )
    metrics = compute_metrics(train_preds, train_labels)
    for k, v in metrics.items():
        logger.info(f"Average train_{k}: {np.mean(v)}")
    metrics = compute_metrics(oof_preds, train_labels)
    for k, v in metrics.items():
        logger.info(f"Average eval_{k}: {np.mean(v)}")

    plt.clf()
    plt.barh(train_level0_oof.columns, feat_importances)
    plt.yticks(fontsize='xx-small')
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'l1_feature_importance.png'))

    logger.info(tim.beep("Level 1 training finished in "))

    submission = pd.read_csv(config.TEST_METAFILE)
    submission['value'] = test_preds
    submission.to_csv(os.path.join(config.OUTPUT_PATH, f'submission_{TIMESTAMP}.csv'), index=False)
