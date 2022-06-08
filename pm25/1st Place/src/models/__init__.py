from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.utils import Timer
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import os

def run_kfold(
	train_features: np.ndarray, 
    train_labels: np.ndarray, 
    test_features: np.ndarray,
    n_folds: int, 
    model: object, 
    model_params: dict,
	save_dir: str, 
    name: str='model', 
    seed=42
):  
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds, 
            shuffle=True,
            random_state=seed
        )
    oof_preds = np.zeros((len(train_labels)))
    train_preds = np.zeros((len(train_labels)))
    test_preds = np.zeros((len(test_features)))
    feat_importances = np.zeros((len(train_features[0])))
    tim = Timer()

    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        x_train, x_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        reg = model(**model_params)
        reg.fit(x_train, y_train)
        
        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds[train_idx] += preds
        
        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds[val_idx] += preds
        
        # Prediction on test data
        preds = reg.predict(test_features)
        test_preds += preds

        # Feature importance
        if hasattr(reg, 'feature_importances_'):
            feat_importances += reg.feature_importances_
        else:
            feat_importances += reg.coef_

        pickle.dump(
            reg, 
            open(os.path.join(save_dir, f"{name}_{seed}_fold-{fold + 1}.pkl"), "wb")
        )
        bar.update()
        
    train_preds /= (n_folds - 1)
    test_preds /= n_folds
    feat_importances /= n_folds

    return train_preds, test_preds, oof_preds, feat_importances

def run_kfold_trainonly(
	train_features: np.ndarray, 
    train_labels: np.ndarray, 
    n_folds: int, 
    model: object, 
    model_params: dict,
    seed=42
):  
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds, 
            shuffle=True,
            random_state=seed
        )
    oof_preds = []
    oof_labels = []
    train_preds = np.zeros((len(train_labels)))
    feat_importances = np.zeros((len(train_features[0])))
    tim = Timer()

    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        x_train, x_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        reg = model(**model_params)
        reg.fit(x_train, y_train)
        
        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds[train_idx] += preds
        
        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds.extend(preds.tolist())
        oof_labels.extend(y_val.tolist())

        # Feature importance
        feat_importances += reg.feature_importances_

        bar.update()
        
    train_preds /= (n_folds - 1)
    feat_importances /= n_folds

    return train_preds, oof_preds, oof_labels, feat_importances

def feature_selection(
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    models: list,
    seeds: list,
    model_params: list,
    n_folds: int,
    features_threshold=None,
    topk_features=None
):  
    features = train_df.columns.tolist()
    total_features_importance = np.zeros((len(features)))
    for i, model in enumerate(models):
        for j, seed in enumerate(seeds):
            train_features = train_df.to_numpy()
            _, _, _, feat_imp = run_kfold_trainonly(
                train_features,
                train_labels,
                n_folds,
                model,
                model_params[i],
                seed=seed
            )
            total_features_importance += feat_imp

    total_features_importance /= (len(models) * len(seeds))
    total_features = [
        [features[i], total_features_importance[i]] for i in range(len(features))
    ]
    total_features.sort(key=lambda x: x[1], reverse=True)

    # plt.barh([el[0] for el in total_features], [el[1] for el in total_features])
    # plt.yticks(fontsize='xx-small')
    # plt.show()

    if topk_features:
        return [el[0] for el in total_features[:topk_features]]
    elif features_threshold:
        return_features = []
        for f in total_features:
            if f[1] < features_threshold:
                break
            return_features.append(f[0])
        return return_features
    else:
        raise ValueError("Either topk_features or features_threshold must be specified")
