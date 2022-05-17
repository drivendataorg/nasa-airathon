#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import zstandard as zstd


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)


from joblib import Parallel, delayed


from collections import defaultdict


from sklearn.metrics import mean_squared_error





dataset = 'tg'





zd = zstd.ZstdDecompressor()


def loadFile(model_path, file):
    try:
        zd = zstd.ZstdDecompressor()
        return pickle.loads(zd.decompress(open(model_path + file, 'rb').read()))
    except Exception as e:
        print(e)
        return pd.Series([])





def loadPreds(dataset, location):
    model_path = 'nn1/{}_{}/'.format(dataset, location.replace(' ', '_'))
    files = sorted([f for f in os.listdir(model_path) if 'ckpt' not in f
                       and '9-' in f
                   ])
    print(len(files), 'files for {}-{}'.format(dataset, location))
    
    r = Parallel(os.cpu_count() * 2)(delayed(loadFile)(model_path, file) for file in files)
    all_preds = list(zip(files, r))    
    
    all_val_preds = defaultdict(list)
    all_test_preds = defaultdict(list)
    for file, preds in all_preds:
        m = file.split('_run')[0]
        e = file.split('epoch=')[-1].split('-')[0]
        mstr = m + '_epoch{}'.format(e)

        vt = file.split('.pkl')[0].split('-')[-1]
        if vt == 'val':
            all_val_preds[mstr].append(preds)
        elif vt == 'test' and '_full' in file:
            all_test_preds[mstr].append(preds)

    val_preds = {}; test_preds = {}
    for k, v in all_val_preds.items():
        v = pd.concat(v)
        v = v.groupby(v.index).mean()
        val_preds[k] = v
    for k, v in all_test_preds.items():
        v = pd.concat(v)
        v = v.groupby(v.index).mean()
        test_preds[k] = v
        
    print(len(test_preds))
    print(len(val_preds))
    
    test_preds = pd.DataFrame(test_preds)#[val_preds.columns]
    val_preds = pd.DataFrame(val_preds)[test_preds.columns]
    return test_preds, val_preds








all_data = pickle.load(open('cache/all_data_{}.pkl'.format(dataset), 'rb'))
submission = pickle.load(open('cache/submission_{}.pkl'.format(dataset), 'rb'))








lgb_files = [f for f in os.listdir('clfs_{}'.format(dataset)) 
                 if 'lgb' in f]
lgb_preds = pickle.load(open('clfs_{}/'.format(dataset) + lgb_files[0], 'rb'))[1]
lgb_preds = pd.concat(lgb_preds)

















# val_preds.corrwith(y).sort_values()[::-1][:20]





from sklearn.linear_model import ElasticNet


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error





import sklearn


import scipy


import random
import datetime


def score(wts, x, y, reg = 1, l1_ratio = 0):
    wts = wts /wts.sum() #/ max(wts.sum() ** 0.5, 1.0)#wts.sum() * 0.9
    blend = ( x * wts[None, :]).sum(axis = 1)
    return ( 
        mean_squared_error(y, blend)
            + reg *( (wts ** 2).sum() + l1_ratio * np.abs(wts).sum()) )


def optimize(x, y, reg = 1, l1_ratio = 0, tol = 1e-4 ):
    wts = scipy.optimize.minimize(
    score, np.ones(x.shape[1]) / x.shape[1],#len(x.columns), 
        tol = tol,
    args=(x, y, reg, l1_ratio), 
    bounds=[(0, 1) for i in range(x.shape[1])],#len(x.columns))],
    ).x
    return wts / wts.sum()# ** 0.5, 1.0)


class CLR(sklearn.base.BaseEstimator):
    def __init__(self, reg = 1.0, l1_ratio = 0, tol = 1e-4):
        self.reg = reg
        self.l1_ratio = l1_ratio
        self.classes_ = np.array((0, 1))
        self.tol = tol
    
    def fit(self, X, y):
        wts = optimize(X.values, y.values, 
                           self.reg, self.l1_ratio, self.tol)
        self.wts = wts #/ max(wts.sum(), 1)# * 0.9
        # print(self.wts.sum())
        
    def predict(self, X):
        return (X * self.wts).sum(axis = 1).values


clr_params = {'reg': [ 1e-4, 3e-3, 1e-3, 3e-3, 
                      0.01, 0.03, 0.1, 0.3, 1, 3,  ],
               'l1_ratio': [ 0, 0.01, 0.03, 0.1, 0.2, 0.5, ]}





class PurgedKFold():
    def __init__(self, n_splits=5, gap = 30):
        self.n_splits = n_splits
        self.gap = gap
        
    def get_n_splits(self, X, y = None, groups = None): return self.n_splits
    
    def split(self, X, y=None, groups=None):
        groups = groups.sort_values()
        X = X.reindex(groups.index)# sort_values(groups)
        y = y.reindex(X.index);
                     
        X, y, groups = sklearn.utils.indexable(X, y, groups)
        indices = np.arange(len(X))
        
        n_splits = self.n_splits
        for i in range(n_splits):
            test = indices[ i * len(X) // n_splits: (i + 1) * len(X) // n_splits ]#.index
            train = indices[ (groups <= groups.iloc[test].min() - datetime.timedelta(days = self.gap) )
                          | (groups >= groups.iloc[test].max() + datetime.timedelta(days = self.gap) ) ]#.index
            yield train, test

class RepeatedPurgedKFold():
    def __init__(self, n_splits = 5, n_repeats = 1, gap = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.gap = gap
        
    def get_n_splits(self, X, y = None, groups = None): 
        return self.n_splits * self.n_repeats + self.n_repeats * ( self.n_repeats - 1) // 2
    
    def split(self, X, y=None, groups=None):
        for i in range(self.n_repeats):
            for f in PurgedKFold(self.n_splits + i, gap = self.gap if self.gap else None).split(X, y, groups):
                yield f
    





for location in ['Delhi', 'Los Angeles (SoCAB)', 'Taipei']:    
    test_preds, val_preds = loadPreds('tg', location)

    y = all_data.value.reindex(val_preds.index)
    g = all_data.datetime.reindex(y.index)
    # print(mean_squared_error(y, val_preds.mean(axis = 1)))
    print(np.corrcoef(val_preds.mean(axis = 1), y )[0, 1])


    lgb_val_preds = lgb_preds.groupby(lgb_preds.index).mean()
    lgb_val_preds = lgb_val_preds.reindex(y.index)
    lgb_val_preds

    lgb_test_preds = pd.read_csv('submissions_{}/new.csv'.format(dataset))
    submission = lgb_test_preds
    lgb_test_preds = lgb_test_preds.value.reindex(test_preds.index)
    # print(mean_squared_error(y, lgb_val_preds ))
    print(np.corrcoef(lgb_val_preds, y)[0, 1])
    print();

    for i in range(10):
        val_preds['lgb{}'.format(i)] = lgb_val_preds
        test_preds['lgb{}'.format(i)] = lgb_test_preds

    enet_val_preds = []
    enet_test_preds = []
    all_coefs = []

    for i in range(8):
        folds = PurgedKFold( random.randrange(4, 7)
                                        if dataset == 'pm' else 
                                        random.randrange(3, 4),
                                    gap = random.randrange(20, 40)).split(
                    val_preds, y, g)
        for train_fold, test_fold in folds:
            # l = random.randrange(0, len(train_fold)//random.randrange(5, 20))
            # s = random.randrange(0, len(train_fold) - l)
            # train_fold = train_fold[:s].tolist() + train_fold[s + l:].tolist()
            vp = val_preds.copy()
            model_drops = random.sample(
                [c for c in list(val_preds.columns) if 'lgb' not in c], 
                   k = int( (0.4 + 0.2 * random.random()) 
                               * len(val_preds.columns) ))
            vp.loc[:, model_drops] = 0
            model = RandomizedSearchCV(
                CLR(#tol = 3e-3,
                   ),  clr_params, cv = RepeatedPurgedKFold( 
                                            random.randrange(4, 7)
                                                    if dataset == 'pm' else 
                                                    random.randrange(3, 6),

                                            random.randrange(1, 3),
                                                 gap = random.randrange(20, 60)),
                        scoring = 'neg_mean_squared_error',
                random_state = datetime.datetime.now().microsecond,
                        n_iter = 4, n_jobs = -1,
            )
            model.fit(vp.iloc[train_fold], y.iloc[train_fold], 
                          groups = g.iloc[train_fold])
            enet_val_preds.append(pd.Series(
                model.predict(val_preds.iloc[test_fold]), 
                                  val_preds.index[test_fold]))
            enet_test_preds.append(pd.Series(
                model.predict(test_preds), test_preds.index))
            enet_val_preds[-1]

            print(np.corrcoef(        enet_val_preds[-1], y.iloc[test_fold])[0, 1])

            clf = model.best_estimator_
            print(clf)
            all_coefs.append(pd.Series(clf.wts, val_preds.columns))
    
    all_coefs = pd.concat(all_coefs)
    all_coefs = all_coefs.groupby(all_coefs.index).mean()
    
    enet_val_preds = (val_preds * all_coefs).sum(axis = 1)
    enet_test_preds = (test_preds * all_coefs).sum(axis = 1)
    
    
    print()
    print(np.corrcoef(y, enet_val_preds)[0, 1])
    print()

    os.makedirs('stack1', exist_ok = True)
    pickle.dump(all_coefs, 
                open('stack1/{}_{}.pkl'.format(dataset, location), 'wb'))

    submission = submission.reindex(enet_test_preds.index)
    submission.value = enet_test_preds

    os.makedirs('submissions_{}/nn1'.format(dataset), exist_ok = True)
    submission.to_csv('submissions_{}/nn1/{}.csv'.format(
                            dataset, location.replace(' ', '_')))








combined = pd.concat([pd.read_csv('submissions_{}/nn1/'.format(
                        dataset) + file, index_col = 0)
            for file in os.listdir('submissions_{}/nn1/'.format(
                        dataset,))]).sort_index()


combined.to_csv('submissions_{}/stack1.csv'.format(dataset), index = False)


# prior = pd.read_csv('../submissions_{}/stack1.csv'.format(dataset))#, index_col = 0)


# prior


# combined


# np.corrcoef(prior.value, combined.value)[0, 1]




