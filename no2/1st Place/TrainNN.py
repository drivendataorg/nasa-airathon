#!/usr/bin/env python
# coding: utf-8

import random, datetime


random.seed(datetime.datetime.now().microsecond)


dataset = random.choice([#'pm', 
                         'tg'
                        ])


max_epochs = 20





import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from adamp import AdamP, SGDP


import numpy as np
import pandas as pd


import datetime


import sklearn
from sklearn.preprocessing import StandardScaler


# import boto3
import pickle
import os
import secrets


from sklearn.metrics import mean_squared_error


import zstandard as zstd








zc = zstd.ZstdCompressor(level = 9)


# s3 = boto3.client('s3')


all_data = pickle.load(open('cache/all_data_{}.pkl'.format(dataset), 'rb'))
submission = pickle.load(open('cache/submission_{}.pkl'.format(dataset), 'rb'))


# pickle.dump(all_data, open('cache/all_data_{}.pkl'.format(dataset), 'wb'))
# pickle.dump(submission, open('cache/submission_{}.pkl'.format(dataset), 'wb'))


# all_data = pickle.loads(s3.get_object(Bucket = 'projects-v', 
#                                      Key = 'aqi/all_data_{}.pkl'.format(dataset) )
#                                           ['Body'].read())

# submission = pickle.loads(s3.get_object(Bucket = 'projects-v', 
#                                      Key = 'aqi/submission_{}.pkl'.format(dataset) )
#                                           ['Body'].read())


all_data.shape


submission.shape





if dataset == 'tg':
    np.random.seed(datetime.datetime.now().microsecond)
    all_data.loc[(all_data.grid_id == '7334C') & (np.random.random(len(all_data)) < 0.15), 'grid_id'] = '7F1D1'
    all_data.loc[(all_data.grid_id == 'HANW9') & (np.random.random(len(all_data)) < 0.15), 'grid_id'] = 'WZNCR'


grid_ids = sorted(all_data.grid_id.unique())


grid_dict = dict(zip(grid_ids, np.arange(len(grid_ids))))
# grid_dict





x = all_data[[c for c in all_data.columns if c not in ['datetime', 'value']]].copy()
xs = submission[x.columns].copy()
y = all_data.value.astype(np.float32)
d = all_data.datetime


x['dayinyear_sin'] = np.sin(x.dayinyear / 366 * 2 * np.pi)#.plot()
x['dayinyear_cos'] = np.cos(x.dayinyear / 366 * 2 * np.pi)#.plot()
xs['dayinyear_sin'] = np.sin(xs.dayinyear / 366 * 2 * np.pi)#.plot()
xs['dayinyear_cos'] = np.cos(xs.dayinyear / 366 * 2 * np.pi)#.plot()





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
    

















class AirDataset(Dataset):
    def __init__(self, x_loc, g_loc, y_loc, idxs, feature_drops = [] ):
        self.x = x_loc.iloc[idxs].drop(columns = feature_drops)
        self.g = g_loc.iloc[idxs]
        self.y = y_loc.iloc[idxs]

    def __getitem__(self, i):
        return self.g.iloc[i], self.x.iloc[i].values.astype(np.float32), self.y.iloc[i]

    def __len__(self):
        return len(self.y)


class path(nn.Module):
    def __init__(self, dims, lr, grid_dims, gn, input_dropout, dropout, ):
        super().__init__()
        self.gn = gn

        self.GridEmbedding = nn.Embedding(len(grid_dict), grid_dims);
        self.dropout0 = nn.Dropout(input_dropout)
        self.linear1 = nn.Linear(x_loc.shape[1] + grid_dims, dims, bias = False)
        self.bn1 = nn.GroupNorm(8, dims)
        self.a1 = nn.RReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(dims, dims, bias = False)
        self.bn2 = nn.GroupNorm(8, dims)
        self.a2 = nn.RReLU()

    def forward(self, g, x):
        g = self.GridEmbedding(g) * 0
        x = torch.cat((x, g), dim = 1)
        if self.training: x += torch.randn(x.shape) * self.gn
        x = self.a1( self.bn1( self.linear1( self.dropout0( x ))))
        if self.training: x += torch.randn(x.shape) * self.gn
        x = self.a2( self.bn2( self.linear2( self.dropout1( x )))) 
        return x


class AirModel(pl.LightningModule):
    def __init__(self, dims = 128, lr = 0.25, 
                 grid_dims = 8, gn = 0.1,
                 input_dropout = 0.2, dropout = 0.5,
                num_paths = 3):
        super().__init__()
        self.save_hyperparameters()
        self.gn = gn
        self.GridEmbedding = nn.Embedding(len(grid_dict), grid_dims);

        self.paths = nn.ModuleList([path(dims, lr, grid_dims, gn,
                                            input_dropout, dropout)
                                    for i in range(num_paths)])

        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(dims * num_paths, dims, bias = False)
        self.bn3 = nn.GroupNorm(8, dims)
        self.a3 = nn.PReLU()

        self.final_dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(dims * num_paths + grid_dims, 1)

    def forward(self, g, x):
        x = torch.cat([p.forward(g, x) for p in self.paths], dim = 1)
        # x = self.a3( self.bn3( self.linear3( self.dropout2( x )))) 
        g = self.GridEmbedding(g)
        # if self.training: x += torch.randn(x.shape) * self.gn
        # x = self.dropout2(x)
        x = torch.cat((x, g), dim = 1)
        x = self.final_linear(self.final_dropout( x ))
        return x[:, 0]

    def on_validation_epoch_start(self):
        self.y = []; self.yp = []

    def validation_step(self, batch, batch_idx):
        g, x, y = batch
        yp = self.forward(g, x) #* yscale
        self.y.append(y); self.yp.append(yp)
        loss = nn.MSELoss()(yp, y)
        return loss

    def training_step(self, batch, batch_idx):
        g, x, y = batch
        yp = self.forward(g, x) #* yscale
        # print(yp[:4], y[:4])
        loss = nn.MSELoss()(yp, y)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        y = torch.cat(self.y); yp = torch.cat(self.yp)
        loss = nn.MSELoss()(yp, y) ** 0.5
        print(loss)
        # self.log('val_loss', loss)


    def configure_optimizers(self):
        return AdamP(self.parameters(), 
                                lr = learning_rate,
                                weight_decay = weight_decay)





def savePreds(model_path, model_str):
    for e in range(5-1, max_epochs, 5):
        model_file = '{}{}-epoch={:02d}.ckpt'.format(
                            model_path, model_str, e)
        model = AirModel.load_from_checkpoint(model_file)
        model.eval();
        
        if 'full' not in model_file: os.remove(model_file)

        val_preds = []; val_y = []; test_preds = []
        with torch.no_grad():
            for g, x, y in val_loader:
                val_preds.append(model(g, x).numpy())
                val_y.append(y.numpy())
            for g, x, y in test_loader:
                test_preds.append(model(g, x).numpy())

        val_preds = np.concatenate(val_preds)
        test_preds = np.concatenate(test_preds)
        test_preds = pd.Series(test_preds, test_dataset.x.index)
        val_preds = pd.Series(val_preds, val_dataset.x.index)

        if 'fold' in model_str:
            print(mean_squared_error(val_dataset.y, val_preds) ** 0.5)
            with open(model_file.replace('.ckpt', '-val.pkl.zstd'), 'wb') as f:
                f.write(zc.compress(pickle.dumps(val_preds)))            
        
        with open(model_file.replace('.ckpt', '-test.pkl.zstd'), 'wb') as f:
            f.write(zc.compress(pickle.dumps(test_preds)))


# !jupyter nbconvert --to script 'TrainNN.ipynb'





for m in range(500):
    # Location / Data
    location = random.choice(['Delhi', 'Delhi',
                           'Los Angeles (SoCAB)', 'Taipei'
                         ])
    
    x_loc = x[x.location == location].drop(columns = ['location', 'grid_id'])
    y_loc = y.reindex(x_loc.index)
    d_loc = d.reindex(x_loc.index)
    g_loc = x.reindex(x_loc.index).grid_id.map(grid_dict).astype(np.int32)

    xs_loc = xs[xs.location == location].drop(columns = ['location', 'grid_id'])
    gs_loc = xs.reindex(xs_loc.index).grid_id.map(grid_dict).astype(np.int32)
    ys_loc = pd.Series(np.nan, index = xs_loc.index)
    
    # Params
    dims = random.choice([64, 128 if location == 'Delhi' else 64 ])
    gn = random.choice([0, 0.1, ])
    learning_rate = random.choice([3e-4, 1e-3,])
    weight_decay = random.choice([1e-4, 1e-3, 1e-2,])
    
    
    folds = list(PurgedKFold(random.choice([4, 5, 6]) if dataset == 'pm' 
                                    else random.choice([3, 4, 5]),
                                gap = random.randrange(20, 60)
                            ).split(x_loc, y_loc, d_loc)) 
    print('{} folds'.format(len(folds)))
    folds += [(np.arange(0, len(x_loc)), [])] 

    seed = datetime.datetime.now().microsecond
    random.seed(seed);
    for fold_idx, (train_fold, test_fold) in enumerate(folds):
        # yscale = y_loc.iloc[folds[0][0]].std()

        l = random.randrange(0, len(train_fold)//random.randrange(5, 20))
        s = random.randrange(0, len(train_fold) - l)
        train_fold = train_fold[:s].tolist() + train_fold[s + l:].tolist()

        scaler = StandardScaler()
        scaler.fit(x_loc.iloc[train_fold])
        x_scaled = pd.DataFrame(scaler.transform(x_loc),
                                 x_loc.index, x_loc.columns)
        xs_scaled = pd.DataFrame(scaler.transform(xs_loc),
                                 xs_loc.index, xs_loc.columns)

        train_dataset = AirDataset(x_scaled, g_loc, y_loc, train_fold)
        val_dataset = AirDataset(x_scaled, g_loc, y_loc, test_fold if len(test_fold) > 0 else train_fold)
        test_dataset = AirDataset(xs_scaled, gs_loc, ys_loc, np.arange(0, len(xs_scaled)))


        train_loader = DataLoader(train_dataset, batch_size = 12, 
                                  shuffle = True, num_workers = os.cpu_count(),
                                  drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = 256, 
                                  shuffle = False, num_workers = os.cpu_count(),
                                  drop_last = False)
        test_loader = DataLoader(test_dataset, batch_size = 256, 
                                  shuffle = False, num_workers = os.cpu_count(),
                                  drop_last = False)


        model = AirModel(input_dropout = 0.5,  grid_dims = 16, gn = gn, 
                             lr = 0.2, dims = dims,
                            num_paths = 4)
        model_path = 'nn1/{}_{}/'.format(
                                    dataset, location.replace(' ', '_'))
        if fold_idx == 0: print(model_path)
        model_str = '{}dims_gn{}_lr{}_wd{}_run{}'.format(
                            dims, gn, learning_rate, weight_decay,
                                seed) + (
                    '_fold{}'.format(fold_idx) if fold_idx < len(folds) - 1
                        else '_full'); print(model_str)
        
        trainer = pl.Trainer(max_epochs = max_epochs, #enable_checkpointing = False, 
                             logger = False,
                                    enable_progress_bar = False,
                             callbacks = [ pl.callbacks.ModelCheckpoint(
                                 every_n_epochs = 5, save_top_k = 10,
                                 monitor = 'val_loss',
                                dirpath = model_path,
            filename = model_str +  "-{epoch:02d}",)])

        trainer.fit(model, train_loader, val_loader)
        savePreds(model_path, model_str)
    # break;


# !jupyter nbconvert --to script 'TrainNN.ipynb' 







