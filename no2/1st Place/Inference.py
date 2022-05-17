#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


from collections import defaultdict


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.max_open_warning'] = 100


import os
import json


import subprocess


import pickle


import requests
import io


import tarfile


import zstandard as zstd


import time
import pygrib
import datetime
import secrets
import random


from joblib import Parallel, delayed


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC


import h5py


import pydap
import xarray as xr
import numpy as np


import cv2


from IPython.display import display
pd.set_option('display.max_columns', 100)





# s3 = boto3.client('s3')
# sqs = boto3.client('sqs')


zd = zstd.ZstdDecompressor()





SEED = datetime.datetime.now().microsecond


random.seed(datetime.datetime.now().microsecond); 
np.random.seed(datetime.datetime.now().microsecond)
run_label = secrets.token_hex(3)





dataset = 'tg'


ASSIM = True








labels = pd.read_csv('data_tg/train_labels.csv')
grid = pd.read_csv("data_tg/grid_metadata.csv")
submission = pd.read_csv('data_tg/verification_no2_submission_format.csv', index_col=0)
files = pd.read_csv('data_tg/verification_no2_satellite_metadata.csv', index_col=0)


labels['location'] = grid.set_index('grid_id')['location'].reindex(labels.grid_id).values
labels['datetime'] = pd.to_datetime(labels.datetime)

submission['location'] = grid.set_index('grid_id').location.reindex(submission.grid_id).values








infer = dict([e.split('=') for e in open('infer.txt', 'r').read().split('\n')])
# infer = {k: v.split(',') for k, v in infer.items()}


infer


start = datetime.datetime(*[int(i) for i in infer['start'].split(',')])
end = datetime.datetime(*[int(i) for i in infer['end'].split(',')])


dt = start
dates = []
while dt <= end:
    dates.append(dt);
    dt += datetime.timedelta(days = 1)
print(len(dates))
print(dates[0]); print(dates[-1])


import pytz


tz_dict = {'Los Angeles (SoCAB)': '08:00:00Z',
 'Delhi': '18:30:00Z', 
 'Taipei': '16:00:00Z'}


srows = []
for t in dates:
    for g in grid.itertuples():
        tutc = t - datetime.timedelta(days = 1 if g.location in ['Taipei', 'Delhi'] else 0)
        srows.append(('{:04}-{:02}-{:02}T'.format(
            tutc.year, tutc.month, tutc.day) + tz_dict[g.location],
                      
        g.grid_id, 0, g.location))


submission = pd.DataFrame(srows, columns = submission.columns)











loc_dict = {'la': 'Los Angeles (SoCAB)', 'tpe': 'Taipei', 'dl': 'Delhi'}

coords = defaultdict(list)
for city, gcity in loc_dict.items():#['Delhi', 'Taipei', 'LA']:
    # gcity = [c for c in grid.location.unique() if 'Los A' in c][0] if city not in grid.location.unique() else city
    grid_points = grid[grid.location == gcity]
    for e in grid_points.itertuples():
        coords[city].append( (e.grid_id, *np.array([(float(p.split(' ')[0]), 
                                                     float(p.split(' ')[1])) 
                                            for p in e.wkt[10:].split(', ')[:4]]).mean(axis = 0).round(4).tolist() ) )

        

def cleanDict(d):
    return {k: cleanDict(v) for k, v in d.items() } if isinstance(d, defaultdict) else d

assert sum([len(v) for c, v in coords.items()]) == grid.grid_id.nunique()
grid.grid_id.nunique()

coords = cleanDict(coords)











import sklearn


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)#
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.FitFailedWarning)








cities = {
 'Taipei': ( (121.5, 121.5), (25.0, 25) ),
 'Delhi': ( (77.0, 77.25), (28.75, 28.5) ),
 'LA': ((360-118.25, 360-117.75), (34.0, 34.0) ) 
}








import cv2





from collections import defaultdict


# %%time
coords = defaultdict(list)

for city in ['Delhi', 'Taipei', 'LA']:
    gcity = [c for c in grid.location.unique() if 'Los A' in c][0] if city not in grid.location.unique() else city
    grid_points = grid[grid.location == gcity]
    for e in grid_points.itertuples():
        coords[city].append( (e.grid_id, *np.array([(float(p.split(' ')[0]), 
                                                     float(p.split(' ')[1])) 
                                            for p in e.wkt[10:].split(', ')[:4]]).mean(axis = 0).tolist() ) )














# ### Load Saved Features

# def loadFiles(file):
#     zd = zstd.ZstdDecompressor()
#     data = []
#     with tarfile.TarFile(file, 'r') as tar:
#         assim_files = tar.getnames()
#         for file in assim_files:
#             if len(file) > 0:
#                 f = tar.extractfile(file)
#                 data.append(pickle.loads(zd.decompress(f.read())))
#     return data


def loadFiles(path):
    path = path.replace('cache', 'inference')[:-4]
    print(path)
    zd = zstd.ZstdDecompressor()
    data = []
    # with tarfile.TarFile(file, 'r') as tar:
        # assim_files = tar.getnames()
    files = sorted(os.listdir(path))
    for file in files:
        data.append(pickle.loads(zd.decompress(
            open(os.path.join(path, file), 'rb').read())))
    return data





def flatten(l): return [e for s in l for e in s]


loc_dict = {'la': 'Los Angeles (SoCAB)', 'tpe': 'Taipei', 'dl': 'Delhi'}

rloc_dict = {v: k for k, v in loc_dict.items()}

tz_dict = grid.groupby('location').tz.first().to_dict()

loc_tz_dict = {rloc_dict[k]: v for k, v in tz_dict.items()}
loc_tz_dict




















# %%time
all_data_assim = loadFiles('cache/assim.tar')


all_data_assim = flatten(all_data_assim)


len(all_data_assim)








rt_dict = {}


def process(d, e):
    t = d['time_end']
    if t not in rt_dict:
        rt = (t + 30).astype('datetime64[m]')
        rt_dict[t] = rt
    rt = rt_dict[t]
    grid_data_assim[grid_id][rt].update(e)
    grid_data_assim[grid_id][rt].update({'location': d['location']})


# %%time
grid_data_assim = defaultdict(lambda: defaultdict(dict))
for d in all_data_assim:
    for grid_id, e in d['d1'].items():
        process(d, e)





def processAssim(grid_id, v):
    df = pd.DataFrame.from_dict(v, orient= 'index')
    
    # tz
    assert df.location.nunique() == 1 
    tz = loc_tz_dict[df.location.unique()[0]]
    df.index = df.index.tz_localize('UTC').tz_convert(tz).floor('1d')
    df.index.name = 'datetime'
    df.drop(columns = 'location', inplace = True)
    df = df.groupby(df.index).mean()#.sort_index()#.head(10)    

    # fill
    t = pd.to_datetime(submission.datetime.min()).tz_convert(tz).floor('1d')
    tf = pd.to_datetime(submission.datetime.max()).tz_convert(tz).floor('1d')
    df_t = set(df.index)
    extra_times = []
    while t <= tf:
        if t not in df_t:
            extra_times.append(t)
        t += datetime.timedelta(days = 1)
    df_extra = pd.DataFrame(np.nan, index = pd.Series(extra_times), columns = df.columns )
    df = pd.concat((df, df_extra)).sort_index()
    
    
    # ewm     
    n = 1
    df_ewm1 = df.ewm(span = n).mean().astype(np.float32).fillna(df.mean())
    df_ewm1.columns = [c + '_{}day'.format(n) for c in df.columns]

    n = 2
    df_ewm2 = df.ewm(span = n).mean().astype(np.float32).fillna(df.mean())
    df_ewm2.columns = [c + '_{}day'.format(n) for c in df.columns]

    n = 5
    df_ewm5 = df.ewm(span = n).mean().astype(np.float32).fillna(df.mean())
    df_ewm5.columns = [c + '_{}day'.format(n) for c in df.columns]
        
    raw_df = df
    df = pd.concat((df_ewm1, df_ewm2, df_ewm5,
                   ), axis = 1)
    
    # compile
    df.index = df.index.tz_localize(None)
    df.insert(0, 'grid_id', grid_id)
    return df;


# %%time
all_dfs = Parallel(os.cpu_count())(delayed(processAssim)(grid_id, v)
                                   for grid_id, v in grid_data_assim.items() )


assim = pd.concat(all_dfs)
assim.index.name = 'datetime'
assim = assim.reset_index().set_index(['datetime', 'grid_id'])


del all_data_assim, grid_data_assim, all_dfs











# %%time
all_data_tropomi = loadFiles('cache/tropomi-fine.tar')


grid_data_tropomi = defaultdict(list)
for d in all_data_tropomi:
    for grid_id, e in d['d1'].items():
        # e = blend(e)#e.copy()
        e['datetime'] = d['time_end']
        e['location'] = d['location']
        grid_data_tropomi[grid_id].append(e)
    # break;





# %%time
all_dfs = []
for grid_id, v in grid_data_tropomi.items():
    df = pd.DataFrame(v)

    # tz
    assert df.location.nunique() == 1 
    tz = loc_tz_dict[df.location.unique()[0]]
    df.datetime = pd.to_datetime(df.datetime).dt.tz_convert(tz).dt.floor('1d')
    df.drop(columns = 'location', inplace = True)

    # group
    for col in [c for c in df.columns if '_mean' in c]:
        ct = np.where(df[col].isnull(), 0, df[col.replace('_mean', '_count')] )
        df[col.replace('_mean', '_sum')] = df[col] * ct
        df[col.replace('_mean', '_count')] = ct
    df = df.groupby(df.datetime).sum()#.sort_index()#.head(10)
    

    for col in [c for c in df.columns if '_mean' in c]:
        df[col] = (df[col.replace('_mean', '_sum')] 
                                / df[col.replace('_mean', '_count')]
                    )
    # filter
    df = df[[c for c in df.columns if
             ('.' in  c) and 
             ( ( '_mean' in c ) or ('column_stdev' in c) )
            ]]
    
    # fill
    t = pd.to_datetime(submission.datetime.min()).tz_convert(tz).floor('1d')
    tf = pd.to_datetime(submission.datetime.max()).tz_convert(tz).floor('1d')
    df_t = set(df.index)
    extra_times = []
    while t <= tf:
        if t not in df_t:
            extra_times.append(t)
        t += datetime.timedelta(days = 1)
    df_extra = pd.DataFrame(np.nan, index = pd.Series(extra_times), columns = df.columns )
    df = pd.concat((df, df_extra)).sort_index()
        
    # clip
    sigma = 4
    high = df.ewm(span = 100).mean() + sigma * df.ewm(span = 100).std().fillna(100000)
    df = df.clip(0, None)
    df = np.minimum(df, high)
    
    # ewm     
    n = 1
    df_ewm1 = df.ewm(span = n).mean().astype(np.float32).fillna(df.mean())
    df_ewm1.columns = [c + '_{}day'.format(n) for c in df.columns]

    n = 3
    df_ewm3 = df.ewm(span = n).mean().astype(np.float32).fillna(df.mean())
    df_ewm3.columns = [c + '_{}day'.format(n) for c in df.columns]
    
    raw_df = df
    df = pd.concat((df_ewm1, df_ewm3, 
                   ), axis = 1)
    
    # compile
    df.index = df.index.tz_localize(None)
    df.insert(0, 'grid_id', grid_id)
    all_dfs.append(df)


tropomi = pd.concat(all_dfs)
tropomi.index.name = 'datetime'
tropomi = tropomi.reset_index().set_index(['datetime', 'grid_id'])


del all_data_tropomi, grid_data_tropomi, all_dfs











# %%time
# downloadFiles(ifs_files)





# def loadData(f):
#     s3 = boto3.client('s3')
#     zd = zstd.ZstdDecompressor()
#     data = pickle.loads(zd.decompress(s3.get_object(Bucket = 'projects-v', Key = f)['Body'].read()))    
#     return data








# ifs_files = sorted(listFiles('aqi/ifs/'));


# ifs_tags = [
#     '128_015_aluvp', '128_134_sp', '128_136_tcw', 
#     # '128_137_tcwv', 
# '128_164_tcc',	   
#  '128_165_10u',
# '128_166_10v',
#  '128_167_2t',
# '128_168_2d',	
# '128_206_tco3',  	
# '228_246_100u',	   
# '228_247_100v',  	   
# ]


# ifs_files = [f for f in ifs_files # if any(z in f for z in ifs_tags) 
#                     if '.oper.fc'  in f
#             ]


# len(ifs_files)#[::1000]








# def loadData(f):
#     s3 = boto3.client('s3')
#     data = s3.get_object(Bucket = 'projects-v', Key = f)['Body'].read() 
#     return data


# %%time
# all_data_ifs = Parallel(os.cpu_count() * 3)(delayed(loadData)(d) for d in 
#                         [f for f in ifs_files[::1] ])
# # data = flatten(all_data)





# %%time
all_data_ifs = loadFiles('cache/ifs.tar')





# with tarfile.open('cache/ifs.tar', "w") as tar:
#     for file, data in zip(ifs_files[:], all_data_ifs[:]):
#         t = tarfile.TarInfo(name = file)
#         t.size = len(data)
#         b = io.BytesIO(data)
#         b.seek(0)
#         tar.addfile(
#             t, b)
#         # print(b)


# %%time
grid_data_ifs = defaultdict(lambda: defaultdict(dict))
for d in all_data_ifs:
    for dt, v in d.items():
        for grid_id, e in v.items():
            # e['timezone'] = grid_tz_dict[grid_id]#d['location']
            grid_data_ifs[grid_id][dt[0]].update(e)


# %%time
all_dfs = []
# def compileIfs(grid_id, v):
for grid_id, v in grid_data_ifs.items():
    df = pd.DataFrame(v).T
    df.index.name = 'datetime'
    # df.sort_index(inplace = True)
    df = df.reset_index()
    df.sort_values('datetime', inplace = True)
    
    # break;
    extra_rows = []
    t = df.datetime.min()
    timestamps = set(df.datetime)
    t_max = df.datetime.max()
    while t < t_max:#max(timestamps):#df.datetime.max():
        t_next = t + datetime.timedelta(seconds = 60 * 60 * 12) 
        if t_next not in timestamps:
            if grid_id == '1X116': print(t_next)
            extra = df[df.datetime == t].copy() #if not last else last.copy()
            extra.datetime = t_next
            extra_rows.append(extra)
            # last = extra
        # else:
            # last = None
        t = t_next;
    if len(extra_rows) > 1 and grid_id == '1X116': 
        print('{} extra rows'.format(len(extra_rows)))
    df = pd.concat((df, *extra_rows))
    
    df.set_index('datetime', inplace = True)    

    # df = addExtraRows(df.reset_index()).set_index('datetime')    
    df = df.ewm(span = 2 * 2).mean().astype(np.float32)
    df.columns = [c + '_{}day'.format(2) for c in df.columns]

        
    # compile
    df.index = df.index.tz_localize(None)
    df.drop(columns = [c for c in df.columns if 'vapour' in c 
                       or 'mean0.05' in c 
                       or 'mean0.2' in c
                       or 'mean1' in c
                       or 'roughness' in c
                       or 'albedo' in c
                      ], inplace = True)
    df.columns = ['ifs_'+ c for c in df.columns]
    df.insert(0, 'grid_id', grid_id)
    # return df;

    all_dfs.append(df)
    # break;
    # break;

# all_dfs = Parallel(#os.cpu_count() * 2
#                   1)(delayed(compileIfs)(grid_id, v) 
#                            for grid_id, v in grid_data_ifs.items())


ifs = pd.concat(all_dfs)
ifs.index.name = 'datetime'
ifs = ifs.reset_index().set_index(['datetime', 'grid_id'])








for i in [0, 10, -1]:
    df = all_dfs[i]
    location = grid.set_index('grid_id').location[df.grid_id.iloc[0]]
    plt.matshow(df.corr()); plt.colorbar(); plt.title(location)


df.columns[1::6]


del all_data_ifs, grid_data_ifs, all_dfs
































# gfs_files = listFiles('aqi/gfs-5/')


# len(gfs_files)


# with tarfile.open('cache/gfs-5.tar', "w") as tar:
#     tar.add('cache/gfs-5/', arcname=os.path.sep)


# %%time
# all_data_gfs = Parallel(os.cpu_count() * 3)(delayed(loadData)(d) for d in 
#                         [f for f in gfs_files if 'f000' in f])
# all_data_gfs = flatten(all_data_gfs)











# %%time
all_data_gfs = loadFiles('cache/gfs-5.tar')
all_data_gfs = flatten(all_data_gfs)


len(all_data_gfs)





# %%time
points = defaultdict(lambda: defaultdict(list))
for e in all_data_gfs:# [e for e in h if e[1] == city]:
    t = datetime.datetime.strptime(e[0][-12:-2], '%Y%m%d%H')
    n = ':'.join(e[0].split(':')[1:2])
    city = e[1]
    
    arr = e[3] 
    arr2 = cv2.resize(arr, None, fx = 5, fy = 5,  )
    
    for grid_id, x, y in coords[city]:
        v = arr2[ int(round(( e[2][0][1] - y ) / 0.05 + 2)), 
            int(round( ( (x % 360) - e[2][1][0] )/0.05 + 2 ) ) ]
        points[grid_id][n].append((t, v, e[-1]))


def addExtraRows(df):
    extra_rows = []
    t = df.datetime.min()
    timestamps = set(df.datetime)
    t_max = df.datetime.max()
    while t < t_max:#max(timestamps):#df.datetime.max():
        t_next = t + datetime.timedelta(seconds = 60 * 60 * 6) 
        if t_next not in timestamps:
            extra = df[df.datetime == t].copy() if not last else last.copy()
            extra.datetime = t_next
            extra_rows.append(extra)
            last = extra
        else:
            last = None
        t = t_next;
    if len(extra_rows) > 1: print('{} extra rows'.format(len(extra_rows)))
    return pd.concat((df, *extra_rows))








all_dfs = []
for grid_point, grid_data in points.items():
    dfs = []
    for label, d in grid_data.items():
        if 'Volumetric soil moisture content' in label: continue;
        df = pd.DataFrame(d, 
                     columns = ['datetime', label + '_local', label + '_city'])
        df.sort_values('datetime', inplace = True)
        dfs.append(df.set_index('datetime'))
                
    df = pd.concat(dfs, axis = 1)
    df = addExtraRows(df.reset_index()).set_index('datetime')    
    df = df.ewm(span = 4 * 3).mean().astype(np.float32)
    df.columns = [c + '_{}day'.format(3) for c in df.columns]
    df['grid_id'] = grid_point
    all_dfs.append(df)
gfs = pd.concat(all_dfs)
gfs = gfs.reset_index().set_index(['datetime', 'grid_id'])
#
        # print(df.corr().iloc[0, 1], label)
        # break;


del all_data_gfs, points, all_dfs











# ### Compile

labels['dayofweek'] = labels.datetime.dt.dayofweek.astype(np.int32)


labels['dayinyear'] = labels.datetime.dt.dayofyear.astype(np.int32)


labels.head()








def addGFS(labels):
    ldo = pd.to_datetime(labels.datetime).dt.floor('6h').dt.tz_localize(None)
    df = pd.concat( ( labels, 
            * [2/3 * gfs.reindex([ldo
                                + datetime.timedelta(seconds = 60 * 60 * 12), 
                             labels.grid_id]).reset_index(drop = True) 
               + 1/3 * gfs.reindex([ldo
                                + datetime.timedelta(seconds = 60 * 60 * 18), 
                             labels.grid_id]).reset_index(drop = True) 
                  #for k, v in gfs_ewms.items() 
              ],                       
                      ), axis = 1)
    return df





# gfs.groupby(['datetime', 'grid_id']).nunique().mean(axis = 1).sort_values()[-50:]








def ifsoffset12(c): return c.replace('ifs', 'ifs12')
def ifsoffset0(c): return c.replace('ifs', 'ifs0')


def addIFS(labels):
    ldo = pd.to_datetime(labels.datetime).dt.floor('12h').dt.tz_localize(None)
    df = pd.concat( ( labels, 
            * [ifs.reindex([ldo
                                + datetime.timedelta(seconds = 60 * 60 * 12), 
                             labels.grid_id]).reset_index(drop = True).rename(
                                   columns = ifsoffset12)

                , 
               ifs.reindex([ldo
                                # + datetime.timedelta(seconds = 60 * 60 * 0)
                                    , 
                             labels.grid_id]).reset_index(drop = True).rename(
                   columns = ifsoffset0)
                  #for k, v in gfs_ewms.items() 
              ],                       
                      ), axis = 1)
    return df





def addSat(labels, sat):
    
    tza = labels.datetime.copy()
    for location in labels.location.unique():
        t = pd.to_datetime(labels.datetime).dt.tz_convert( tz_dict[location] )
        t = t.dt.floor('1d').dt.tz_localize(None)
        tza = np.where(labels.location == location, t, tza)
    tza = pd.to_datetime(pd.Series(tza, index = labels.index))

    df = pd.concat( ( labels, 
            sat.reindex([tza,
                             labels.grid_id]).reset_index(drop = True) 
                     
                      ), axis = 1)
    return df














all_data = addGFS(labels)
all_data = addIFS(all_data)
all_data = addSat(all_data, maiac if dataset == 'pm' else tropomi)
if ASSIM: all_data = addSat(all_data, assim)
# all_data = addSat(all_data, misr)


# all_data.tail()


# all_data.head()





x = all_data[[c for c in all_data.columns if c not in ['datetime', 'value']]].copy()


# '7334C' - > '7F1D1'
# 'HANW9' - > 'WZNCR'
if dataset == 'tg':
    np.random.seed(SEED)
    x.loc[(x.grid_id == '7334C') & (np.random.random(len(x)) < 0.1), 'grid_id'] = '7F1D1'
    x.loc[(x.grid_id == 'HANW9') & (np.random.random(len(x)) < 0.1), 'grid_id'] = 'WZNCR'


x.grid_id = x.grid_id.astype('category');
x.location = x.location.astype('category');

y = all_data.value.astype(np.float32)
d = all_data.datetime


# pickle.dump(all_data, open('cache/all_data_{}.pkl'.format(dataset), 'wb'))








x.columns = [c.replace(':', '_') for c in x.columns]


# x.shape





# x.fillna(method = 'ffill')


# x.isnull().sum()


# x.isnull().sum().sort_values()


# assert x.isnull().sum().sum() == 0





import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
import sklearn


from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


# enet_params = {
#     'alpha': [  1e-2, 3e-2, 1e-1, 0.3, 1, 3, ],
#     'l1_ratio': [ 0.01, 0.03, 0.1, 0.2, 0.5, 0.8, 0.9, ],
# }


# class PurgedKFold():
#     def __init__(self, n_splits=5, gap = 30):
#         self.n_splits = n_splits
#         self.gap = gap
        
#     def get_n_splits(self, X, y = None, groups = None): return self.n_splits
    
#     def split(self, X, y=None, groups=None):
#         groups = groups.sort_values()
#         X = X.reindex(groups.index)# sort_values(groups)
#         y = y.reindex(X.index);
                     
#         X, y, groups = sklearn.utils.indexable(X, y, groups)
#         indices = np.arange(len(X))
        
#         n_splits = self.n_splits
#         for i in range(n_splits):
#             test = indices[ i * len(X) // n_splits: (i + 1) * len(X) // n_splits ]#.index
#             train = indices[ (groups <= groups.iloc[test].min() - datetime.timedelta(days = self.gap) )
#                           | (groups >= groups.iloc[test].max() + datetime.timedelta(days = self.gap) ) ]#.index
#             yield train, test

# class RepeatedPurgedKFold():
#     def __init__(self, n_splits = 5, n_repeats = 1, gap = None):
#         self.n_splits = n_splits
#         self.n_repeats = n_repeats
#         self.gap = gap
        
#     def get_n_splits(self, X, y = None, groups = None): 
#         return self.n_splits * self.n_repeats + self.n_repeats * ( self.n_repeats - 1) // 2
    
#     def split(self, X, y=None, groups=None):
#         for i in range(self.n_repeats):
#             for f in PurgedKFold(self.n_splits + i, gap = self.gap if self.gap else None).split(X, y, groups):
#                 yield f
    





from sklearn.metrics import make_scorer, mean_squared_error


# def posRMSE(y, y_pred):
#     return mean_squared_error(y, y_pred.clip(0, None) * 2/3 + y_pred * 1/3) ** 0.5
# pRMSE = make_scorer(posRMSE, greater_is_better = False)
# SCORING = pRMSE





# COLUMN_WIPE = 0# len(x.columns) // 10











# def runENet(drop_cols = [], verbose = 1):
#     all_y_enet = []; all_y_pred_enet = []; enet_clfs = []; enet_scalers = []
#     for location in x.location.unique():
#         if verbose > 0: print(location)
#         x_loc = x[x.location == location].drop(columns = drop_cols)
#         y_loc = y.reindex(x_loc.index)
#         d_loc = d.reindex(x_loc.index)
    

#         folds = list(PurgedKFold(4 if dataset == 'pm' else 3).split(x_loc, y_loc, d_loc))
#         folds += [(np.arange(0, len(x_loc)), [])] * 2
#         for train_fold, test_fold in folds:
#             y_preds = []
#             for i in range(8):
#                 scaler = StandardScaler()
#                 clf = ElasticNet(max_iter = 50000, tol = 1e-3, 
#                                  selection = 'random', precompute = True,
#                                 random_state = datetime.datetime.now().microsecond)                     
#                 model = RandomizedSearchCV(clf, enet_params, 
#                                            scoring = SCORING,
#                                            # 'neg_root_mean_squared_error',
#                                            cv = RepeatedPurgedKFold(random.randrange(3, 6), 
#                                                 n_repeats = random.randrange(2, 4), 
#                                                     gap = random.randrange(60, 120)), 
#                                                    n_iter = random.randrange(4, 7),
#                                               random_state = datetime.datetime.now().microsecond)
#                 # subset = train_fold#[:s].tolist() + train_fold[s + l:].tolist()
#                 l = random.randrange(0, len(train_fold)//10)
#                 s = random.randrange(0, len(train_fold) - l)
#                 subset = train_fold[:s].tolist() + train_fold[s + l:].tolist()
#                 xt = x_loc.iloc[subset, 2:].copy()
#                 # for c in misr.columns:
                
#                 if dataset == 'pm':
#                     sample_cols = [c for c in xt.columns if any(z in c for z in 
#                                     ['precision', 'air_mass', 'stdev']
#                                     + ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc', 
#                                        'ifs']
#                     )]
#                     xt[random.sample(sample_cols, k = random.randrange( len(sample_cols) * 1 // 3,
#                                                                         len(sample_cols) * 2 // 3) )] = 0

                
#                 if dataset == 'tg':
#                     if random.random() < 0.5: xt[[c for c in xt.columns if 'precision' in c]] = 0
#                     # if random.random() < 0.5: xt[[c for c in xt.columns if 'mean1' in c or 'mean2' in c]] = 0
#                 # xt[misr.columns] = 0
#                 # fj_drop = 
#                 if random.random() < {'Delhi': 0.5, 'Taipei': 0.2}.get(location, 0):
#                     xt[[c for c in xt.columns if 'FineMode' in c or 'Injection' in c]] = 0
#                 # if random.random() < {'Delhi': 0.2, 'Taipei': 0.1}.get(location, 0):
#                 #     xt[[c for c in xt.columns if 'Optical' in c]] = 0
#                 if random.random() < {'Delhi': 0.85, 'Taipei': 0.5}.get(location, 0.2):
#                     xt[[c for c in xt.columns if c.startswith('ifs')]] = 0

#                 # xt = x_loc.iloc[subset].copy()
#                 xt[random.choices(xt.columns[2:],
#                         k = int(round(random.random() * COLUMN_WIPE)))] = 0
#                 if random.random() < 0.3: xt['dayinyear'] = 0;

#                 model.fit( pd.DataFrame(scaler.fit_transform(xt).astype(np.float32), 
#                                         xt.index, xt.columns)
#                                  , y_loc.iloc[subset], groups = d_loc.iloc[subset])
#                 enet_clfs.append(model.best_estimator_)
#                 enet_scalers.append(scaler)

                
#                 if i == 0 and verbose > 0: display(pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(
#                         columns = 'params'))

#                 if len(test_fold) > 0:
#                     y_pred =  pd.Series( model.predict(
#                                 pd.DataFrame(scaler.transform(x_loc.iloc[test_fold, 2:]).astype(np.float32),
#                                                                        columns = x_loc.columns[2:])),# .clip(0, None), 
#                                                                             index = y_loc.iloc[test_fold].index)
#                     y_preds.append(y_pred)

#             if len(test_fold) > 0:
#                 y_pred = pd.concat(y_preds)
#                 y_pred = y_pred.clip(0, None).groupby(y_pred.index).mean().clip(0, None)
#                 y_pred = ( y_pred.groupby(y_pred.index).mean() )#*2/3
#                              # + y_pred.groupby(y_pred.index).median() * 1/3 )

#                 all_y_enet.append(y_loc.iloc[test_fold])
#                 all_y_pred_enet.append(y_pred)

#                 if verbose >= 1: print(location, round(np.corrcoef(y_pred, all_y_enet[-1])[0, 1],4 ) )

#                 if verbose >= 3:
#                     all_y_enet[-1].reset_index(drop = True).ewm(span = 10).mean().plot()
#                     plt.plot(y_pred.reset_index(drop = True))

#                     plt.title(location); plt.figure()

#                     print((x_loc.iloc[subset, 2:].std().clip(1, 1) *
#                          model.best_estimator_.coef_ ).sort_values())
                
#     return  all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers


# all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers = runENet(verbose = 2)





# os.makedirs('clfs_{}'.format(dataset), exist_ok = True)


# pickle.dump( (all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers), 
#                      open('clfs_{}/enet_clfs_{}.pkl'.format(
#                                 dataset, run_label), 'wb'))


# for i in range(3):
#     pd.Series(np.mean([e.coef_ for e in 
#                    enet_clfs[len(enet_clfs) * i//3:len(enet_clfs) * (i + 1)//3]],
#         axis = 0), index = x.columns[2:]).sort_values().plot(kind = 'barh',
#                                                                 figsize = (10, x.shape[1]// 4),
#                                                                 title = list(x.location.unique())[i], )
#     plt.figure()











# print(round(np.corrcoef( pd.concat(all_y_pred_enet).clip(0, None),#.reindex(df.index), 
#                             pd.concat(all_y_enet)#.reindex(df.index)
#            )[0, 1], 3))


# cs = []
# for location, df in x.groupby('location'):
#     c = np.corrcoef( pd.concat(all_y_pred_enet).clip(0, None).reindex(df.index), 
#                 pd.concat(all_y_enet).reindex(df.index))[0, 1]; cs.append(c)
#     print(location, round(c, 3) )
# print('\nBlend:', round(np.mean(cs), 3))








# lgb_params = {
#     'n_estimators': np.arange(200, 400, 10) if dataset == 'pm' else np.arange(300, 600, 20),#[ 150, 200, 300, ],
#     'learning_rate': np.arange(0.01, 0.04, 0.003) if dataset == 'pm' else np.arange(0.01, 0.061, 0.005),# [0.03, 0.05, 0.07, ],
#     'num_leaves': np.arange(4, 30) if dataset == 'pm' else np.arange(10, 30),# [5, 7, 10, 15, 20,],
#     'min_child_weight': np.arange(0.02, 0.1, 0.01),#[  0.1, 0.2, ],
#     'min_child_samples': [ 140, 170, 200, 250, 300, 400, 500, 600, 700, 850, 1000, 1400, ]
#                                     if dataset == 'pm' else
#                          [ 30, 40, 50, 60, 80, 100, 120, 150, 170, 200, 300, 500, 700, ]  
#     , 
#     'reg_lambda': [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1,  ],
#     'reg_alpha':  [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, ],
#     'linear_tree': [True, ],
#     'subsample': np.arange(0.4, 0.901, 0.05),#[0.3, 0.5,  0.8],
#     'subsample_freq': [1],
#     'colsample_bytree': np.arange(0.3, 0.71, 0.05) if dataset == 'pm' else np.arange(0.2, 0.81, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
#     'colsample_bynode': np.arange(0.4, 1.01, 0.05)  if dataset == 'pm' else np.arange(0.2, 1.01, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
#     'linear_lambda': [1e-3, 3e-3, 1e-2, 3e-2, 0.1,],
#     # 'max_bins': np.arange(120, 400, 20),
#     # 'min_data_in_bin': np.exp(np.arange(np.log(3), np.log(12), 0.1)).astype(int), #np.arange(2, 10),# [2, 3, 4, 5, 10],
#                 # [ 192, 
#                  #255, 255, 384, 512],
#     'min_data_per_group': [10, 20, 50, 100],
#     'max_cat_threshold': [  8, 16, 32, ], 
    
#     'cat_l2': [0.1, 1, 10],#     if dataset == 'pm' else [1e-3, 1e-2, 1e-1],
#     'cat_smooth': [0.1, 1, 10],#      if dataset == 'pm' else [1e-3, 1e-2, 1e-1],
# }


# FAST = False
# GAUSSIAN = 0.05 #if dataset == 'pm' else 0.1
# COLUMN_WIPE = len(x.columns) // 5 #  if dataset == 'pm' else 10)





# def runLGB(drop_cols = [], verbose = 1, n_bags = 8, ):
#     all_y_lgb = []; all_y_pred_lgb = []; tidx = 0;     lgb_clfs = []; lgb_scalers = [];
#     for location in x.location.unique():
#         random.seed(datetime.datetime.now().microsecond);
#         np.random.seed(datetime.datetime.now().microsecond)
#         if FAST and location != 'Delhi': continue;
#         if verbose > 0: print(location)
#         x_loc = x[x.location == location].drop(columns = drop_cols)#.iloc[:, :4]
#         y_loc = y.reindex(x_loc.index)
#         d_loc = d.reindex(x_loc.index)

#         folds = list((PurgedKFold(4) if dataset == 'pm' else PurgedKFold(3, gap = 20))
#                          .split(x_loc, y_loc, d_loc)) 
#         folds += [(np.arange(0, len(x_loc)), [])] * 3
#         for train_fold, test_fold in folds:
#             y_preds = []
#             for i in range(n_bags):# if location == 'Delhi' else 4):
#                 model = RandomizedSearchCV(lgb.LGBMRegressor(seed = datetime.datetime.now().microsecond,
#                                                                  # n_jobs = 2, # os.cpu_count()
#                                                             ), lgb_params,
#                                                cv = RepeatedPurgedKFold( random.randrange(3, 6),
#                                                                          n_repeats = random.randrange(1, 3),
#                                                                          gap = random.randrange(60, 120)), 
#                                                n_iter = random.randrange(3, 5),
#                                            scoring = SCORING,
#                                            n_jobs = -1, #os.cpu_count(),
#                                            random_state = datetime.datetime.now().microsecond)
#                 l = random.randrange(0, len(train_fold)//10)
#                 s = random.randrange(0, len(train_fold) - l)
#                 subset = train_fold[:s].tolist() + train_fold[s + l:].tolist()

#                 xt = x_loc.iloc[subset].copy()
#                 xt.iloc[:, 2:] += ( GAUSSIAN * random.random()  * 
#                                        np.random.default_rng().standard_normal( size = xt.iloc[:, 2:].shape)
#                                                     * xt.iloc[:, 2:].std().values[None, :] )
#                 for c in random.choices(xt.columns[2:], k = int(round(random.random() * COLUMN_WIPE))):
#                     xt[c] = 0
#                 if dataset == 'pm' and random.random() < 0.3: xt['dayinyear'] = 0;
                
#                 sample_cols = [c for c in xt.columns if any(z  in c for z in 
#                                 ['precision', 'air_mass', 'stdev']
#                                 + ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc', 'ifs' ]
#                 )]
#                 xt[random.sample(sample_cols, k = random.randrange( len(sample_cols) * 1 // 3,
#                                                                     len(sample_cols) * 2 // 3) )] = 0
                
#                 if random.random() < {'Delhi': 0.7, 'Taipei': 0.2}.get(location, 0):
#                     xt[[c for c in xt.columns if 'FineMode' in c or 'Injection' in c]] = 0
#                 if random.random() < {'Delhi': 0.1, 'Taipei': 0.1}.get(location, 0):
#                     xt[[c for c in xt.columns if 'Optical' in c]] = 0
#                 if random.random() < {'Delhi': 1.0, 'Taipei': 0.2}.get(location, 0.2):
#                     xt[[c for c in xt.columns if c.startswith('ifs')]] = 0
  
#                 if random.random() < {'Delhi': 0.9,}.get(location, 0.12):
#                     xt[[c for c in xt.columns if any(z + '_' in c for z in 
#                          ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc',  ] ) ]] = 0

#                 scaler = StandardScaler()
#                 xt.iloc[:, 2:] = scaler.fit_transform(xt.iloc[:, 2:]).astype(np.float32)

#                 model.fit( xt, y_loc.iloc[subset], groups = d_loc.iloc[subset],
#                          )

#                 lgb_clfs.append(model.best_estimator_)
#                 lgb_scalers.append(scaler)

#                 if len(test_fold) > 0:
#                     xtst =  x_loc.iloc[test_fold].copy();
#                     xtst.iloc[:, 2:] = scaler.transform(xtst.iloc[:, 2:]).astype(np.float32)

#                     y_pred =  pd.Series( model.predict(xtst ), index = y_loc.iloc[test_fold].index)

#                     y_preds.append( y_pred)

#                 df = pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(columns = 'params')
#                 if i == 0 and verbose > 1: # df.mean_test_score.min() < -0: 
#                     display(df); print()

#             if len(test_fold) > 0: 
#                 y_pred = pd.concat(y_preds)
#                 y_pred = ( y_pred.groupby(y_pred.index).mean() )#* 2/3
#                              # + y_pred.groupby(y_pred.index).median() * 1/3 )

#                 all_y_lgb.append(y_loc.iloc[test_fold])
#                 all_y_pred_lgb.append(y_pred)

#                 if verbose >= 3:
#                     all_y_lgb[-1].reset_index(drop = True).ewm(span = 10).mean().plot()
#                     plt.plot(y_pred.reset_index(drop = True), linewidth = 0.8)
#                     try: plt.plot(all_y_pred_enet[tidx].clip(0, None).reset_index(drop = True))
#                     except: pass

#                     plt.title(location); plt.figure()
#                 tidx += 1

#                 if verbose > 0: print(location, round(np.corrcoef(y_pred, all_y_lgb[-1])[0, 1],4 ) )

#     return  all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers


# all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers = runLGB(verbose = 2, )


# pickle.dump( (all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers), 
#                      open('clfs_{}/lgb_clfs_{}.pkl'.format(
#                                 dataset, run_label), 'wb'))





# # print(round(np.corrcoef( pd.concat(all_y_pred_enet).clip(0, None), pd.concat(all_y_lgb))[0, 1], 3)  )
# print(round(np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None), pd.concat(all_y_lgb))[0, 1], 3) )
# print(round(np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None)
#                   +  0.1  * pd.concat(all_y_pred_enet).clip(0, None)
#                   , pd.concat(all_y_lgb))[0, 1], 4) )
#  # without weather ~0.79 lgb


# cs = []
# for location, df in x.groupby('location'):
#     c = np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None).reindex(df.index) 
#                     + 0.1 * pd.concat(all_y_pred_enet).clip(0, None).reindex(df.index) 
#                 , pd.concat(all_y_lgb).reindex(df.index))[0, 1]; cs.append(c)
#     print(location, 
#           round(c, 3) )
# print('\nBlend: ', round(np.mean(cs), 3))











# submission = pd.read_csv('data_{}/submission_format.csv'.format(dataset))


submission['dayinyear'] = pd.to_datetime(submission.datetime).dt.dayofyear
submission['dayofweek'] = pd.to_datetime(submission.datetime).dt.dayofweek


submission['location'] = grid.set_index('grid_id').location.reindex(submission.grid_id).values
submission['location'] = submission.location.astype('category')
submission['grid_id'] = submission.grid_id.astype(x.grid_id.dtype)


submission = addGFS(submission)
submission = addIFS(submission)
submission = addSat(submission, maiac if dataset == 'pm' else tropomi)
if ASSIM: submission = addSat(submission, assim)





# pickle.dump(submission, open('cache/submission_{}.pkl'.format(dataset), 'wb'))


# submission.corr()





# xs





xs = submission[x.columns]


submission.location.value_counts()





clf_path = 'clfs_{}'.format(dataset)
all_clfs = os.listdir(clf_path)


enet_tuples = [pickle.load(open( os.path.join(clf_path, f), 'rb'))
                   for f in all_clfs if 'enet_clfs' in f]
lgb_tuples = [pickle.load(open( os.path.join(clf_path, f), 'rb'))
                   for f in all_clfs if 'lgb_clfs' in f]


# enet_tuples = [(all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers)]
# lgb_tuples = [(all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers)]


len(lgb_tuples)


len(enet_tuples)


# labels


y_true = pd.concat(flatten([e[0] for e in lgb_tuples]))
y_true = y_true.groupby(y_true.index).mean()


y_pred_lgb = pd.concat(flatten([e[1] for e in lgb_tuples]))
y_pred_lgb =  y_pred_lgb.groupby(y_pred_lgb.index).mean() #* 2/3
                  # + y_pred_lgb.groupby(y_pred_lgb.index).median() * 1/3)





# if dataset == 'pm':
#     y_pred_enet = pd.concat(flatten([e[1] for e in enet_tuples]))
#     y_pred_enet = ( y_pred_enet.groupby(y_pred_enet.index).mean() )
#                       # + y_pred_enet.groupby(y_pred_enet.index).mean() * 1/3)


# cs = []
# for location, df in x.groupby('location'):
#     c = np.corrcoef( y_pred_lgb.clip(0, None).reindex(df.index) 
#                     + 1/5 * ( y_pred_enet.clip(0, None).reindex(df.index) if dataset == 'pm' else 0), 
#                 y_true.reindex(df.index))[0, 1]; cs.append(c)
#     print(location, 
#           round(c, 3) )
# print('\nBlend: ', round(np.mean(cs), 3))








# [c.n_features_in_ for c in enet_scalers]





lgb_clfs, lgb_scalers, enet_clfs, enet_scalers = [], [], [], []
for i in range(3):
    lgb_clfs.extend(flatten([e[2][i * len(e[2]) //3 : (i + 1) * len(e[2]) // 3]
                                    for e in lgb_tuples]))
    lgb_scalers.extend(flatten([e[3][i * len(e[3]) //3 : (i + 1) * len(e[3]) // 3]
                                    for e in lgb_tuples]))
    enet_clfs.extend(flatten([e[2][i * len(e[2]) //3 : (i + 1) * len(e[2]) // 3]
                                    for e in enet_tuples]))
    enet_scalers.extend(flatten([e[3][i * len(e[3]) //3 : (i + 1) * len(e[3]) // 3]
                                    for e in enet_tuples]))








# lgb_scalers[0].mean_





len(lgb_clfs)


len(lgb_scalers)


# xs


# %%time
lgb_ys = []
for clf_idx, clf in enumerate(lgb_clfs):
    x_loc = xs[xs.location == x.location.unique()[clf_idx * 3 // len(lgb_clfs)]].copy()
    x_loc.iloc[:, 2:] = lgb_scalers[clf_idx].transform(x_loc.iloc[:, 2:])
    lgb_ys.append(pd.Series(clf.predict(x_loc), index = x_loc.index))
    # break;
lgb_ys = pd.concat(lgb_ys)








# %%time
# if dataset == 'pm':
enet_ys = []
for clf_idx, clf in enumerate(enet_clfs):
    x_loc = xs[xs.location == x.location.unique()[clf_idx * 3 // len(enet_clfs)]].iloc[:, 2:]
    enet_ys.append(pd.Series(clf.predict(
        pd.DataFrame(enet_scalers[clf_idx].transform(x_loc), 
                     columns = x_loc.columns)), index = x_loc.index))
enet_ys = pd.concat(enet_ys)


# lgb_ys.groupby(lgb_ys.index).std().plot()


lgb_y = lgb_ys.groupby(lgb_ys.index).mean().clip(0, None)#.sort_values()


# if dataset == 'pm': 
enet_y = enet_ys.groupby(enet_ys.index).mean().clip(0, None)#.sort_values()


# # if dataset == 'pm':
# plt.scatter(enet_y, lgb_y, s= 0.1)
# print(round(np.corrcoef(enet_y, lgb_y)[0,1], 4) )#, s= 0.1)


f = 8 if dataset == 'pm' else 100000#lgb_y
ys = (lgb_y * (f - 1)/f + enet_y * 1/f)


ys.name = 'value'


out = pd.concat((submission[['datetime', 'grid_id']], ys.reindex(submission.index)), axis = 1)
out


# match = out.merge(pd.read_csv('submissions_tg/new.csv'), 
#               on = ['datetime', 'grid_id'],
#           suffixes = ('_new', '')
#          )
# match

# np.corrcoef(match.value, match.value_new)











os.makedirs('inference_{}'.format(dataset), exist_ok = True)
out.to_csv('inference_{}/new.csv'.format(dataset), index = False)








# ### NN

# import random, datetime


# random.seed(datetime.datetime.now().microsecond)


# dataset = 'tg'


# max_epochs = 20





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





# all_data = pickle.load(open('cache/all_data_{}.pkl'.format(dataset), 'rb'))


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








all_preds = {}
for location in submission.location.unique():
    start = time.time()
    all_preds[location] = {}
    
    model_path = 'nn1/{}_{}/'.format(dataset, location.replace(' ', '_'))
    files = sorted([f for f in os.listdir(model_path) if 'ckpt' in f
                       and '9.ckpt' in f and 'full' in f
                   ])
    print(len(files), 'models for {}-{}'.format(dataset, location))
    
    models = defaultdict(list)
    for f in files:
        k = '_'.join(f.split('_')[:4]) + f.split('_')[-1] 
        if len(models[k]) < 4: models[k].append(f)

    files = flatten(list(models.values()))
    print(len(files), 'models for {}-{}'.format(dataset, location))
    

    x_loc = x[x.location == location].drop(columns = ['location', 'grid_id'])
    # y_loc = y.reindex(x_loc.index)
    # d_loc = d.reindex(x_loc.index)
    # g_loc = x.reindex(x_loc.index).grid_id.map(grid_dict).astype(np.int32)

    xs_loc = xs[xs.location == location].drop(columns = ['location', 'grid_id'])
    gs_loc = xs.reindex(xs_loc.index).grid_id.map(grid_dict).astype(np.int32)
    ys_loc = pd.Series(np.nan, index = xs_loc.index)
    
    scaler = StandardScaler()
    scaler.fit(x_loc)

    # x_scaled = pd.DataFrame(scaler.transform(x_loc),
    #                          x_loc.index, x_loc.columns)
    xs_scaled = pd.DataFrame(scaler.transform(xs_loc),
                             xs_loc.index, xs_loc.columns)

    test_dataset = AirDataset(xs_scaled, gs_loc, ys_loc, np.arange(0, len(xs_scaled)))

    test_loader = DataLoader(test_dataset, batch_size = 256, 
                              shuffle = False, num_workers = os.cpu_count(),
                              drop_last = False)


    for model_file in files:
        model = AirModel.load_from_checkpoint(
                    os.path.join(model_path, model_file))
        model.eval();

        val_preds = []; val_y = []; test_preds = []
        with torch.no_grad():
            for g, xb, yb in test_loader:
                test_preds.append(model(g, xb).numpy())

        test_preds = np.concatenate(test_preds)
        test_preds = pd.Series(test_preds, test_dataset.x.index)

        all_preds[location][model_file.replace('.ckpt', '-test.pkl.zstd')] = test_preds

    print('{:.1f}s elapsed for {}'.format(time.time() - start, location))





full_test_preds = {}
for location, ap in all_preds.items():
    all_test_preds = defaultdict(list)
    for file, preds in ap.items():
        m = file.split('_run')[0]
        e = file.split('epoch=')[-1].split('-')[0]
        mstr = m + '_epoch{}'.format(e)

        vt = file.split('.pkl')[0].split('-')[-1]
        all_test_preds[mstr].append(preds)

    test_preds = {}
    for k, v in all_test_preds.items():
        v = pd.concat(v)
        v = v.groupby(v.index).mean()
        test_preds[k] = v
        
    print(len(test_preds))
    df = pd.DataFrame(test_preds)
    print(df.shape)
    
    full_test_preds[location] = df





for location in ['Delhi', 'Los Angeles (SoCAB)', 'Taipei']:    
    test_preds = full_test_preds[location]
    
    lgb_test_preds = pd.read_csv('inference_{}/new.csv'.format(dataset))
    submission = lgb_test_preds
    lgb_test_preds = lgb_test_preds.value.reindex(test_preds.index)
    
    for i in range(10):
        test_preds['lgb{}'.format(i)] = lgb_test_preds

    all_coefs = pickle.load( 
                open('stack1/{}_{}.pkl'.format(dataset, location), 'rb'))
    enet_test_preds = (test_preds * all_coefs).sum(axis = 1)
    
    submission = submission.reindex(enet_test_preds.index)
    submission.value = enet_test_preds

    os.makedirs('inference_{}/nn1'.format(dataset), exist_ok = True)
    submission.to_csv('inference_{}/nn1/{}.csv'.format(
                            dataset, location.replace(' ', '_')))





combined = pd.concat([pd.read_csv('inference_{}/nn1/'.format(
                        dataset) + file, index_col = 0)
            for file in os.listdir('inference_{}/nn1/'.format(
                        dataset,))]).sort_index()

combined.to_csv('inference_{}/stack1.csv'.format(dataset), index = False)


combined





# match = combined.merge(pd.read_csv('../submissions_tg/stack1.csv'), 
#               on = ['datetime', 'grid_id'],
#           suffixes = ('_new', '')
#          )
# match

# np.corrcoef(match.value, match.value_new)





# !jupyter nbconvert --no-prompt --to script 'Inference.ipynb' 







