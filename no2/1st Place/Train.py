#!/usr/bin/env python
# coding: utf-8

# !pip install -r requirements.txt








# !pip install pipreqs


# print(os.listdir('data_tg'))





# print(os.listdir('cache'))





# !pipreqs --force 





# !jupyter nbconvert --to script 'Train.ipynb' 








# !pip install netCDF4
# !pip install h5py
# !pip install pyhdf
# !pip install basemap


# !pip install pydap


# !pip install xarray
# !pip install pygrib


# !pip install opencv-python


# !pip install zstandard





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








labels = pd.read_csv('data_{}/train_labels.csv'.format(dataset))
grid = pd.concat( (
        pd.read_csv('data_tg/grid_metadata.csv'),
         # pd.read_csv('data_pm/grid_metadata.csv') 
) ).drop_duplicates().reset_index(drop = True) 

submission = pd.read_csv('data_{}/submission_format.csv'.format(dataset))
files = pd.read_csv('data_{}/{}_satellite_metadata{}.csv'.format(
                    dataset, *(('pm25', '') if dataset == 'pm' 
                                else ('no2', '_0AF3h09'))))

labels['location'] = grid.set_index('grid_id')['location'].reindex(labels.grid_id).values
labels['datetime'] = pd.to_datetime(labels.datetime)

submission['location'] = grid.set_index('grid_id').location.reindex(submission.grid_id).values


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








# t = datetime.datetime(2016, 1, 1)
# # t = datetime.datetime(2020, 5, 26)
# while t < datetime.datetime.now() - datetime.timedelta(days = 1):
#     print(t)
#     for tag in ifs_tags:
#     # for tag in ifs_forecasts:# ifs_tags:#[:1]:
#         domain = 'ec.oper.an.sfc'
#         # domain = 'ec.oper.fc.sfc'
#         file =  '{}/{}/{}.{}.regn1280sc.{}.nc'.format(domain,
#                         datetime.datetime.strftime(t, '%Y%m'), 
#                         domain, tag, 
#                         datetime.datetime.strftime(t, '%Y%m%d') )
#         if file.split('/')[-1] not in existing_files:
#             sqs.send_message(
#                 QueueUrl='https://sqs.us-east-2.amazonaws.com/815054066888/aqi-ifs',
#                 MessageBody=json.dumps({'file': file}
#             ) )
#         # else:
#         #     print(file, 'exists')
#     t += datetime.timedelta(days = 1)
#     # break;





fields = ['nitrogendioxide_tropospheric_column',
 'nitrogendioxide_tropospheric_column_precision',
 'air_mass_factor_troposphere',
 'air_mass_factor_total']





def loadFile(row):
    
    my_config = Config(signature_version = botocore.UNSIGNED)
    s3a = boto3.client('s3', config = my_config)

    
    filename, url, cksum, sz = [row[k] for k in ['granule_id', 'us_url', 'cksum', 'granule_size']]
    print(filename, url, cksum, sz) 
    file = '/data/' + filename
    
    bucket = url.split('//')[-1].split('/')[0]
    key = '/'.join(url.split('//')[-1].split('/')[1:])
    
    s = s3a.download_file(bucket, key, file)
 
    assert ( subprocess.check_output(['cksum',file])
            .decode('utf-8').split(' ')[:2] == [str(cksum), str(sz)])
    return file
 


import sklearn


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)#
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.FitFailedWarning)





fields = ['nitrogendioxide_tropospheric_column',
 'nitrogendioxide_tropospheric_column_precision',
 'air_mass_factor_troposphere',
 'air_mass_factor_total']
 


def run(hdf, location, fine = True):
    zones = {}; # defaultdict(dict)
    sat_data = defaultdict(lambda: defaultdict(dict))
    hp = hdf['PRODUCT']
    lat = hp['latitude'][:][0]#.values
    lon = hp['longitude'][:][0]#.values
    
    for field in fields:
        v = hp[field][:][0]
        data = np.ma.masked_array(v, (v == v.max() ) | (v == v.min())).clip(0, None)
        assert data.shape == lat.shape
        
        for grid_id, plon, plat in coords[location]:
            for r in  ([ 0.07, 0.1, 0.14, 0.2, 0.3, 0.5, 1, 2] if fine else [ 0.1, 0.25, 0.5, 1, 2, ]):
                if (grid_id, r) not in zones:
                    zones[(grid_id, r)] = (lat - plat) ** 2 + (lon - plon) ** 2 < r ** 2
                zone = zones[(grid_id, r)]
                ct = data[zone].count()
                m = data[zone].mean() if ct > (0 if 'fine' else 3) else np.nan
                s = data[zone].std() if ct >= 3 else np.nan

                sat_data[grid_id][field + '_mean{}'.format(r)] = m
                sat_data[grid_id][field + '_stdev{}'.format(r)] = s
                sat_data[grid_id][field + '_count{}'.format(r)] = ct
                if '2' in grid_id:#.startswith('9'):
                    print(field, '_count{}'.format(r), ct, m ,s )
    return sat_data  


def runTropomi(row, fine = True):
    
    assert row['product'].startswith('tropomi')
    
    file = loadFile(row)
    hdf = h5py.File(file, 'r')
    sat_data = run(hdf, row['location'], fine)
    output = row.copy()
    output['d1'] = cleanDict(sat_data)
    
    s3 = boto3.client('s3')
    zc = zstd.ZstdCompressor(level = 15)

    
    o = pickle.dumps(output)
    c = zc.compress(o)
    
    s3.put_object(Bucket = 'projects-v',
                  Key = 'aqi/tropomi{}/{}'.format('-fine' if fine else '', file.split('/')[-1]),
                  Body = c)
    
    os.remove(file)

    # TODO implement
    return {
        'statusCode': 200,
        'len': len(o),#pickle.dumps(output))
        'clen': len(c),
        'body': json.dumps('Success')
    }








# runTropomi(rows[-4], fine = True)





# rows[-1]


# %%time
# Parallel(os.cpu_count() #// 2
#             )(delayed(runTropomi)(row) for row in rows)








# d = pd.to_datetime(files[(files['product'] == 'misr')
#        & (files.location == 'la')].time_end)
# d = d.sort_values().reset_index(drop = True)
# (d.diff().dt.total_seconds() / (60*60*24)).iloc[:].plot()








# coords['LA']














# sqs.send_message(
#     QueueUrl='https://sqs.us-east-2.amazonaws.com/815054066888/aqi-gfs25-extra',
#     MessageBody=json.dumps((2020, 1, 1, 6, 0))
# )


# t = datetime.datetime(2016, 10, 1)
# while t < datetime.datetime.now():
#     print(t)
#     for hr in range(0, 24, 6):
#         for fwd in (0, ):
#             sqs.send_message(
#                 QueueUrl='https://sqs.us-east-2.amazonaws.com/815054066888/aqi-gfs25-extra',
#                 MessageBody=json.dumps((t.year, t.month, t.day, hr, fwd))
#             )
#     t += datetime.timedelta(days = 1)











# [e[0] for e in pickle.loads(zd.decompress(
#     s3.get_object(Bucket = 'projects-v',
#               Key = 'aqi/gfs-5/{}'.format('gfs.0p25.2016100218.f000.grib2'))['Body'].read()
# )) if 'LA' in e[1]]


# [e[0] for e in pickle.loads(
#     s3.get_object(Bucket = 'projects-v',
#               Key = 'aqi/gfs/{}'.format('gfs.0p25.2016100112.f000.grib2'))['Body'].read()
# ) if 'LA' in e[1]]


# [e for e in pickle.loads(
#     s3.get_object(Bucket = 'projects-v',
#               Key = 'aqi/gfs/{}'.format('gfs.0p25.2019100612.f000.grib2'))['Body'].read()
# ) if 'metre temp' in e[0]]











# grid.wkt


# grid.sort_values('location').wkt


# p = pygrib.open(# 'gfs.0p25.2020072500.f000.grib2')#'gfs.0p25.2020010100.f000.grib2')
#     # 'gfs.0p25.2017072500.f000.grib2'
#     'gfs.0p25.2017072500.f000.grib2'
# )





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

def loadFiles(file):
    zd = zstd.ZstdDecompressor()
    data = []
    with tarfile.TarFile(file, 'r') as tar:
        assim_files = tar.getnames()
        for file in assim_files:
            if len(file) > 0:
                f = tar.extractfile(file)
                data.append(pickle.loads(zd.decompress(f.read())))
    return data





# def listFiles(prefix, bucket = 'projects-v'): 
#     paginator = s3.get_paginator('list_objects_v2')
#     page_iterator = paginator.paginate(Bucket = bucket,
#                                        Prefix = prefix )
#     files = []
#     for page in page_iterator:
#         files.extend([e['Key'] for e in page['Contents']])
#     return files


# def listFiles(prefix):
#     return [os.path.join('cache/', prefix[4:], f) 
#             for f in os.listdir('cache/' + prefix[4:])]








# def loadData(file):
#     zd = zstd.ZstdDecompressor()
#     data = pickle.loads(zd.decompress(open(file, 'rb').read()))
#     return data





def flatten(l): return [e for s in l for e in s]


loc_dict = {'la': 'Los Angeles (SoCAB)', 'tpe': 'Taipei', 'dl': 'Delhi'}

rloc_dict = {v: k for k, v in loc_dict.items()}

tz_dict = grid.groupby('location').tz.first().to_dict()

loc_tz_dict = {rloc_dict[k]: v for k, v in tz_dict.items()}
loc_tz_dict














# def downloadFile(file):
#     s3 = boto3.client('s3')
#     s3.download_file('projects-v', file, 'cache/' + file[4:])


# def downloadFiles(files):
#     for path in set([f.split('/')[1] for f in files]):
#         os.makedirs('cache/{}'.format(path), exist_ok = True)
                    
#     Parallel(os.cpu_count() * 3)(delayed(downloadFile)(file)
#                                      for file in files)





# downloadFiles(assim_files)











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


all_data.tail()


all_data.head()





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


pickle.dump(all_data, open('cache/all_data_{}.pkl'.format(dataset), 'wb'))








x.columns = [c.replace(':', '_') for c in x.columns]


x.shape





# x.fillna(method = 'ffill')


# x.isnull().sum()


# x.isnull().sum().sort_values()


assert x.isnull().sum().sum() == 0





import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
import sklearn


from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


enet_params = {
    'alpha': [  1e-2, 3e-2, 1e-1, 0.3, 1, 3, ],
    'l1_ratio': [ 0.01, 0.03, 0.1, 0.2, 0.5, 0.8, 0.9, ],
}


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
    





from sklearn.metrics import make_scorer, mean_squared_error


def posRMSE(y, y_pred):
    return mean_squared_error(y, y_pred.clip(0, None) * 2/3 + y_pred * 1/3) ** 0.5
pRMSE = make_scorer(posRMSE, greater_is_better = False)
SCORING = pRMSE





COLUMN_WIPE = 0# len(x.columns) // 10











def runENet(drop_cols = [], verbose = 1):
    all_y_enet = []; all_y_pred_enet = []; enet_clfs = []; enet_scalers = []
    for location in x.location.unique():
        if verbose > 0: print(location)
        x_loc = x[x.location == location].drop(columns = drop_cols)
        y_loc = y.reindex(x_loc.index)
        d_loc = d.reindex(x_loc.index)
    

        folds = list(PurgedKFold(4 if dataset == 'pm' else 3).split(x_loc, y_loc, d_loc))
        folds += [(np.arange(0, len(x_loc)), [])] * 2
        for train_fold, test_fold in folds:
            y_preds = []
            for i in range(8):
                scaler = StandardScaler()
                clf = ElasticNet(max_iter = 50000, tol = 1e-3, 
                                 selection = 'random', precompute = True,
                                random_state = datetime.datetime.now().microsecond)                     
                model = RandomizedSearchCV(clf, enet_params, 
                                           scoring = SCORING,
                                           # 'neg_root_mean_squared_error',
                                           cv = RepeatedPurgedKFold(random.randrange(3, 6), 
                                                n_repeats = random.randrange(2, 4), 
                                                    gap = random.randrange(60, 120)), 
                                                   n_iter = random.randrange(4, 7),
                                              random_state = datetime.datetime.now().microsecond)
                # subset = train_fold#[:s].tolist() + train_fold[s + l:].tolist()
                l = random.randrange(0, len(train_fold)//10)
                s = random.randrange(0, len(train_fold) - l)
                subset = train_fold[:s].tolist() + train_fold[s + l:].tolist()
                xt = x_loc.iloc[subset, 2:].copy()
                # for c in misr.columns:
                
                if dataset == 'pm':
                    sample_cols = [c for c in xt.columns if any(z in c for z in 
                                    ['precision', 'air_mass', 'stdev']
                                    + ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc', 
                                       'ifs']
                    )]
                    xt[random.sample(sample_cols, k = random.randrange( len(sample_cols) * 1 // 3,
                                                                        len(sample_cols) * 2 // 3) )] = 0

                
                if dataset == 'tg':
                    if random.random() < 0.5: xt[[c for c in xt.columns if 'precision' in c]] = 0
                    # if random.random() < 0.5: xt[[c for c in xt.columns if 'mean1' in c or 'mean2' in c]] = 0
                # xt[misr.columns] = 0
                # fj_drop = 
                if random.random() < {'Delhi': 0.5, 'Taipei': 0.2}.get(location, 0):
                    xt[[c for c in xt.columns if 'FineMode' in c or 'Injection' in c]] = 0
                # if random.random() < {'Delhi': 0.2, 'Taipei': 0.1}.get(location, 0):
                #     xt[[c for c in xt.columns if 'Optical' in c]] = 0
                if random.random() < {'Delhi': 0.85, 'Taipei': 0.5}.get(location, 0.2):
                    xt[[c for c in xt.columns if c.startswith('ifs')]] = 0

                # xt = x_loc.iloc[subset].copy()
                xt[random.choices(xt.columns[2:],
                        k = int(round(random.random() * COLUMN_WIPE)))] = 0
                if random.random() < 0.3: xt['dayinyear'] = 0;

                model.fit( pd.DataFrame(scaler.fit_transform(xt).astype(np.float32), 
                                        xt.index, xt.columns)
                                 , y_loc.iloc[subset], groups = d_loc.iloc[subset])
                enet_clfs.append(model.best_estimator_)
                enet_scalers.append(scaler)

                
                if i == 0 and verbose > 0: display(pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(
                        columns = 'params'))

                if len(test_fold) > 0:
                    y_pred =  pd.Series( model.predict(
                                pd.DataFrame(scaler.transform(x_loc.iloc[test_fold, 2:]).astype(np.float32),
                                                                       columns = x_loc.columns[2:])),# .clip(0, None), 
                                                                            index = y_loc.iloc[test_fold].index)
                    y_preds.append(y_pred)

            if len(test_fold) > 0:
                y_pred = pd.concat(y_preds)
                y_pred = y_pred.clip(0, None).groupby(y_pred.index).mean().clip(0, None)
                y_pred = ( y_pred.groupby(y_pred.index).mean() )#*2/3
                             # + y_pred.groupby(y_pred.index).median() * 1/3 )

                all_y_enet.append(y_loc.iloc[test_fold])
                all_y_pred_enet.append(y_pred)

                if verbose >= 1: print(location, round(np.corrcoef(y_pred, all_y_enet[-1])[0, 1],4 ) )

                if verbose >= 3:
                    all_y_enet[-1].reset_index(drop = True).ewm(span = 10).mean().plot()
                    plt.plot(y_pred.reset_index(drop = True))

                    plt.title(location); plt.figure()

                    print((x_loc.iloc[subset, 2:].std().clip(1, 1) *
                         model.best_estimator_.coef_ ).sort_values())
                
    return  all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers


all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers = runENet(verbose = 2)





os.makedirs('clfs_{}'.format(dataset), exist_ok = True)


pickle.dump( (all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers), 
                     open('clfs_{}/enet_clfs_{}.pkl'.format(
                                dataset, run_label), 'wb'))


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


cs = []
for location, df in x.groupby('location'):
    c = np.corrcoef( pd.concat(all_y_pred_enet).clip(0, None).reindex(df.index), 
                pd.concat(all_y_enet).reindex(df.index))[0, 1]; cs.append(c)
    print(location, round(c, 3) )
print('\nBlend:', round(np.mean(cs), 3))








lgb_params = {
    'n_estimators': np.arange(200, 400, 10) if dataset == 'pm' else np.arange(300, 600, 20),#[ 150, 200, 300, ],
    'learning_rate': np.arange(0.01, 0.04, 0.003) if dataset == 'pm' else np.arange(0.01, 0.061, 0.005),# [0.03, 0.05, 0.07, ],
    'num_leaves': np.arange(4, 30) if dataset == 'pm' else np.arange(10, 30),# [5, 7, 10, 15, 20,],
    'min_child_weight': np.arange(0.02, 0.1, 0.01),#[  0.1, 0.2, ],
    'min_child_samples': [ 140, 170, 200, 250, 300, 400, 500, 600, 700, 850, 1000, 1400, ]
                                    if dataset == 'pm' else
                         [ 30, 40, 50, 60, 80, 100, 120, 150, 170, 200, 300, 500, 700, ]  
    , 
    'reg_lambda': [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1,  ],
    'reg_alpha':  [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, ],
    'linear_tree': [True, ],
    'subsample': np.arange(0.4, 0.901, 0.05),#[0.3, 0.5,  0.8],
    'subsample_freq': [1],
    'colsample_bytree': np.arange(0.3, 0.71, 0.05) if dataset == 'pm' else np.arange(0.2, 0.81, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
    'colsample_bynode': np.arange(0.4, 1.01, 0.05)  if dataset == 'pm' else np.arange(0.2, 1.01, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
    'linear_lambda': [1e-3, 3e-3, 1e-2, 3e-2, 0.1,],
    # 'max_bins': np.arange(120, 400, 20),
    # 'min_data_in_bin': np.exp(np.arange(np.log(3), np.log(12), 0.1)).astype(int), #np.arange(2, 10),# [2, 3, 4, 5, 10],
                # [ 192, 
                 #255, 255, 384, 512],
    'min_data_per_group': [10, 20, 50, 100],
    'max_cat_threshold': [  8, 16, 32, ], 
    
    'cat_l2': [0.1, 1, 10],#     if dataset == 'pm' else [1e-3, 1e-2, 1e-1],
    'cat_smooth': [0.1, 1, 10],#      if dataset == 'pm' else [1e-3, 1e-2, 1e-1],
}


FAST = False
GAUSSIAN = 0.05 #if dataset == 'pm' else 0.1
COLUMN_WIPE = len(x.columns) // 5 #  if dataset == 'pm' else 10)





def runLGB(drop_cols = [], verbose = 1, n_bags = 8, ):
    all_y_lgb = []; all_y_pred_lgb = []; tidx = 0;     lgb_clfs = []; lgb_scalers = [];
    for location in x.location.unique():
        random.seed(datetime.datetime.now().microsecond);
        np.random.seed(datetime.datetime.now().microsecond)
        if FAST and location != 'Delhi': continue;
        if verbose > 0: print(location)
        x_loc = x[x.location == location].drop(columns = drop_cols)#.iloc[:, :4]
        y_loc = y.reindex(x_loc.index)
        d_loc = d.reindex(x_loc.index)

        folds = list((PurgedKFold(4) if dataset == 'pm' else PurgedKFold(3, gap = 20))
                         .split(x_loc, y_loc, d_loc)) 
        folds += [(np.arange(0, len(x_loc)), [])] * 3
        for train_fold, test_fold in folds:
            y_preds = []
            for i in range(n_bags):# if location == 'Delhi' else 4):
                model = RandomizedSearchCV(lgb.LGBMRegressor(seed = datetime.datetime.now().microsecond,
                                                                 # n_jobs = 2, # os.cpu_count()
                                                            ), lgb_params,
                                               cv = RepeatedPurgedKFold( random.randrange(3, 6),
                                                                         n_repeats = random.randrange(1, 3),
                                                                         gap = random.randrange(60, 120)), 
                                               n_iter = random.randrange(3, 5),
                                           scoring = SCORING,
                                           n_jobs = -1, #os.cpu_count(),
                                           random_state = datetime.datetime.now().microsecond)
                l = random.randrange(0, len(train_fold)//10)
                s = random.randrange(0, len(train_fold) - l)
                subset = train_fold[:s].tolist() + train_fold[s + l:].tolist()

                xt = x_loc.iloc[subset].copy()
                xt.iloc[:, 2:] += ( GAUSSIAN * random.random()  * 
                                       np.random.default_rng().standard_normal( size = xt.iloc[:, 2:].shape)
                                                    * xt.iloc[:, 2:].std().values[None, :] )
                for c in random.choices(xt.columns[2:], k = int(round(random.random() * COLUMN_WIPE))):
                    xt[c] = 0
                if dataset == 'pm' and random.random() < 0.3: xt['dayinyear'] = 0;
                
                sample_cols = [c for c in xt.columns if any(z  in c for z in 
                                ['precision', 'air_mass', 'stdev']
                                + ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc', 'ifs' ]
                )]
                xt[random.sample(sample_cols, k = random.randrange( len(sample_cols) * 1 // 3,
                                                                    len(sample_cols) * 2 // 3) )] = 0
                
                if random.random() < {'Delhi': 0.7, 'Taipei': 0.2}.get(location, 0):
                    xt[[c for c in xt.columns if 'FineMode' in c or 'Injection' in c]] = 0
                if random.random() < {'Delhi': 0.1, 'Taipei': 0.1}.get(location, 0):
                    xt[[c for c in xt.columns if 'Optical' in c]] = 0
                if random.random() < {'Delhi': 1.0, 'Taipei': 0.2}.get(location, 0.2):
                    xt[[c for c in xt.columns if c.startswith('ifs')]] = 0
  
                if random.random() < {'Delhi': 0.9,}.get(location, 0.12):
                    xt[[c for c in xt.columns if any(z + '_' in c for z in 
                         ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc',  ] ) ]] = 0

                scaler = StandardScaler()
                xt.iloc[:, 2:] = scaler.fit_transform(xt.iloc[:, 2:]).astype(np.float32)

                model.fit( xt, y_loc.iloc[subset], groups = d_loc.iloc[subset],
                         )

                lgb_clfs.append(model.best_estimator_)
                lgb_scalers.append(scaler)

                if len(test_fold) > 0:
                    xtst =  x_loc.iloc[test_fold].copy();
                    xtst.iloc[:, 2:] = scaler.transform(xtst.iloc[:, 2:]).astype(np.float32)

                    y_pred =  pd.Series( model.predict(xtst ), index = y_loc.iloc[test_fold].index)

                    y_preds.append( y_pred)

                df = pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(columns = 'params')
                if i == 0 and verbose > 1: # df.mean_test_score.min() < -0: 
                    display(df); print()

            if len(test_fold) > 0: 
                y_pred = pd.concat(y_preds)
                y_pred = ( y_pred.groupby(y_pred.index).mean() )#* 2/3
                             # + y_pred.groupby(y_pred.index).median() * 1/3 )

                all_y_lgb.append(y_loc.iloc[test_fold])
                all_y_pred_lgb.append(y_pred)

                if verbose >= 3:
                    all_y_lgb[-1].reset_index(drop = True).ewm(span = 10).mean().plot()
                    plt.plot(y_pred.reset_index(drop = True), linewidth = 0.8)
                    try: plt.plot(all_y_pred_enet[tidx].clip(0, None).reset_index(drop = True))
                    except: pass

                    plt.title(location); plt.figure()
                tidx += 1

                if verbose > 0: print(location, round(np.corrcoef(y_pred, all_y_lgb[-1])[0, 1],4 ) )

    return  all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers


all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers = runLGB(verbose = 2, )


pickle.dump( (all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers), 
                     open('clfs_{}/lgb_clfs_{}.pkl'.format(
                                dataset, run_label), 'wb'))





# # print(round(np.corrcoef( pd.concat(all_y_pred_enet).clip(0, None), pd.concat(all_y_lgb))[0, 1], 3)  )
# print(round(np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None), pd.concat(all_y_lgb))[0, 1], 3) )
# print(round(np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None)
#                   +  0.1  * pd.concat(all_y_pred_enet).clip(0, None)
#                   , pd.concat(all_y_lgb))[0, 1], 4) )
#  # without weather ~0.79 lgb


cs = []
for location, df in x.groupby('location'):
    c = np.corrcoef( pd.concat(all_y_pred_lgb).clip(0, None).reindex(df.index) 
                    + 0.1 * pd.concat(all_y_pred_enet).clip(0, None).reindex(df.index) 
                , pd.concat(all_y_lgb).reindex(df.index))[0, 1]; cs.append(c)
    print(location, 
          round(c, 3) )
print('\nBlend: ', round(np.mean(cs), 3))








len(lgb_clfs)














submission = pd.read_csv('data_{}/submission_format.csv'.format(dataset))


submission['dayinyear'] = pd.to_datetime(submission.datetime).dt.dayofyear
submission['dayofweek'] = pd.to_datetime(submission.datetime).dt.dayofweek


submission['location'] = grid.set_index('grid_id').location.reindex(submission.grid_id).values
submission['location'] = submission.location.astype('category')
submission['grid_id'] = submission.grid_id.astype(x.grid_id.dtype)#'category')


submission = addGFS(submission)
submission = addIFS(submission)
submission = addSat(submission, maiac if dataset == 'pm' else tropomi)
if ASSIM: submission = addSat(submission, assim)





pickle.dump(submission, open('cache/submission_{}.pkl'.format(dataset), 'wb'))





xs = submission[x.columns]


submission.location.value_counts()





clf_path = 'clfs_{}'.format(dataset)
all_clfs = os.listdir(clf_path)


# enet_tuples = [pickle.load(open( os.path.join(clf_path, f), 'rb'))
#                    for f in all_clfs if 'enet_clfs' in f] if dataset == 'pm' else []
# lgb_tuples = [pickle.load(open( os.path.join(clf_path, f), 'rb'))
#                    for f in all_clfs if 'lgb_clfs' in f]


enet_tuples = [(all_y_enet, all_y_pred_enet, enet_clfs, enet_scalers)]
lgb_tuples = [(all_y_lgb, all_y_pred_lgb, lgb_clfs, lgb_scalers)]


len(lgb_tuples)


len(enet_tuples)


# labels


y_true = pd.concat(flatten([e[0] for e in lgb_tuples]))
y_true = y_true.groupby(y_true.index).mean()


y_pred_lgb = pd.concat(flatten([e[1] for e in lgb_tuples]))
y_pred_lgb =  y_pred_lgb.groupby(y_pred_lgb.index).mean() #* 2/3
                  # + y_pred_lgb.groupby(y_pred_lgb.index).median() * 1/3)


if dataset == 'pm':
    y_pred_enet = pd.concat(flatten([e[1] for e in enet_tuples]))
    y_pred_enet = ( y_pred_enet.groupby(y_pred_enet.index).mean() )
                      # + y_pred_enet.groupby(y_pred_enet.index).mean() * 1/3)


cs = []
for location, df in x.groupby('location'):
    c = np.corrcoef( y_pred_lgb.clip(0, None).reindex(df.index) 
                    + 1/5 * ( y_pred_enet.clip(0, None).reindex(df.index) if dataset == 'pm' else 0), 
                y_true.reindex(df.index))[0, 1]; cs.append(c)
    print(location, 
          round(c, 3) )
print('\nBlend: ', round(np.mean(cs), 3))








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





len(lgb_clfs)


# %%time
lgb_ys = []
for clf_idx, clf in enumerate(lgb_clfs):
    x_loc = xs[xs.location == x.location.unique()[clf_idx * 3 // len(lgb_clfs)]].copy()
    x_loc.iloc[:, 2:] = lgb_scalers[clf_idx].transform(x_loc.iloc[:, 2:])
    lgb_ys.append(pd.Series(clf.predict(x_loc), index = x_loc.index))
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





# out[pd.to_datetime(out.datetime) < datetime.datetime(2018, 1, 1)]


# if dataset == 'pm' and ASSIM:
#     pivot = '2018-01-10'
#     b1 = pd.read_csv('submissions_pm/blend1.csv')
#     mix = pd.concat((
#         b1[b1.datetime < pivot],
#         out[out.datetime >= pivot] ) )
#     assert out.shape == mix.shape
#     out = mix





# for g in out.grid_id.unique():
#     out[out.grid_id == g].set_index(pd.to_datetime(out[out.grid_id == g].datetime)).value.plot(
#         marker = '.', linewidth = 0.1,)# markersize = 3)
# plt.xlim(pd.to_datetime(submission.datetime).min(), 
#          pd.to_datetime(submission.datetime).min() + datetime.timedelta(days = 400));


# for g in out.grid_id.unique():
#     out[out.grid_id == g].set_index(pd.to_datetime(out[out.grid_id == g].datetime)).value.plot(
#         marker = '.', linewidth = 0.1,)# markersize = 3)
# plt.xlim(datetime.datetime(2020, 10, 15), datetime.datetime(2021, 9, 1));





# if dataset == 'pm':
#     sp = pd.read_csv('submissions_pm/lgb_baseline.csv')
#     sp2 = pd.read_csv('submissions_pm/lgb_2.csv')
#     g1 = pd.read_csv('submissions_pm/first_gfs.csv')
#     m1 = pd.read_csv('submissions_pm/maiac.csv')
#     m2 = pd.read_csv('submissions_pm/maiac2.csv')
#     m3 = pd.read_csv('submissions_pm/maiac3.csv')
#     b1 = pd.read_csv('submissions_pm/blend1.csv')
#     b1f = pd.read_csv('submissions_pm/blend1f.csv')
# else:
#     sp = pd.read_csv('../submissions_tg/third_gfs.csv') 
#     s1 = pd.read_csv('../submissions_tg/sat1.csv') 
#     s2 = pd.read_csv('../submissions_tg/sat2.csv') 
#     s4 = pd.read_csv('../submissions_tg/sat4.csv') 
#     a1 = pd.read_csv('../submissions_tg/assim1.csv') 
#     n = pd.read_csv('../submissions_tg/new.csv') 
#     st1 = pd.read_csv('../submissions_tg/stack1.csv') 
#     n2 = pd.read_csv('submissions_tg/new.csv')


# if dataset == 'pm':
#     # plt.scatter(sp.value, out.value, s= 0.1);
#     # plt.scatter(sp2.value, out.value, s= 0.1);
#     # plt.scatter(g1.value, out.value, s= 0.1);
#     plt.scatter(m1.value, out.value, s= 0.1);
#     plt.scatter(m2.value, out.value, s= 0.1);
#     plt.scatter(m3.value, out.value, s= 0.1);
#     plt.scatter(b1.value, out.value, s= 0.1);
#     plt.scatter(b1f.value, out.value, s= 0.1);
# else:
#     plt.scatter(sp.value, out.value, s= 0.1);
#     plt.scatter(s1.value, out.value, s= 0.1);
#     plt.scatter(s2.value, out.value, s= 0.1);
#     plt.scatter(s4.value, out.value, s= 0.1);
#     plt.scatter(a1.value, out.value, s= 0.1);
#     plt.scatter(n.value, out.value, s= 0.1);
#     plt.scatter(n2.value, out.value, s= 0.1)
#     plt.scatter(st1.value, out.value, s= 0.1)


# if dataset == 'pm': 
#     print(np.corrcoef((g1.value, m1.value, m2.value, m3.value, b1.value, b1f.value, out.value)).round(4))
# else:
#     print(np.corrcoef((sp.value, s1.value, s2.value, s4.value, a1.value, n.value, n2.value, st1.value, out.value)).round(4))








out[out.datetime < '2021-04-01']





os.makedirs('submissions_{}'.format(dataset), exist_ok = True)
out.to_csv('submissions_{}/new.csv'.format(dataset), index = False)




