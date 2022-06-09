#!/usr/bin/env python
# coding: utf-8

# !rm -r inference


# !pip install -r requirements.txt





import json
import sys, os
import requests
import datetime
import numpy as np
import pickle
import time
import random
import zstandard as zstd


import tarfile


import pandas as pd


import boto3
import botocore
from botocore.client import Config


import pygrib
import pydap
import xarray as xr
import h5py


from collections import defaultdict


from joblib import Parallel, delayed, parallel_backend

import subprocess


import cv2





secure = dict([e.split('=') for e in open('secure.txt', 'r').read().split('\n')])


infer = dict([e.split('=') for e in open('infer.txt', 'r').read().split('\n')])
# infer = {k: v.split(',') for k, v in infer.items()}





dataset = 'tg'


labels = pd.read_csv('data_{}/train_labels.csv'.format(dataset))

grid = pd.concat( (
        pd.read_csv('data_tg/grid_metadata.csv'),
) ).drop_duplicates().reset_index(drop = True) 

submission = pd.read_csv('data_{}/submission_format.csv'.format(dataset))

files = pd.read_csv('data_{}/{}_satellite_metadata{}.csv'.format(
                    dataset, *(('pm25', '') if dataset == 'pm' 
                                else ('no2', '_0AF3h09'))))



labels['location'] = grid.set_index('grid_id')['location'].reindex(labels.grid_id).values
labels['datetime'] = pd.to_datetime(labels.datetime)

submission['location'] = grid.set_index('grid_id').location.reindex(submission.grid_id).values


files.time_end = pd.to_datetime(files.time_end)








cities = {
 'Taipei': ( (121.5, 121.5), (25.0, 25) ),
 'Delhi': ( (77.0, 77.25), (28.75, 28.5) ),
 'LA': ((360-118.25, 360-117.75), (34.0, 34.0) ) 
}


feats = [
    #  (6, 'Maximum/Composite radar reflectivity:dB (instant):regular_ll:atmosphere:level 0', ),
    #  (7, 'Visibility:m (instant):regular_ll:surface:level 0', ),
     (11, 'Wind speed (gust):m s**-1 (instant):regular_ll:surface:level 0', ),
(402, 'Surface pressure:Pa (instant):regular_ll:surface:level 0'),
# (404, 'Temperature:K (instant):regular_ll:surface:level 0'),
# (405, 'Soil Temperature:K (instant):regular_ll:depthBelowLandLayer:levels 0.0-0.1 m'),
(406, 'Volumetric soil moisture content:Proportion (instant):regular_ll:depthBelowLandLayer:levels 0.0-0.1 m'),
(415, '2 metre temperature:K (instant):regular_ll:heightAboveGround:level 2 m'),
(416, '2 metre specific humidity:kg kg**-1 (instant):regular_ll:heightAboveGround:level 2 m'),
# (417, '2 metre dewpoint temperature:K (instant):regular_ll:heightAboveGround:level 2 m:'),#fcst time 0 hrs:from 202001010000
(418, '2 metre relative humidity:% (instant):regular_ll:heightAboveGround:level 2 m:'), #fcst time 0 hrs:from 202001010000
(419, 'Apparent temperature:K (instant):regular_ll:heightAboveGround:level 2 m:'),#fcst time 0 hrs:from 202001010000
(420, '10 metre U wind component:m s**-1 (instant):regular_ll:heightAboveGround:level 10 m:'),#fcst time 0 hrs:from 202001010000
(421, '10 metre V wind component:m s**-1 (instant):regular_ll:heightAboveGround:level 10 m:'),#fcst time 0 hrs:from 202001010000
# (435, 'Precipitable water:kg m**-2 (instant):regular_ll:atmosphereSingleLayer:level 0 considered as a single layer'),#:fcst time 0 hrs:from 202001010000
(436, 'Cloud water:kg m**-2 (instant):regular_ll:atmosphereSingleLayer:level 0 considered as a single layer:'),#fcst time 0 hrs:from 202001010000
(437, 'Relative humidity:% (instant):regular_ll:atmosphereSingleLayer:level 0 considered as a single layer:'),#fcst time 0 hrs:from 202001010000
(438, 'Total ozone:DU (instant):regular_ll:atmosphereSingleLayer:level 0 considered as a single layer:'),#fcst time 0 hrs:from 202001010000        
    # (424,  'Precipitation rate:kg m**-2 s**-1 (instant):regular_ll:surface:level 0'),
    # (484, 'Temperature:K (instant):regular_ll:pressureFromGroundLayer', ),
    # (485, 'Relative humidity:% (instant):regular_ll:pressureFromGroundLayer:levels 3000-0 Pa', ),
    # (486, 'Specific humidity:kg kg**-1 (instant):regular_ll:pressureFromGroundLayer:levels 3000-0 Pa', ),
    # (487, 'U component of wind:m s**-1 (instant):regular_ll:pressureFromGroundLayer:levels 3000-0 Pa', ),
    # (488, 'V component of wind:m s**-1 (instant):regular_ll:pressureFromGroundLayer:levels 3000-0 Pa', ),
    # (520, 'Pressure reduced to MSL:Pa (instant):regular_ll:meanSea:level 0:', ),        
    ]





cities2 = {
 'tpe': ( 121.5, 25 ),
 'dl': ( 77.0, 28.5 ),
 'la': (-118.25, 34.0 ) 
}

coords = {'la': [('3A3IE', -117.9114, 34.1494),
  ('3S31A', -117.9563, 33.8142),
  ('7II4T', -118.0461, 34.0006),
  ('8BOQH', -118.4504, 34.0379),
  ('A2FBI', -117.4173, 34.0006),
  ('A5WJI', -117.9563, 33.9261),
  ('B5FKJ', -117.5071, 34.1123),
  ('C8HH7', -116.519, 33.8516),
  ('DHO4M', -118.3605, 34.1866),
  ('DJN0F', -117.6419, 34.1123),
  ('E5P9N', -117.5071, 34.0006),
  ('FRITQ', -118.1809, 33.8516),
  ('H96P6', -118.5402, 34.1866),
  ('HUZ29', -117.2825, 34.1123),
  ('I677K', -117.5071, 34.0751),
  ('IUON3', -117.7317, 34.0751),
  ('JNUQF', -118.2258, 33.8142),
  ('PG3MI', -118.2258, 34.0751),
  ('QH45V', -118.4504, 33.9634),
  ('QJHW4', -118.5402, 34.3722),
  ('QWDU8', -118.1359, 34.1494),
  ('VBLD0', -118.2258, 33.8888),
  ('VDUTN', -117.9114, 33.8142),
  ('WT52R', -116.8783, 33.9261),
  ('X5DKW', -117.597, 34.0379),
  ('Z0VWC', -118.1809, 33.7769),
  ('ZP1FZ', -117.8665, 34.1494),
  ('ZZ8JF', -117.3275, 33.6648)],
 'tpe': [('1X116', 121.5033, 24.998),
  ('90BZ1', 121.5482, 25.0387),
  ('9Q6TA', 121.5482, 25.0794),
  ('KW43U', 121.5931, 25.0387),
  ('VR4WG', 121.5033, 25.0794),
  ('XJF9O', 121.5033, 25.0387),
  ('XNLVD', 121.5033, 25.1201)],
 'dl': [('1Z2W7', 77.2821, 28.5664),
  ('6EIL6', 77.0575, 28.5664),
  ('7334C', 77.1024, 28.5664),
  ('78V83', 76.9227, 28.5664),
  ('7F1D1', 77.1024, 28.6058),
  ('8KNI6', 77.2821, 28.4874),
  ('90S79', 77.1922, 28.6452),
  ('A7UCQ', 77.2372, 28.6058),
  ('AZJ0Z', 77.2372, 28.724),
  ('C7PGV', 77.1922, 28.5269),
  ('CPR0W', 77.2821, 28.6846),
  ('D72OT', 77.1473, 28.724),
  ('D7S1G', 77.327, 28.6846),
  ('E2AUK', 77.0126, 28.6058),
  ('GAC6R', 77.1024, 28.7634),
  ('GJLB2', 77.1024, 28.4874),
  ('GVQXS', 77.1922, 28.6846),
  ('HANW9', 77.1922, 28.5664),
  ('HM74A', 77.1024, 28.6846),
  ('IUMEZ', 77.2372, 28.6452),
  ('KZ9W9', 77.1473, 28.6452),
  ('NE7BV', 77.1024, 28.8421),
  ('P8JA5', 77.2372, 28.5664),
  ('PJNW1', 77.1922, 28.724),
  ('PW0JT', 76.9227, 28.6846),
  ('S77YN', 77.0575, 28.724),
  ('SZLMT', 77.1473, 28.6846),
  ('UC74Z', 77.2821, 28.5269),
  ('VXNN3', 77.1473, 28.8028),
  ('VYH7U', 77.0575, 28.7634),
  ('WZNCR', 77.1473, 28.5664),
  ('YHOPV', 77.2821, 28.6452),
  ('ZF3ZW', 77.0575, 28.6846)]}





def cleanDict(d):
    return {k: cleanDict(v) for k, v in d.items() } if isinstance(d, defaultdict) else d











def processGFS(file, d):
    p = pygrib.open(file)
    lat, lon = p[1].latlons()
    spots = {}
    for city, ( (lonmin, lonmax) , (latmin, latmax) ) in cities.items():
        xmin = np.argmax( (lat == latmin).sum(axis = 1)  )#[0]
        xmax = np.argmax( (lat == latmax).sum(axis = 1)  )#[0]
        ymin = np.argmax( (lon == lonmin).sum(axis = 0)  )#[0]
        ymax = np.argmax( (lon == lonmax).sum(axis = 0)  )#[0]
        spots[city] = ((xmin, xmax), (ymin, ymax))

    data = []
    for e in p:
        if any(z in str(e) for i, z in feats): 
            arr = e.values
            assert arr.shape == lat.shape
            for spot, ((xmin, xmax), (ymin, ymax)) in spots.items():
                data.append( (str(e), 
                                spot,
                ((lat[xmin - d :xmax + 1 + d, ymin - d :ymax + 1 + d].min(),
                  lat[xmin - d :xmax + 1 + d, ymin - d :ymax + 1 + d].max()),
                 (lon[xmin - d :xmax + 1 + d, ymin - d :ymax + 1 + d].min(),
                  lon[xmin - d :xmax + 1 + d, ymin - d :ymax + 1 + d].max())),
                
                arr[xmin - d :xmax + 1 + d, ymin - d :ymax + 1 + d].astype(np.float32),
                
                arr[xmin:xmax + 1, ymin:ymax + 1].mean() ) );
                # if len(data) == 1: print(data)
                # print(data); return data
    return data
    # break;





def pullGFS(files):
    results = []
    for i in range(10):
        try:
            pswd = secure['password']
            values = {'email' : secure['username'], 'passwd' : pswd, 'action' : 'login'}
            login_url = 'https://rda.ucar.edu/cgi-bin/login'

            ret = requests.post(login_url, data=values)
            if ret.status_code != 200:
                print('Bad Authentication'); time.sleep(i); continue;

        except Exception as e:
            print(e)
            time.sleep(i)
                
    print('Downloading {} gfs files'.format(len(files)))    
    # print(filelist); return;

    dspath = 'https://rda.ucar.edu/data/ds084.1/'
    save_dir = '/tmp/'
        
    zc = zstd.ZstdCompressor(level = 9)
    for file in files:
        start = time.time()
        for i in range(10):
            try:
                filename = dspath + file
                outfile = save_dir + os.path.basename(filename)
                print('Downloading', file)
                with requests.get(filename, cookies = ret.cookies, 
                    allow_redirects = True, stream=True) as r:
                    r.raise_for_status()
                    with open(outfile, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024): 
                            f.write(chunk)
            
                s = os.path.getsize(outfile); 
                data = processGFS(outfile, 5)
                os.remove(outfile)
                pkl = pickle.dumps(data)
                compr = zc.compress(pkl)

                os.makedirs('inference/gfs-5/', exist_ok = True)
                with open('inference/gfs-5/{}'.format(os.path.basename(filename)), 'wb') as f:
                    f.write(compr)

                results.append({
                    # 'statusCode': 200,
                    'file': os.path.basename(filename),
                    'body': s/1e6, #os.path.getsize(outfile), #json.dumps('Hello from Lambda!'),
                    'outlen': len(pkl),#len(pickle.dumps(data)),
                    'outlen-compr': len(compr),#zc.compress(pickle.dumps(data))),
                    'elaspsed_time': round(time.time() - start, 1)
                    # 'data': json.dumps(data),
                }); break;
        
            except Exception as e:
                print(e)
                time.sleep(i)
                try: os.remove(outfile)
                except: pass;
    return results








ifs_tags = ['128_057_uvb',
 '128_134_sp',
 '128_136_tcw',
 '128_137_tcwv',
 '128_146_sshf',
 '128_147_slhf',
 '128_164_tcc',
 '128_165_10u',
 '128_166_10v',
 '128_167_2t',
 '128_168_2d',
 '128_169_ssrd',
 '128_175_strd',
 '128_176_ssr',
 '128_177_str',
 '128_189_sund',
 '128_206_tco3',
 '128_228_tp',
 '128_243_fal',
 '128_244_fsr',
 '128_245_flsr',
 '228_246_100u',
 '228_247_100v']





def processIFS(file):
    dataset = xr.open_dataset(file)
    vars = list(dataset.variables)
    assert len(vars) == 5 if 'oper.an' in file else 6 if 'oper.fc' in file else -1;
    # assert vars[-4:] == ['latitude', 'longitude', 'time', 'utc_date']
    
    field = vars[0]
    name = dataset.variables[field].attrs['long_name']
    # print(name)     
    clean_name = name.lower().replace(' ', '_').replace('-', '_')
    # print(clean_name)
    
    sat_data = defaultdict(lambda: defaultdict(dict))
    for location, (clon, clat) in cities2.items():
        minimum_latitude = clat + 8
        minimum_longitude = (clon - 10 ) % 360
        maximum_latitude = clat - 8
        maximum_longitude = (clon + 10) % 360
    

        data = dataset[field].loc[{
                                           'latitude':slice(minimum_latitude,maximum_latitude),
                                           'longitude':slice(minimum_longitude,maximum_longitude)}]
        # print(data.shape)
        
        
        a = data
        
        v = a.values
        lat = np.tile( np.stack([a['latitude']], axis = 1), ( 1, v.shape[-1]))
        lon = np.tile( np.stack([a['longitude']], axis = 0), ( v.shape[-2], 1))
        assert v.shape == (4, 227, 285) if 'oper.an' in file else (2, 2, 227, 285) if 'oper.fc' in file else None
        if 'oper.an' in file: 
            times = a.time.values.astype('datetime64[s]')
            assert len(times) == 4
            assert v.shape[0] == len(times)
        elif 'oper.fc' in file:
            start_times = np.repeat(a.forecast_initial_time.values.astype('datetime64[s]'), 2)
            deltas = np.tile([np.timedelta64(int(h), 'h') for h in a.forecast_hour.values], 2)
            times = list(zip(start_times, deltas))
            # print(times)
            v = v.reshape(4, v.shape[-2], v.shape[-1])
            # print(times); print(deltas)
        assert v.shape[1:] == lat.shape
        assert v.shape[1:] == lon.shape
        
        
    
        zones = {}# defaultdict(dict)
        
        for tidx, t in enumerate(times):
            for grid_id, plon, plat in coords[location]:
                for r in [ 0.05, 0.1, 0.2, 0.5, 1, 2, 5]:
                    if (grid_id, r) not in zones:
                        zones[(grid_id, r)] = (lat - plat) ** 2 + (lon - plon%360) ** 2 < r ** 2
                    zone = zones[(grid_id, r)]
                    # ct = len(v[tidx][zone])#.count()
                    sat_data[t][grid_id][clean_name + '_mean{}'.format(r)] = v[tidx][zone].mean() #if ct > 3 else np.nan
    
    # for k, v in sat_data.items():
    #     print(k, len(v))
    
    # print(v['1X116']) 
    
    def clean(d):
        if isinstance(d, defaultdict):
            d = {k: clean(v) for k, v in d.items()}
        return d
    
    return clean(sat_data)
        


def pullIFS(files):
    results = []
    for i in range(10):
        try:
            pswd = secure['password']
            values = {'email' : secure['username'], 'passwd' : pswd, 'action' : 'login'}
            login_url = 'https://rda.ucar.edu/cgi-bin/login'

            ret = requests.post(login_url, data=values)
            if ret.status_code != 200:
                print('Bad Authentication'); time.sleep(i); continue;

        except Exception as e:
            print(e)
            time.sleep(i)

    save_dir = '/tmp/'
    dspath = 'https://rda.ucar.edu/data/ds113.1/'
    
    print('Downloading {} ifs files'.format(len(files)))    

    
    zc = zstd.ZstdCompressor(level = 9)
    for file in files:
        start = time.time()
        for i in range(10):
            try:
                filename = dspath + file
                outfile = save_dir + os.path.basename(filename)
                print('Downloading', file)
                with requests.get(filename, cookies = ret.cookies, 
                    allow_redirects = True, stream=True) as r:
                    r.raise_for_status()
                    with open(outfile, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024): 
                            f.write(chunk)
            
                s = os.path.getsize(outfile); 
                data = processIFS(outfile)
                os.remove(outfile)
                pkl = pickle.dumps(data)
                compr = zc.compress(pkl)

                os.makedirs('inference/ifs/', exist_ok = True)
                with open('inference/ifs/{}'.format(os.path.basename(filename)), 'wb') as f:
                    f.write(compr)

                results.append({
                    # 'statusCode': 200,
                    'file': os.path.basename(filename),
                    'body': s/1e6, #os.path.getsize(outfile), #json.dumps('Hello from Lambda!'),
                    'outlen': len(pkl),#len(pickle.dumps(data)),
                    'outlen-compr': len(compr),#zc.compress(pickle.dumps(data))),
                    'elaspsed_time': round(time.time() - start, 1)
                }); break;        

            except Exception as e:
                print(e)
                time.sleep(i)
                try: os.remove(outfile)
                except: pass
    return results














tropomi_fields = ['nitrogendioxide_tropospheric_column',
 'nitrogendioxide_tropospheric_column_precision',
 'air_mass_factor_troposphere',
 'air_mass_factor_total']
 





def loadFileS3(row):    
    my_config = Config(signature_version = botocore.UNSIGNED)
    s3a = boto3.client('s3', config = my_config)
    
    filename, url, cksum, sz = [row[k] for k in ['granule_id', 'us_url', 'cksum', 'granule_size']]
    print(filename, url, cksum, sz) 
    file = '/tmp/' + filename
    
    bucket = url.split('//')[-1].split('/')[0]
    key = '/'.join(url.split('//')[-1].split('/')[1:])
    
    s = s3a.download_file(bucket, key, file)
 
    assert ( subprocess.check_output(['cksum',file])
            .decode('utf-8').split(' ')[:2] == [str(cksum), str(sz)])
    return file
 


def processTropomi(hdf, location, fine = True):
    zones = {}; # defaultdict(dict)
    sat_data = defaultdict(lambda: defaultdict(dict))
    hp = hdf['PRODUCT']
    lat = hp['latitude'][:][0]#.values
    lon = hp['longitude'][:][0]#.values
    
    for field in tropomi_fields:
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
                # if '2' in grid_id:#.startswith('9'):
                    # print(field, '_count{}'.format(r), ct, m ,s )
    return sat_data  


def pullTropomi(row, fine = True):
    results = []
    
    start = time.time()
    assert row['product'].startswith('tropomi')
    
    file = loadFileS3(row)
    hdf = h5py.File(file, 'r')
    s = os.path.getsize(file); 
    sat_data = processTropomi(hdf, row['location'], fine)
    output = row.copy()
    output['d1'] = cleanDict(sat_data)
    
    s3 = boto3.client('s3')
    zc = zstd.ZstdCompressor(level = 15)

    pkl = pickle.dumps(output)
    compr = zc.compress(pkl)
    
    filename = file.split('/')[-1]
    os.makedirs('inference/tropomi-fine/', exist_ok = True)
    with open('inference/tropomi-fine/{}'.format(filename), 'wb') as f:
        f.write(compr)
        
    try:     os.remove(file)
    except Exception as e: print(e); pass
    return {
        # 'statusCode': 200,
        'file': os.path.basename(filename),
        'body': s/1e6, #os.path.getsize(outfile), #json.dumps('Hello from Lambda!'),
        'outlen': len(pkl),#len(pickle.dumps(data)),
        'outlen-compr': len(compr),#zc.compress(pickle.dumps(data))),
        'elaspsed_time': round(time.time() - start, 1)
    }; 














def loadAssim(field, location, year, month, min_day, max_day):
    url = 'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/aqc_tavg_1hr_g1440x721_v1'
    DATASET = xr.open_dataset(url)
    start_time = np.datetime64('{}-{:02d}-{:02d} 00:00:00'.format(year, month, min_day))
    end_time = np.datetime64('{}-{:02d}-{:02d} 23:59:00'.format(year, month, max_day))
    # end_time = np.datetime64('{}-01-01 00:00:00'.format(year + 1))
    minimum_latitude = min([e[-1] for e in coords[location]]) - 3
    minumum_longitude = min([e[-2] for e in coords[location]]) - 3
    maximum_latitude = max([e[-1] for e in coords[location]]) + 3
    maximum_longitude = max([e[-2] for e in coords[location]]) + 3

    data = DATASET[field].loc[{'time':slice(start_time,end_time),
                                       'lat':slice(minimum_latitude,maximum_latitude),
                                       'lon':slice(minumum_longitude,maximum_longitude)}]
    return data


def processAssim(a, location, field):
    t = a.time.values.astype('datetime64[s]')
    sat_data = defaultdict(dict)
    v = a.values[0]
    if (v == 1.0e15).sum() > 0:
        return {'location': location, 'time_end': t, 'd1': cleanDict(sat_data)}
 
    lat = np.tile( np.stack([a['lat']], axis = 1), ( 1, v.shape[1]))
    lon = np.tile( np.stack([a['lon']], axis = 0), ( v.shape[0], 1))

    lat = cv2.resize(lat, None, fx = 5, fy = 5)
    lon = cv2.resize(lon, None, fx = 5, fy = 5)
    v2 = cv2.resize(v, None, fx = 5, fy = 5)

    zones = {}
    for grid_id, plon, plat in coords[location]:
        for r in [ 0.1, 0.25, 0.5, 1, 2, ]:
            if (grid_id, r) not in zones:
                z = (lat - plat) ** 2 + (lon - plon) ** 2 < r ** 2
                zones[(grid_id, r)] = z#, z.sum())
            zone = zones[(grid_id, r)]
            m = v2[zone].mean()#, 1#zone.sum()
            sat_data[grid_id][field + '_mean{}'.format(r)] = m #data[zone].mean()# if ct > 3 else np.nan

    return {'location': location, 'time_end': t, 'd1': cleanDict(sat_data)}


def pullAssim(year, month, min_day, max_day):
    for field in ['no2', 'so2', 'co', 'o3', 'pm25_rh35_gcc']:
        for location in coords.keys():
            start = time.time()
            for i in range(10):
                try:
                    data = loadAssim(field, location, year, month, min_day, max_day)
                    print('{}-{:02d} {} {} {}'.format(
                            year, month, field, location, len(data)))
                    # assert len(data) == 24

                    with parallel_backend('threading'):
                        r = Parallel(os.cpu_count())(
                            delayed(processAssim)(a, location, field) for a in data) 
                    zc = zstd.ZstdCompressor(level = 9)
                    out = pickle.dumps(r)
                    compr = zc.compress(out)

                    filename = '{}_{}_{}_{:02}.pkl'.format(
                              field, location, year, month, )
                    os.makedirs('inference/assim/', exist_ok = True)
                    with open('inference/assim/{}'.format(filename), 'wb') as f:
                        f.write(compr)

                    print({
                        # 'statusCode': 200,
                        'file': filename.split('.')[0],
                        # 'body': s/1e6, #os.path.getsize(outfile), #json.dumps('Hello from Lambda!'),
                        'outlen': len(out),#len(pickle.dumps(data)),
                        'outlen-compr': len(compr),#zc.compress(pickle.dumps(data))),
                        'elaspsed_time': round(time.time() - start, 1)
                    }); break;
                except Exception as e:
                    print(e); time.sleep(i)


def listAssimDates(dates):
    months = {}
    for t in dates:
        k = (t.year, t.month)
        prior = months.get(k, [])
        if sum(prior) > 0:
            months[k] = (min(prior[0], t.day), max(prior[0], t.day))
        else:
            months[k] = (t.day, t.day)
    return [(*k, *v) for k, v in months.items()]














start = datetime.datetime(*[int(i) for i in infer['start'].split(',')])
end = datetime.datetime(*[int(i) for i in infer['end'].split(',')])


dt = start - datetime.timedelta(days = 10)
dates = []
while dt <= end + datetime.timedelta(days = 1):
    dates.append(dt);
    dt += datetime.timedelta(days = 1)
print(len(dates))
print(dates[0]); print(dates[-1])





def listGFSFiles(dates):
    filelist = []; fwd = 0
    for t in dates:
        dt = t.strftime('%Y%m%d')
        for hr in [0, 6, 12, 18]:
            filelist.append('{}/{}/gfs.0p25.{}{:02d}.f{:03d}.grib2'.format(
                    dt[:4], dt, dt, hr, fwd))
    return filelist


def listIFSFiles(dates):
    filelist = []
    for t in dates:
        for tag in ifs_tags:
            domain = 'ec.oper.fc.sfc'
            file =  '{}/{}/{}.{}.regn1280sc.{}.nc'.format(domain,
                            datetime.datetime.strftime(t, '%Y%m'), 
                            domain, tag, 
                            datetime.datetime.strftime(t, '%Y%m%d') )
            filelist.append(file)
    return filelist


def listTropomiRows(dates):
    tropomi_rows = [e.to_dict() for idx, e in 
         files[files['product'].str.startswith('tropomi')
               & (files.time_end.dt.tz_localize(None) >= min(dates) )
               & (files.time_end.dt.tz_localize(None) 
                      <= max(dates) + datetime.timedelta(days = 1) )
              ].iterrows()]
    return tropomi_rows











# %%time
N_THREADS = min(10, os.cpu_count() )
Parallel(N_THREADS)(delayed(pullIFS)(
    listIFSFiles(dates)[i::N_THREADS]) 
                        for i in range(N_THREADS)) 


# %%time
N_THREADS = min(4, os.cpu_count() )
Parallel(N_THREADS)(delayed(pullGFS)(
    listGFSFiles(dates)[:][i::N_THREADS]) 
                        for i in range(N_THREADS))


# %%time
N_THREADS = min(5, os.cpu_count()) 
Parallel(N_THREADS)(delayed(pullTropomi)(row) 
                    for row in listTropomiRows(dates))


# %%time
[pullAssim(*d) for d in listAssimDates(dates)]






if start.year <= 2018 and end.year >= 2021:
    os.makedirs('cache', exist_ok = True)
    for path in os.listdir('inference'):
        with tarfile.open('cache/{}.tar'.format(path), 'w') as f:
            for file in os.listdir('inference/{}'.format(path)):
                f.add('inference/{}/{}'.format(path, file), 
                          arcname = file)





# !jupyter nbconvert --no-prompt --to script 'RunFeatures.ipynb' 

