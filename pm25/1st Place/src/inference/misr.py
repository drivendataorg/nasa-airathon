import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict
import netCDF4

def format_geometry(geometry):
    geometry = geometry.replace('(', '', -1)
    geometry = geometry.replace(')', '', -1)
    geometry = geometry.replace(',', '', -1)
    geometry = list(map(float, geometry.split()[1:]))
    geometry = [geometry[i:i+2] for i in range(0, len(geometry), 2)]
    return geometry

def fillna(data, col):
    temp = data[col]
    fillvalue = temp._FillValue
    temp = np.array(temp)
    temp[temp == fillvalue] = np.nan
    return temp

def floor(x, n=0):
    return np.floor(x * 10**n) / 10**n

def ceil(x, n=0):
    return np.ceil(x * 10**n) / 10**n

def get_bounds(geometry):
    geometry = np.array(geometry)
    long = [
        floor(geometry[:, 0].min(), 1),
        ceil(geometry[:, 0].max(), 1)
    ]
    lat = [
        floor(geometry[:, 1].min(), 1),
        ceil(geometry[:, 1].max(), 1)
    ]
    return long, lat

def mask(config, data, geometry):
    assets = {}
    for key in data.keys():
        assets[key] = data[key].ravel()
    longb, latb = get_bounds(geometry)
    latitude = assets['Latitude']
    longitude = assets['Longitude']
    indices = (latitude >= latb[0]) & (latitude <= latb[1]) & (longitude >= longb[0]) & (longitude <= longb[1])

    new_ass = {}
    for k in config.MISR_BANDS:
        new_ass[k] = assets[k][indices]
                                                                   
    df = pd.DataFrame(new_ass)

    return df

def get_misr_data(config, sat_data, grid_metadata) -> pd.DataFrame:
    obs_end_dt = config.OBS_START_TIME + datetime.timedelta(1, 0)
    location = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['location'].values[0]
    cur_satdata = sat_data[sat_data['product'] == 'misr']
    cur_satdata = cur_satdata[cur_satdata['location'] == config.LOCATION_MAP[location]]
    geometry = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['wkt'].values[0]
    geometry = format_geometry(geometry)
    features = defaultdict(lambda: [])

    for dt in [config.OBS_START_TIME, obs_end_dt]:
        possible = cur_satdata[cur_satdata['time_end'] < dt].sort_values('time_end', ascending=False).reset_index()
        # print(possible.head())
        if len(possible) == 0:
            for key in config.MISR_BANDS:
                features[key + '_mean'].append(np.nan)
                features[key + '_var'].append(np.nan)
            continue
        url = possible.iloc[0]['us_url']
        filename = os.path.basename(url)
        if not os.path.exists(filename):
            os.system(f"aws s3 cp {url} ./ --no-sign-request")
        data = netCDF4.Dataset(filename, mode='r')
        data = data.groups['4.4_KM_PRODUCTS'].variables
        assets = {}
        for band in config.MISR_BANDS:
            assets[band] = fillna(data, band)
        assets = mask(config, assets, geometry)
        for key in config.MISR_BANDS:
            _band = assets[key].values
            _band = np.concatenate((
                _band[_band <= 0], _band[_band > 0]
            ))
            features[f"{key}_mean"].append(_band.mean())
            features[f"{key}_var"].append(_band.std() ** 2)
    features = pd.DataFrame(features)
    features = features.drop(columns=config.MISR_DROP)
    return features       