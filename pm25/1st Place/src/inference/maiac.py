import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import rioxarray as rxr
import os
from rioxarray.exceptions import NoDataInBounds

def format_geometry(geometry):
    geometry = geometry.replace('(', '', -1)
    geometry = geometry.replace(')', '', -1)
    geometry = geometry.replace(',', '', -1)
    geometry = list(map(float, geometry.split()[1:]))
    geometry = [geometry[i:i+2] for i in range(0, len(geometry), 2)]
    return geometry

def get_maiac_data(config, sat_data, grid_metadata) -> pd.DataFrame:
    # Download relevant maiac data from s3 bucket and return features.
    obs_end_dt = config.OBS_START_TIME + datetime.timedelta(1, 0)
    location = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['location'].values[0]
    cur_satdata = sat_data[sat_data['product'] == 'maiac']
    cur_satdata = cur_satdata[cur_satdata['location'] == config.LOCATION_MAP[location]]
    geometry = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['wkt'].values[0]
    geometries = [
        {
            'type': 'Polygon',
            'coordinates': [format_geometry(geometry)]
        }
    ]
    features = defaultdict(lambda: [])

    for dt in [config.OBS_START_TIME, obs_end_dt]:
        possible = cur_satdata[cur_satdata['time_end'] < dt].sort_values('time_end', ascending=False).reset_index()
        if len(possible) == 0:
            for key in config.MAIAC_BANDS:
                features[key + '_mean'].append(np.nan)
                features[key + '_var'].append(np.nan)
            continue
        for k in range(len(possible)):
            if dt - possible.iloc[k]['time_end'] > datetime.timedelta(1, 0):
                continue
            url = possible.iloc[k]['us_url']
            filename = os.path.basename(url)
            if not os.path.exists(filename):
                os.system(f"aws s3 cp {url} ./ --no-sign-request")
            data = rxr.open_rasterio(filename, masked=True)
        
            try:
                clipped = data[0].rio.clip(geometries, crs=4326)
            except NoDataInBounds:
                continue
            
            assets = {}
            for key in config.MAIAC_BANDS:
                assets[key] = np.array(clipped[key].as_numpy())
            for key in config.MAIAC_BANDS:
                _band = np.array(assets[key]).ravel()
                _band = np.concatenate((
                    _band[_band >= 0], _band[_band < 0]
                )) # removing nan values
                mean, var = _band.mean(), _band.std() ** 2
                features[key + '_mean'].append(mean)
                features[key + '_var'].append(var)
            break
        else:
            print(f"No MAIAC data found within 24 hours prior to given datetime. Using NaNs instead.")
            for key in config.MAIAC_BANDS:
                features[key + '_mean'].append(np.nan)
                features[key + '_var'].append(np.nan)

    return pd.DataFrame(features)
