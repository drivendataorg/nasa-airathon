import pandas as pd
import numpy as np
import datetime
import os
import pygrib
from collections import defaultdict
import requests

def format_geometry(geometry):
    geometry = geometry.replace('(', '', -1)
    geometry = geometry.replace(')', '', -1)
    geometry = geometry.replace(',', '', -1)
    geometry = list(map(float, geometry.split()[1:]))
    geometry = [geometry[i:i+2] for i in range(0, len(geometry), 2)]
    return geometry

def get_nearest_cycle(dt):
    hour = dt.hour
    ans = None
    for h in [0, 6, 12, 18]:
        if hour >= h:
            ans = h
    return ans
    
def get_nearest_forecast(dt):
    cycle = get_nearest_cycle(dt)
    available_forecasts = [0, 3, 6, 9, 12, 15, 18, 21]
    forecast = 0
    for f in available_forecasts:
        if cycle + f > dt.hour:
            break
        forecast = f
    return "{:02d}".format(cycle), "{:03d}".format(forecast)

def round_off(point, res):
    spread = np.arange(np.floor(point), np.ceil(point) + 1, res)
    adiff = np.abs(spread - point)
    return spread[np.argmin(adiff)]

def get_boundary(geometry, res=0.25):
    long = np.array(geometry)[:, 0]
    lat = np.array(geometry)[:, 1]
    
    min_lat, max_lat = lat.min(), lat.max()
    min_long, max_long = long.min(), long.max()
    
    min_lat = round_off(min_lat - res / 2, res)
    max_lat = round_off(max_lat + res / 2, res)
    min_long = round_off(min_long - res / 2, res)
    max_long = round_off(max_long + res / 2, res)
    
    return [min_long, max_long], [min_lat, max_lat]

def get_gfs_data(config, grid_metadata) -> pd.DataFrame:
    email = config.NCAR_EMAIL
    pswd = config.NCAR_PSWD
    url = 'https://rda.ucar.edu/cgi-bin/login'
    values = {'email' : email, 'passwd' : pswd, 'action' : 'login'}
    # Authenticate
    ret = requests.post(url,data=values)
    if ret.status_code != 200:
        print('Bad Authentication for NCAR')
        print(ret.text)
        exit(1)
    DSPATH = 'https://rda.ucar.edu/data/ds084.1/'

    def check_file_status(filepath, filesize):
        import sys
        sys.stdout.write('\r')
        sys.stdout.flush()
        size = int(os.stat(filepath).st_size)
        percent_complete = (size/filesize)*100
        sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
        sys.stdout.flush()

    features = defaultdict(lambda: [])
    for dt in [config.OBS_START_TIME - datetime.timedelta(1, 0), config.OBS_START_TIME]:
        cycle, forecast = get_nearest_forecast(dt)
        year = dt.year
        month = "{:02d}".format(dt.month)
        day = "{:02d}".format(dt.day)
        filename = f'{year}/{year}{month}{day}/gfs.0p25.{year}{month}{day}{cycle}.f{forecast}.grib2'
        # print(f"{dt} => {filename}")

        filename = DSPATH + filename
        file_base = os.path.basename(filename)

        if not os.path.exists(file_base):
            req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
            filesize = int(req.headers['Content-length'])

            with open(file_base, 'wb') as outfile:
                chunk_size=1048576
                for chunk in req.iter_content(chunk_size=chunk_size):
                    outfile.write(chunk)
                    if chunk_size < filesize:
                        check_file_status(file_base, filesize)
            check_file_status(file_base, filesize)

        gr = pygrib.open(file_base)
        assets = {}
        for g in gr:
            if g.name in config.GFS_BANDS and g.typeOfLevel == 'surface':
                assets[g.name] = np.array(g.values).ravel()
        
        latitude = np.array(g.latlons()[0]).ravel()
        longitude = np.array(g.latlons()[1]).ravel()
        geometry = grid_metadata[grid_metadata['grid_id'] == config.GRID_ID]['wkt'].values[0]
        geometry = format_geometry(geometry)
        longb, latb = get_boundary(geometry)
        cur_indices = (latitude >= latb[0]) & (latitude <= latb[1]) & (longitude >= longb[0]) & (longitude <= longb[1])
        assets = {k: v[cur_indices] for k, v in assets.items()}

        for key in config.GFS_BANDS:
            try:
                _band = assets[key]
                _band = np.concatenate((
                    _band[_band <= 0], _band[_band > 0]
                ))
            except KeyError:
                _band = np.array([np.nan, np.nan])
            features[f"{key}_mean"].append(_band.mean())
            features[f"{key}_var"].append(_band.std() ** 2)
    features = pd.DataFrame(features)
    return features