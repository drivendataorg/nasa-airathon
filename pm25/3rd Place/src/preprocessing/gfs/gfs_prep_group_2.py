import cfgrib
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from tqdm import tqdm, tnrange
import pandas as pd
import math
import shapely.wkt
from scipy.stats import mode

def get_csv_from_grib_files(grib_files: list) -> pd.DataFrame:
    """_summary_

    Args:
        grib_files (list): _description_

    Returns:
        pd.DataFrame: _description_
    """

    res = []
    for file in tqdm(grib_files):
        temp = xr.open_dataset(str(file), engine="cfgrib")
        res.append(temp.to_dataframe().reset_index())

    df = pd.concat(res)
    return df

def getBounds(shape):
    x1 = []
    y1 = []
      
    x = shape.exterior.coords.xy[0]
    y = shape.exterior.coords.xy[1]
    x1.append(min(x))
    x1.append(max(x))
    y1.append(min(y))
    y1.append(max(y))

    return x1,y1

def get_mask(df, min_lon, max_lon, min_lat, max_lat):
    lon_mask = (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
    lat_mask = (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
    return (lon_mask) & (lat_mask)


def prep_group_2(root, read_path, save_path, loc, loc_keys, grid_meta, var_group):

    # Defining the path to raw grib files
    path = path = Path(read_path)
    # folders = [i for i in path.iterdir() if i.is_dir()] 
    # folder = list(filter(lambda x: var_group in x.name, folders))[0]

    # Getting grib file names 
    grib_files = list(path.glob('*.grib2'))

    # Converting each file into csv and concatenating
    df = get_csv_from_grib_files(grib_files)
    print(f'finished getting csv from grib files for {loc} using files at path {path} for group {var_group}')
    df.rename(columns={'unknown':'cat_rain'}, inplace=True)

    df['time_date'] = df['time'].dt.date 
    df = df.groupby(['latitude', 'longitude', 'time_date']).agg(prate_max=pd.NamedAgg(column="prate", aggfunc="mean"), 
    prate_mean=pd.NamedAgg(column="prate", aggfunc="mean"), cat_rain_max=pd.NamedAgg(column="cat_rain", aggfunc="max"),
    cat_rain_mode=pd.NamedAgg(column="cat_rain", aggfunc=mode)).reset_index()
    df['cat_rain_mode'] = df['cat_rain_mode'].apply(lambda x: x[0][0]) 

    grid_meta = grid_meta[grid_meta['location'] == loc_keys[loc]] 

    res = []
    for i in  tqdm(range(grid_meta.shape[0])):
        grid_id = grid_meta.iloc[i].grid_id
        grid_shape = shapely.wkt.loads(grid_meta.iloc[i]['wkt'])

        lon, lat = getBounds(grid_shape)
        lon[0] = lon[0] if lon[0] > 0 else lon[0] + 360 # converting neg lon to pos
        lon[1] = lon[1] if lon[1] > 0 else lon[1] + 360

        min_lon, max_lon = round(lon[0]*4)/4, math.ceil(lon[1]*4)/4
        min_lat, max_lat = round(lat[0]*4)/4, math.ceil(lat[1]*4)/4

        mask = get_mask(df, min_lon, max_lon, min_lat, max_lat)
        
        agg_df = df.groupby('time_date').agg({'prate_max': 'max', 'prate_mean': 'mean', 'cat_rain_max': 'max', 'cat_rain_mode': mode})
        agg_df['cat_rain_mode'] = agg_df['cat_rain_mode'].apply(lambda x: x[0][0]) 
        agg_df['grid_id'] = grid_id
        agg_df['location'] = loc

        res.append(agg_df)

    pd.concat(res).reset_index().to_csv(f'{save_path}/{loc}_gfs_{var_group}.csv', index=None)


if __name__ == "__main__":
    
    root = '../../data/gfs'
    read_path = f'{root}/downloaded_files'
    save_path = f'{root}/merged_csv'
    loc = 'la'
    loc_keys = {'la':'Los Angeles (SoCAB)', 'tp':'Taipei', 'dl':'Delhi'}
    grid_meta = pd.read_csv('../data/grid_metadata.csv')
    var_group = 'group_2'
    
    prep_group_2(root, read_path, save_path, loc, loc_keys, grid_meta, var_group)

    