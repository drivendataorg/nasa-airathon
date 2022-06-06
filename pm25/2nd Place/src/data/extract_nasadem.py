"""
    Extracts elevaton features for each grid from NASADEM
"""
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,argparse,time,shutil
import numpy as np, pandas as pd, geopandas as gpd
import xarray as xr
from tqdm.auto import tqdm
import planetary_computer as pc
from pystac_client import Client
import scipy.stats as scistats 

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)

args = parser.parse_args()
DATA_DIR = args.data_dir 

DATA_DIR = Path(DATA_DIR) 

DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_INTERIM = DATA_DIR / "interim"
SAVE_DIR = DATA_DIR_INTERIM #directory to save final output


df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
# df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})

for d in [DATA_DIR_INTERIM,SAVE_DIR]:
    os.makedirs(d,exist_ok=True)


mins=[]
maxs=[]
means=[]
medians=[]
stds = []
skews = []
kurts = []

URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'

for _,row in tqdm(df_grid_meta.iterrows(),total=len(df_grid_meta)):
    geom=row.geometry
    x,y = geom.centroid.xy[0][0],geom.centroid.xy[1][0]
    area_of_interest = {"type": "Point", "coordinates": [x,y]}
    catalog = Client.open(URL)
    nasadem = catalog.search(collections=["nasadem"], intersects=area_of_interest)

    items = [item for item in nasadem.get_items()]

    #print(f"{row.grid_id} #{len(items)} Items")
    item = items[0]
    signed_asset = pc.sign(item.assets["elevation"])

    da = (xr.open_rasterio(signed_asset.href))
    xs = da.x.data
    ys=da.y.data
    assert(ys[0]>ys[1])#lats ordered descending
    assert(xs[1]>xs[0])#lons ordered ascending

    minlon = geom.bounds[0]
    maxlon = geom.bounds[2]
    minlat = geom.bounds[1]
    maxlat = geom.bounds[3]

    da = da.loc[{'y':slice(maxlat,minlat),'x':slice(minlon,maxlon)}]

    d=da.data
    mins.append(d.min())
    maxs.append(d.max())
    means.append(np.mean(d))
    medians.append(np.median(d))
    stds.append(np.std(d))
    skews.append(scistats.skew(d.flatten()))
    kurts.append(scistats.kurtosis(d.flatten()))
    


df = df_grid_meta[['grid_id','location']].copy()
df['elev_mean'] = means
df['elev_median'] = medians
df['elev_min'] = mins
df['elev_max'] = maxs
df['elev_std'] = stds
df['elev_skew'] = skews
df['elev_kurt'] = kurts

path = SAVE_DIR/ f'elevation.csv'
df.to_csv(path,index=False)
print(f'Saved data to {path}')
