"""
Downloads 1 day NASA GEOS-CF NO2 hindcasts for the time range in the labels file.
"""
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,argparse,time,shutil
import numpy as np, pandas as pd, geopandas as gpd
import xarray as xr
from tqdm.auto import tqdm
from pqdm.processes import pqdm
import multiprocessing as mp
import pydap
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--stage", help="Stage of data to process, i.e train,test,prod", required=True) 
parser.add_argument("--labels", help="Path to labels.csv file", required=True)
parser.add_argument("--n_jobs",type=int, help="Number of processes. Can use more than available cpus to speed up download since the program is I/O bound. Defaults to 2*number of processors" ) #
parser.add_argument("--cleanup", help="Delete downloaded files after extracting useful data. Leave default to only download missing files in case of failure", action="store_true") ##

args = parser.parse_args()
DATA_DIR = args.data_dir 
STAGE = args.stage 
LABELS_PATH = args.labels 
DO_CLEANUP = True if args.cleanup else False

# DATA_DIR = 'data'
# LABELS_PATH = 'data/raw/train_labels.csv'
# DO_CLEANUP=False

DATA_DIR = Path(DATA_DIR) 

DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_INTERIM = DATA_DIR / "interim"
SAVE_DIR = DATA_DIR_INTERIM/"geos" #directory to save final output

TMP_DIR = DATA_DIR_INTERIM/'tmp/geos'/STAGE #directory to temporarily save downloaded files
N_JOBS = mp.cpu_count()*2
if args.n_jobs:
    N_JOBS = args.n_jobs

print(f'using {N_JOBS} processes')

df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})
LOCATIONS = sorted(df_grid_meta.location.unique())
df_labels = pd.read_csv(LABELS_PATH)
df_labels = pd.merge(df_labels,df_grid_meta[['grid_id','location']]).sort_values(by=['datetime','grid_id']).reset_index(drop=True)
df_labels['obs_datetime_start']=pd.to_datetime(df_labels.datetime).dt.tz_localize(None)


for d in [DATA_DIR_INTERIM,TMP_DIR,SAVE_DIR]:
    os.makedirs(d,exist_ok=True)


BASEURL = 'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/aqc_tavg_1hr_g1440x721_v1'
VARIABLE = 'no2'


#for each grid, find the nearest coordinates in the 0.25 degree resolution data
lons_data=[]
lats_data=[]

DATASET = xr.open_dataset(BASEURL)
for _,row in tqdm(df_grid_meta.iterrows(),total=len(df_grid_meta)):
    geom=row.geometry
    x,y = geom.centroid.xy[0][0],geom.centroid.xy[1][0]
    r=0.25/2
    minlon = x-r
    maxlon = x+r
    minlat = y-r
    maxlat = y+r
    
    DS = DATASET[VARIABLE].loc[{'lat':slice(minlat,maxlat),'lon':slice(minlon,maxlon)}]
    assert(DS.lat.data.shape[0]==1)
    assert(DS.lon.data.shape[0]==1)
    
    lons_data.append(DS.lon.data[0])
    lats_data.append(DS.lat.data[0])    
  
    
DATASET.close()
df_grid_meta['lon'] = lons_data
df_grid_meta['lat'] = lats_data

def download_data_for_loc_date(DATASET,minlon,minlat,maxlon,maxlat,start_time,end_time,save_to):
    """
        Extract and save no2 hindcasts for give coordinates and time range
    """
    success = False
    res = None
    fail_ctr = 0
    
    try:
        DS = DATASET[VARIABLE].loc[{'time':slice(start_time,end_time),'lat':slice(minlat,maxlat),'lon':slice(minlon,maxlon)}]
        DS.load()
        res = DS.squeeze('lev').drop_vars('lev')
        success=True
        res.to_netcdf(save_to)
      
    except RuntimeError:
        DATASET.close()
        fail_ctr+=1
        sleep_time = fail_ctr*10
        
        print(f'RuntimeError at {start_time}, ctr {fail_ctr} ')
        time.sleep(sleep_time)
        DATASET = xr.open_dataset(BASEURL)
        return download_data_for_loc_date(DATASET,minlon,minlat,maxlon,maxlat,start_time,end_time,save_to)
    
    return success

for location in LOCATIONS:
    print(f'###### Extracting data for location {location} ######')
    R = 0.5 #add padding for location bounds 
    d = df_grid_meta[df_grid_meta.location==location]

    minlon,minlat,maxlon,maxlat = d.geometry.total_bounds
    minlon = minlon-R
    minlat = minlat-R
    maxlon = maxlon+R
    maxlat = maxlat+R
    
    dates = df_labels[df_labels.location==location].obs_datetime_start.tolist()
    dates = sorted(list(set(dates)))
 
    if len(dates)>0:
        ##pad with 2 days before,after
        dates = [dates[0] - pd.DateOffset(days=2),dates[0] - pd.DateOffset(days=1)] + dates
        dates = dates + [dates[-1] + pd.DateOffset(days=1),dates[-1] + pd.DateOffset(days=2)]
    
    ##download no2 predictions for each date
    args = []
    for start_time in dates:
        save_to = f"{TMP_DIR}/{start_time.date()}_{location}.nc"
        if not os.path.exists(save_to):
            end_time = start_time + pd.DateOffset(days=1)
            args.append((DATASET,minlon,minlat,maxlon,maxlat,start_time,end_time,save_to))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = pqdm(args, download_data_for_loc_date, n_jobs=N_JOBS,argument_type='args')

paths = glob.glob(f'{TMP_DIR}/*.nc')
dfs = []
for p in tqdm(paths):
    d = xr.open_dataset(p).to_dataframe()
    dfs.append(d)

if DO_CLEANUP:
    shutil.rmtree(TMP_DIR,ignore_errors=True)

df_data = pd.concat(dfs).reset_index()
print(df_data.shape)
df_data = df_data.drop_duplicates(subset=['lon','lat','time'])
print(df_data.shape)
df_data = pd.merge(df_grid_meta,df_data,on=['lon','lat']).sort_values(by=['time','grid_id']).reset_index(drop=True)[['grid_id','lon','lat','time','no2']]

path = SAVE_DIR/ f'geos_{STAGE}.csv'
df_data.to_csv(path,index=False)
print(f'Saved data to {path}')
