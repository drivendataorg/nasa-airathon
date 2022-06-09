"""
    Downloads TROPOMI data from an s3 bucket and extracts relevant features
"""
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,multiprocessing,argparse
import numpy as np, pandas as pd, geopandas as gpd
import multiprocessing as mp
from tqdm.auto import tqdm
from pqdm.processes import pqdm
from netCDF4 import Dataset
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--stage", help="Stage of data to process, i.e. train,test,prod", required=True) ##
parser.add_argument("--path_satmeta", help="Path to satellite metadata csv file",required=True ) ##
parser.add_argument("--s3url", help="Metadata column corresponding to the file location in s3", default='us_url' ) #
parser.add_argument("--n_jobs",type=int, help="Number of processes. Can use more than available cpus to speed up download since the program is I/O bound. Defaults to 2*number of processors" ) #
parser.add_argument("--cleanup", help="Delete downloaded files after extracting useful data",  default=False, action="store_true") ##

args = parser.parse_args()
DATA_DIR = args.data_dir 
PATH_SATMETA = args.path_satmeta 
STAGE = args.stage 
DO_CLEANUP = True if args.cleanup else False


DATA_DIR = Path(DATA_DIR) 

DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_INTERIM = DATA_DIR / "interim"
SAVE_DIR = DATA_DIR_INTERIM/"tropomi"

TMP_DIR = DATA_DIR_INTERIM/'tmp/tropomi'/STAGE #to temporarily save downloaded files
N_JOBS = mp.cpu_count()*2
if args.n_jobs:
    N_JOBS = args.n_jobs

print(f'using {N_JOBS} processes')


df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})

df_satellite_meta = pd.read_csv(PATH_SATMETA)
df_satellite_meta = df_satellite_meta[(df_satellite_meta['product'].isin(['tropomi','tropomi_hires'])) & (df_satellite_meta.split==STAGE)]
df_satellite_meta['s3url'] = df_satellite_meta[args.s3url]

for d in [DATA_DIR_INTERIM,TMP_DIR,SAVE_DIR]:
    os.makedirs(d,exist_ok=True)


VARIABLES = ['latitude','longitude',  'qa_value', 'nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision',
             'nitrogendioxide_tropospheric_column_precision_kernel', 'air_mass_factor_troposphere', 'air_mass_factor_total', 'tm5_tropopause_layer_index']



def extract_data(s3_path,location):
    granule_id = s3_path.split('/')[-1]
    local_path = TMP_DIR/granule_id
    # assert(location in granule_id)
    utils.download_s3_file(s3_path,local_path)
    #read file
    nc = Dataset(local_path) 
    
    #list to store data for all variablels
    data = []
    for variable in VARIABLES:
        #d = nc.groups['PRODUCT'].variables[variable][0,...].data
        v = nc.groups['PRODUCT'].variables[variable]
        v.set_auto_maskandscale(False)
        d = v[:].astype(np.float64)
        d[d==v._FillValue] = np.nan
        assert(d.shape[0]==1)
        d=d[0].astype(np.float32)
        data.append(d)


    lat=nc.groups['PRODUCT'].variables['latitude'][0,...]
    lon=nc.groups['PRODUCT'].variables['longitude'][0,...]
    
    data = np.array(data)
    nc.close() #close file
    if DO_CLEANUP:
        os.remove(local_path) # delete local file
    
    #init list to store dfs with data extracted for each grid
    ds = [] 
    for _,row in df_grid_meta[df_grid_meta.location==location].iterrows():
        bounds=row.geometry.bounds
        OFF=0.0 #bounds offset to use
        eps=0.001 #epsilon to increment bounds offset
        while True:
            ixs=np.where((lon>=bounds[0]-OFF) & (lon<=bounds[2]+OFF) & (lat>=bounds[1]-OFF) & (lat<=bounds[3]+OFF)) 
            if len(ixs[0])==0:
                OFF+=eps
            else:
                break    

        d=data[:,ixs[0],ixs[1]].transpose(1,0)
        d=pd.DataFrame(d)
        d['grid_id'] = row.grid_id
        d['granule_id'] = granule_id
        ds.append(d)
    
    df = pd.concat(ds)
    
    return df
  

args = list(zip(df_satellite_meta.s3url.tolist(),df_satellite_meta.location.tolist()))

results = pqdm(args, extract_data, n_jobs=N_JOBS,argument_type='args')
df_data=pd.concat(results).reset_index(drop=True)
df_data = df_data.rename(columns={i:VARIABLES[i] for i in range(len(VARIABLES))})

df_sat_meta = df_satellite_meta[['granule_id','time_start','time_end']].copy()
df_sat_meta['time_start'] = pd.to_datetime(df_sat_meta.time_start).dt.tz_localize(None)
df_sat_meta['time_end'] = pd.to_datetime(df_sat_meta.time_end).dt.tz_localize(None)
df_data = pd.merge(df_sat_meta,df_data).rename(columns={'time_start':'file_datetime_start','time_end':'file_datetime_end'})

path = SAVE_DIR/ f'tropomi_{STAGE}.csv'
df_data.to_csv(path,index=False)
print(f'Saved data to {path}')