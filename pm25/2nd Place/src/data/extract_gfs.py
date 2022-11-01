"""
  Downloads 3-hour forecasts of various meteorological components from the NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive
  hosted on NCAR Servers (https://rda.ucar.edu/datasets/ds084.1/index.html), for the time range in the labels file.
  Request parameters have been grouped to minimize the number of requests sent to the server.
  Requests are sent separately for each location, for each parameter group.
  
"""
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,argparse,time,shutil,copy,tarfile
import numpy as np, pandas as pd, geopandas as gpd
import xarray as xr
from collections import defaultdict
from tqdm.auto import tqdm
from pqdm.processes import pqdm
import multiprocessing as mp

import rdams_client as rc


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--stage", help="Stage of data to process, i.e train,test,prod", required=True) ##
parser.add_argument("--labels", help="Path to labels.csv file. Data extracted ranges from the first to the last observation date in the label file.", required=True)
parser.add_argument("--n_jobs",type=int, help="Number of processes. Can use more than available cpus to speed up download since the program is I/O bound. Defaults to 2*number of processors" ) #
parser.add_argument("--cleanup", help="Delete downloaded files after extracting useful data. Leave default to only download missing files in case of failure", action="store_true") ##

args = parser.parse_args()
DATA_DIR = args.data_dir 
STAGE = args.stage 
LABELS_PATH = args.labels 
DO_CLEANUP = True if args.cleanup else False
N_JOBS = mp.cpu_count()*2
if args.n_jobs:
    N_JOBS = args.n_jobs
    

MAX_CONCURRENT_REQUESTS = 8 #Number of maximum requests sent at a time. Leave default to prevent exceeding NCAR server quota of 10 concurrent requests
DATA_DIR = Path(DATA_DIR) 

DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_INTERIM = DATA_DIR / "interim"
SAVE_DIR = DATA_DIR_INTERIM/"gfs" #directory to save final output

TMP_DIR = DATA_DIR_INTERIM/'tmp/gfs'/STAGE #directory to temporarily save downloaded files


df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})
df_labels = pd.read_csv(LABELS_PATH)
df_labels = pd.merge(df_labels,df_grid_meta[['grid_id','location']]).sort_values(by=['datetime','grid_id']).reset_index(drop=True)
df_labels['obs_datetime_start']=pd.to_datetime(df_labels.datetime).dt.tz_localize(None)
LOCATIONS = sorted(df_labels.location.unique())

# for d in [DATA_DIR_INTERIM,TMP_DIR,SAVE_DIR]:
#     os.makedirs(d,exist_ok=True)

dsid = 'ds084.1' #NCEP GFS 0.25 Degree Global Forecast dataset
PRODUCT = '3-hour Forecast'
result = rc.query(['-get_summary', dsid])



def check_ready(rqst_id, wait_interval=120):
    """Checks if a request is ready."""
    for i in range(100): 
        res = rc.get_status(rqst_id)
        request_status = res['result']['status']
        
        if request_status == 'Completed':
            return True
            
        if request_status=='Error':
          print('Request failed')
          return False

        print(f'Request {request_status}. Waiting {wait_interval} seconds for request {rqst_id}' )
        time.sleep(wait_interval)

    return False

def download_data(request):
    template = request['template']
    out_dir = request['out_dir']
    filepath = request['filepath']
    ret = {'param':template['param'], 'location':request['location'],'success':False}
    #submit request
    response = rc.submit_json(template)
    try: 
        rqst_id = response['result']['request_id']
        #wait until it's processed
        if check_ready(rqst_id):
            os.makedirs(out_dir,exist_ok=True)
            dl_response = rc.download(rqst_id,out_dir)
            dl_fname = dl_response['result']['web_files'][-1]['web_path'].split('/')[-1]
            shutil.move(f'{out_dir}/{dl_fname}',filepath)
            ret['success']=True
        
        rc.purge_request(rqst_id)
    except:
            ret['failed_response'] = response
            print(f'Request failed: {response}')
            if response['messages']==['User has more than 10 open requests. Purge requests before trying again.']:
                print('Queuieng request')
                time.sleep(60*2)
                return download_data(request)
    return ret


param_group0 = [
                # Precipitable water/ atmosphere_water_vapor_content (kg m^-2)
                {'param':'P WAT','level':'EATM:0','typeOfLevel':'unknown','orig_name':'pwat','target_name':'pwat','pivot':False},
                #Total ozone (Dobson)
                {'param':'TOZNE','level':'EATM:0','typeOfLevel':'unknown','orig_name':'tozne','target_name':'tozne','pivot':False},
                # Dewpoint temperature (K)
                {'param':'DPT','level':'HTGL:2','typeOfLevel':'heightAboveGround','orig_name':'d2m','target_name':'dpt','pivot':False},
                # Apparent temperature (K)
                {'param':'APTMP','level':'HTGL:2','typeOfLevel':'heightAboveGround','orig_name':'aptmp','target_name':'aptmp','pivot':False},
                #Potential temperature (K)
                {'param':'POT','level':'SIGL:0.995','typeOfLevel':'sigma','orig_name':'pt','target_name':'pot','pivot':False},  
                #Wind speed/ gust (m s^-1)
                {'param':'GUST','level':'SFC:0','typeOfLevel':'surface','orig_name':'gust','target_name':'gust','pivot':False},
                # Air Pressure (Pa)
                {'param':'PRES','level':'SFC:0','typeOfLevel':'surface','orig_name':'sp','target_name':'pres','pivot':False},
                #Pressure reduced to MSL (Pa)
                {'param':'PRMSL','level':'MSL:0','typeOfLevel':'meanSea','orig_name':'prmsl','target_name':'prmsl','pivot':False},
                #Potential water evaporation rate (W m^-2)
                {'param':'PEVPR','level':'SFC:0','typeOfLevel':'surface','orig_name':'unknown','target_name':'pevpr','pivot':False},
]

isobar_levels = 'ISBL:1/5/10/20/50/100/150/200/250/300/350/400/450/500/550/600/650/700/750/800/850/900/925/950/975/1000'
param_group1 = [
                #Geopotential height
                {'param':'HGT','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'gh','target_name':'hgt','pivot':True},
                #Air temperature
                {'param':'TMP','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'t','target_name':'tmp','pivot':True},
                #Relative humidity
                {'param':'R H','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'r','target_name':'rh','pivot':True},
                #'u-component of wind/ eastward wind
                {'param':'U GRD','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'u','target_name':'ugrd','pivot':True},
                #v-component of wind/ northward wind
                {'param':'V GRD','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'v','target_name':'vgrd','pivot':True},
]

param_group2 = [
                #Vertical wind speed shear (s^-1)
                {'param':'VW SH','level':'PVL:2000/-2000','typeOfLevel':'potentialVorticity','orig_name':'unknown','target_name':'vwsh','pivot':True},
                #Convective available potential energy (J kg^-1)
                {'param':'CAPE','level':'SPDL:0,255/0,90/0,180','typeOfLevel':'pressureFromGroundLayer','orig_name':'unknown','target_name':'cape','pivot':True},
                #Volumetric soil moisture content
                {'param':'SOILW','level':'DBLL:2,1/1,0.4/0.1,0/0.4,0.1','typeOfLevel':'depthBelowLandLayer','orig_name':'unknown','target_name':'soilw','pivot':True},
]

param_group3 = [
                #Soil temperature (K)
                {'param':'TSOIL','level':'DBLL:2,1/1,0.4/0.1,0/0.4,0.1','typeOfLevel':'depthBelowLandLayer','orig_name':'st','target_name':'tsoil','pivot':True},
                #Convective inhibition (J kg^-1)
                {'param':'CIN','level':'SPDL:0,255/0,90/0,180', 'typeOfLevel':'pressureFromGroundLayer','orig_name':'unknown','target_name':'cin','pivot':True},
                #Cloud water mixing ratio (kg kg^-1)
                {'param':'CLWMR','level':'ISBL:100/150/200/250/300/350/400/450/500/550/600/650/700/750/800/850/900/925/950/975/1000',
                  'typeOfLevel':'isobaricInhPa','orig_name':'clwmr','target_name':'clwmr','pivot':True}
]

param_group4 = [
                #Ventilation rate
                {'param':'VRATE','level':'PBLRI:0','typeOfLevel':'unknown','orig_name':'unknown','target_name':'vrate','pivot':False},
                #u-component of storm motion
                {'param':'USTM','level':'HTGL:0,6000','typeOfLevel':'heightAboveGroundLayer','orig_name':'unknown','target_name':'ustm','pivot':False},  
]

param_group5 = [
                #v-component of storm motion
                {'param':'VSTM','level':'HTGL:0,6000','typeOfLevel':'heightAboveGroundLayer','orig_name':'unknown','target_name':'vstm','pivot':False},  
                #Ozone mixing ratio
                {'param':'O3MR','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'unknown','target_name':'o3mr','pivot':True},
                #Sunshine duration (s)
                {'param':'SUNSD','level':'SFC:0','typeOfLevel':'surface','orig_name':'unknown','target_name':'sunsd','pivot':False},
]
param_group6 = [
                #Planetary boundary layer height
                {'param':'HPBL','level':'SFC:0','typeOfLevel':'surface','orig_name':'unknown','target_name':'hpbl','pivot':False},
                #Atmosphere absolute vorticity
                {'param':'ABS V','level':isobar_levels,'typeOfLevel':'isobaricInhPa','orig_name':'absv','target_name':'absv','pivot':True},              
]

ALL_PARAMS = [param_group0,param_group1,param_group2,param_group3,param_group4,param_group5,param_group6]

response = rc.get_control_file_template(dsid)
template = response['result']['template'] 
base_template = rc.read_control_file(template)
base_template['product'] = PRODUCT
base_template['param'] = None
base_template['level'] = None

def generate_requests():
    REQUESTS = []
    PARAM_PATHS = defaultdict(list)
    for location in LOCATIONS:
        d = df_labels[df_labels.location==location]
        start_date = pd.Timestamp(d.obs_datetime_start.min()).floor(freq='1D') -  pd.DateOffset(days=3)
        end_date = pd.Timestamp(d.obs_datetime_start.max()).ceil(freq='1D') +  pd.DateOffset(days=1)
        start_date = start_date.strftime('%Y%m%d%H%M')
        end_date = end_date.strftime('%Y%m%d%H%M')
        
        R = 0.25 #to pad location bounds
        d = df_grid_meta[df_grid_meta.location==location]
        minlon,minlat,maxlon,maxlat = d.geometry.total_bounds
        minlon = minlon-R
        minlat = minlat-R
        maxlon = maxlon+R
        maxlat = maxlat+R

        loc_template = copy.deepcopy(base_template)
        loc_template['date'] = f'{start_date}/to/{end_date}' 
        print('Location ',location, ' data request range: ',loc_template['date'])
        loc_template['wlon'] =  str(minlon)
        loc_template['elon'] = str(maxlon)
        loc_template['slat'] = str(minlat)
        loc_template['nlat'] = str(maxlat)
        

        #create data request parameters
        for grp_id,params in enumerate(ALL_PARAMS):
            out_dir = f'{TMP_DIR}/param_group{grp_id}/{location}/'
            filepath = f"{out_dir}/{start_date.split(' ')[0]}_{end_date.split(' ')[0]}.tar"
            PARAM_PATHS[location].append((Path(filepath),params))
            if os.path.exists(filepath):
                print(f'Found existing file {filepath}. Skipping request')
                continue
            #request files for parameter group if they don't exist
            template = copy.deepcopy(loc_template)
            template['param'] = params[0]['param']
            template['level'] = params[0]['level']
            for val in params[1:]:
                param = val['param']
                level = val['level']
                template['param'] = template['param']+f'/{param}'
                template['level'] = template['level']+f';{level}'
            REQUESTS.append({'template':template, 'out_dir':out_dir, 'filepath':filepath, 'location': location})
    return REQUESTS,PARAM_PATHS

while True:
    REQUESTS,PARAM_PATHS = generate_requests()
    results = pqdm(REQUESTS, download_data, n_jobs=MAX_CONCURRENT_REQUESTS,argument_type=None) 
    completed = [res for res in results if res['success']==True]
    if len(completed)>0:
        print('Completed requests: ',completed)
    failed = [res for res in results if res['success']==False]
    if len(failed)>0:
        print('The following requests failed. Retrying...')
        print(failed)
    else:
        break

# if len(failed)>0:
#     print('####### The following requests failed. Please rerun the script to try again ##########')
#     raise SystemExit


print('######## All downloads complete. Processing data ##########')

#Done downloading data; read grib files and process data into dataframe
def read_grib_to_df(path,level_filter):
    d = xr.load_dataset(path, engine="cfgrib",filter_by_keys={'typeOfLevel': level_filter}).to_dataframe().reset_index()
    return d

def extract_data(paths,param_grp):
    #process each file in paths for each parameter in param_grp
    cols_index=['longitude','latitude','valid_time']
    ds = []
    for param in param_grp:
        args = [(p,param['typeOfLevel']) for p in paths]
        results = pqdm(args, read_grib_to_df, n_jobs=N_JOBS,argument_type='args') 
        d = pd.concat(results)
        d = d.rename(columns={param['orig_name']:param['target_name']})
        cols_values = [param['target_name']]
        if param['pivot']:
            cols_pivot = [param['typeOfLevel']]
            d=d.pivot(index=cols_index, columns=cols_pivot, values=cols_values)
            cols_vars = [c for c in d if c not in cols_index]
            d.columns = d.columns.map(lambda x: (f"{x[0]}_{int(x[1]) if round(x[1])==x[1] else x[1]}"))
        else:
            d = d.set_index(cols_index)[cols_values]
        d = d.sort_index()
        ds.append(d)
        df = pd.concat(ds,axis=1)

    return df

df_data = [] #data for all locations
for location,param_paths in PARAM_PATHS.items():
    df_location = [] #data for all parameters for location
    for filepath,param_grp in param_paths:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'{filepath} not found. Please rerun script to download')
        outdir = f'{filepath.parent}/{filepath.stem}'
        os.makedirs(outdir,exist_ok=True)
        with tarfile.open(filepath, 'r:tar') as f:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, outdir)
            paths = glob.glob(f'{outdir}/*.grib2')
        df = extract_data(paths,param_grp)
        df_location.append(df)
    df_location = pd.concat(df_location,axis=1)
    df_data.append(df_location)

#for each grid, find the nearest coordinates in the 0.25 degree resolution data
dfs=[]
for location in LOCATIONS:
    path = glob.glob(f'{PARAM_PATHS[location][0][0].parent}/*/*.grib2')[0]
    dataset = xr.load_dataset(path, engine="cfgrib")
    df_loc = df_grid_meta[df_grid_meta.location==location].reset_index(drop=True)
    print(df_loc.geometry.total_bounds)
    print(dataset.longitude.data.min(),dataset.latitude.data.min(),dataset.longitude.data.max(),dataset.latitude.data.max())
    lons_data=[]
    lats_data=[]
    for _,row in tqdm(df_loc.iterrows()):
        geom=row.geometry
        x,y = geom.centroid.xy[0][0],geom.centroid.xy[1][0]
        r=0.25/2
        minlon = x-r
        maxlon = x+r
        minlat = y-r
        maxlat = y+r
        if minlon<0:
            minlon+=360
        if maxlon<0:
            maxlon+=360

        #latitude decrease
        DS = dataset.loc[{'latitude':slice(maxlat,minlat),'longitude':slice(minlon,maxlon)}]
        assert(DS.latitude.data.shape[0]==1)
        assert(DS.longitude.data.shape[0]==1)

        lons_data.append(DS.longitude.data[0])
        lats_data.append(DS.latitude.data[0])    
      
    df_loc['longitude'] = lons_data
    df_loc['latitude'] = lats_data
    dfs.append(df_loc)

df_grids = pd.concat(dfs).reset_index(drop=True)[['grid_id','longitude','latitude']]
df_data = pd.concat(df_data).drop(columns='vwsh_4294967294').reset_index()
df_data = pd.merge(df_grids,df_data,on=['longitude','latitude']).sort_values(by=['valid_time','grid_id']).reset_index(drop=True)
df_data.loc[df_data['longitude']>180,'longitude']=df_data.loc[df_data['longitude']>180]['longitude']-360

    
path = SAVE_DIR/ f'gfs_{STAGE}.csv'
df_data.to_csv(path,index=False)
print(df_data.shape)
print(f'Saved data to {path}')

if DO_CLEANUP:
    shutil.rmtree(TMP_DIR,ignore_errors=True)
