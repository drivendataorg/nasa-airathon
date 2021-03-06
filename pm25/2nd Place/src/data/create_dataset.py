"""
Generate final training/ evaluation dataset
"""
import pandas as pd, numpy as np
import geopandas as gpd
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,argparse,time,shutil
from pqdm.processes import pqdm
from pyproj import Geod

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--stage", help="Stage of data to process, i.e train,test,prod", required=True) 
parser.add_argument("--labels", help="Path to labels.csv file", required=True)
parser.add_argument("--n_gfs",type=int, help="Number of prior GFS forecasts time steps to use", required=True )
parser.add_argument("--n_jobs",type=int, help="Number of processes. Defaults to number of processors", required=True)

args = parser.parse_args()
DATA_DIR = args.data_dir 
STAGE = args.stage 
LABELS_PATH = args.labels 
N_GFS_SAMPLES = args.n_gfs
N_JOBS = args.n_jobs


DATA_DIR = Path(DATA_DIR) 
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_INTERIM = DATA_DIR / "interim"
DATA_DIR_PROCESSED = DATA_DIR/"processed"
SAVE_DIR = DATA_DIR_PROCESSED/ STAGE #directory to save final output
os.makedirs(SAVE_DIR,exist_ok=True)

df_labels = pd.read_csv(LABELS_PATH)
df_labels['obs_datetime_start']=pd.to_datetime(df_labels.datetime).dt.tz_localize(None)
df_labels['obs_datetime_end'] = df_labels['obs_datetime_start'] + pd.DateOffset(hours=24)
df_labels['obs_datetime_start_dow'] = df_labels.obs_datetime_start.dt.day_of_week
df_labels['obs_datetime_start_month'] = df_labels.obs_datetime_start.dt.month
df_labels['obs_datetime_start_hour'] = df_labels.obs_datetime_start.dt.hour
df_labels=df_labels.sort_values(by=['obs_datetime_start','grid_id']).reset_index(drop=True)


#grid metadata
df_meta = gpd.read_file(DATA_DIR_RAW/'grid_metadata.csv')
locations=['Delhi', 'Los Angeles (SoCAB)', 'Taipei']
locations={locations[i]:i for i in range(len(locations))}
df_meta['location_code'] = df_meta.location.map(locations)

#add elevation data
df_elevation = pd.read_csv(f'{DATA_DIR_INTERIM}/elevation.csv').drop(columns='location')
df_meta=pd.merge(df_meta,df_elevation,on='grid_id')

## create grid bounds features
d = df_meta.geometry.bounds.copy()
for c in d:
    df_meta[f'grid_{c}']=d[c]

    
maiac = pd.read_csv(f'{DATA_DIR_INTERIM}/maiac/maiac_{STAGE}.csv')
maiac['file_datetime_end']=pd.to_datetime(maiac.date)
maiac['file_date']= maiac['file_datetime_end'].dt.date
maiac = pd.merge(maiac,df_meta,on='grid_id').sort_values(by=['grid_id','date']).reset_index(drop=True)

##Process GFS data
df_gfs = pd.read_csv(f'{DATA_DIR_INTERIM}/gfs/gfs_{STAGE}.csv',parse_dates=['valid_time'])
drop_cols = ['absv_1','absv_5','cape_9000','cin_9000','o3mr_450','o3mr_500','o3mr_550','o3mr_600','o3mr_650','o3mr_700','o3mr_750','o3mr_800','o3mr_850','o3mr_900','o3mr_925','o3mr_950','o3mr_975','o3mr_1000']
cols = [c for c in df_gfs if c not in drop_cols]
df_gfs = df_gfs[cols]

stats_gfs_singlelevel = ['min','mean','max']
stats_gfs_multilevel = ['mean']
ix_cols = ['valid_time','grid_id']
vars_gfs = [c for c in df_gfs.columns.tolist() if c not in ix_cols]
vars_gfs_singlelevel=['pot','aptmp','dpt','gust','vrate','ustm','vstm','pwat','tozne','pres','prmsl','hpbl','sunsd','pevpr']
vars_gfs_multilevel = [c for c in vars_gfs if c not in vars_gfs_singlelevel]

feats_gfs_singlelevel = [f'{v}_{s}' for v in vars_gfs_singlelevel for s in stats_gfs_singlelevel]
feats_gfs_multilevel = [f'{v}_{s}' for v in vars_gfs_multilevel for s in stats_gfs_multilevel]

feat_gfs = feats_gfs_singlelevel+feats_gfs_multilevel


#calculate direction,distance of extracted GFS data from grid bounds
df_gfs = pd.merge(df_gfs,df_meta[['grid_id','grid_minx','grid_miny','grid_maxx','grid_maxy']],on='grid_id').sort_values(by=ix_cols).reset_index(drop=True)
g = Geod(ellps='WGS84') 

fwd_azimuth,back_azimuth,distance = g.inv( df_gfs.longitude,df_gfs.latitude, df_gfs.grid_minx, df_gfs.grid_miny)
df_gfs['grid_min_fwd_az_25'] = fwd_azimuth
df_gfs['grid_min_bck_az_25'] = back_azimuth
df_gfs['grid_min_dist_25'] = distance

fwd_azimuth,back_azimuth,distance = g.inv( df_gfs.longitude,df_gfs.latitude, df_gfs.grid_maxx, df_gfs.grid_maxy)
df_gfs['grid_max_fwd_az_25'] = fwd_azimuth
df_gfs['grid_max_bck_az_25'] = back_azimuth
df_gfs['grid_max_dist_25'] = distance



##features list
cols_maiac =['Optical_Depth_047','Optical_Depth_055','AOD_Uncertainty','Column_WV','AOD_QA','AOD_MODEL','Injection_Height','cosSZA','cosVZA','RelAZ','Scattering_Angle','Glint_Angle',] #FineModeFraction
stats_maiac = ['mean','median']

feat_meta_grid = ['location_code','grid_minx', 'grid_miny', 'grid_maxx', 'grid_maxy']
feat_meta25  = ['latitude','longitude'] +['grid_min_fwd_az_25', 'grid_min_bck_az_25', 'grid_max_fwd_az_25', 'grid_max_bck_az_25','grid_min_dist_25', 'grid_max_dist_25']

cols_maiac = ['Optical_Depth_047', 'Optical_Depth_055', 'AOD_Uncertainty', 'Column_WV', 'AOD_QA',
        'AOD_MODEL', 'Injection_Height', 'cosSZA', 'cosVZA', 'RelAZ', 'Scattering_Angle', 'Glint_Angle']
feat_maiac = [f'{c}_{s}' for c in cols_maiac for s in stats_maiac]

feat_elev = ['elev_mean','elev_median','elev_min','elev_max','elev_std','elev_skew','elev_kurt']
feat_meta_maiac = feat_meta_grid+feat_elev
feat_meta = feat_meta_maiac+feat_meta25

feat_label = ['obs_datetime_start_dow','obs_datetime_start_month','obs_datetime_start_hour']

feat_all=feat_maiac+feat_meta+feat_label+['nhour']+feat_gfs

assert len(list(set(feat_all)))==len(feat_all)

grid_gfs = {}
for grid_id,grp in df_gfs.groupby('grid_id'):
    grid_gfs[grid_id]=grp
    
def get_data(obs):
    """
    Extract data required to predict a single observation i.e. PM2.5 for a specific grid_id and datetime
    """
    
    data_maiac = maiac[(maiac.grid_id==obs.grid_id)&(maiac.file_datetime_end<obs.obs_datetime_end)].tail().copy()

    if len(data_maiac) == 0:
        print(f'No data found for {obs.grid_id}: {obs.obs_datetime_end}')


    grp = data_maiac.copy()
    if len(data_maiac)>1:
        grp['file_datetime_end_diff'] = (grp.file_datetime_end-grp.file_datetime_end.values[0])/pd.Timedelta('1 hour') ##time diff between first and last available files

        ##group by date and select data from last date only
        grp = data_maiac.groupby(['grid_id','file_date'])[feat_maiac].mean().reset_index().tail(1).copy() 
        assert(len(data_maiac[feat_meta_maiac].drop_duplicates())==1)


    grp = grp.set_index('grid_id')
    assert(len(grp)==1)

    #add maiac meta
    for c in feat_meta_maiac:
        grp[c]=data_maiac[c].values[0]

    for c in feat_label:
        grp[c]=obs[c]

    ##add gfs features
    data_gfs = grid_gfs[obs.grid_id]
    data_gfs = data_gfs[(data_gfs.valid_time<obs.obs_datetime_end)].tail(N_GFS_SAMPLES)


    gfs_sl=data_gfs.drop(columns='valid_time').groupby('grid_id')[vars_gfs_singlelevel].agg(stats_gfs_singlelevel)
    gfs_sl.columns = gfs_sl.columns.map('_'.join)

    gfs_ml = data_gfs.drop(columns='valid_time').groupby('grid_id')[vars_gfs_multilevel].agg('mean')
    gfs_ml.columns = gfs_ml.columns.map(lambda x:f'{x}_mean')

    data = pd.concat([grp,gfs_sl,gfs_ml],axis=1)

    #add gfs meta
    for c in feat_meta25:
        data[c] = data_gfs[c].values[0]

    nhour= (obs.obs_datetime_end-data_maiac.file_datetime_end.values[-1])/pd.Timedelta('1 hour') #time diff between last available file time and obs end time
    data['nhour'] = nhour

    data = data.reset_index()
    data['datetime'] = obs.datetime
    data = data[['datetime','grid_id']+feat_all]
    assert(len(data)==1)
    
    res=pd.DataFrame([data.values[0]],columns=data.columns)
    res['target']=obs.value
    return res

args = [row for _,row in df_labels.iterrows()]
results = pqdm(args, get_data, n_jobs=N_JOBS,argument_type=None)
df_data=pd.concat(results).sort_values(by=['datetime','grid_id']).reset_index(drop=True)
df_data[feat_all]=df_data[feat_all].astype(np.float32)
df_data.insert(1,'obs_datetime_start',pd.to_datetime(df_data.datetime).dt.tz_localize(None))

##add location maiac mean features
feat_maiac_mean = [c for c in feat_maiac if '_mean' in c]
loc_maiac_mean = df_data.groupby(['datetime','location_code'])[feat_maiac_mean].mean().reset_index()
feat_maiac_loc = [f'{c}_loc' for c in feat_maiac_mean]
loc_maiac_mean= loc_maiac_mean.rename(columns={feat_maiac_mean[i]:feat_maiac_loc[i] for i in range(len(feat_maiac_mean))})
df_data=pd.merge(df_data,loc_maiac_mean,on=['datetime','location_code'])

path = SAVE_DIR/ f'{STAGE}_tail{N_GFS_SAMPLES}.pkl'
df_data.to_pickle(path)

print(f'Saved data to {path}')