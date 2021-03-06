"""
Generate final training/ inference dataset
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
# locations=sorted(df_meta.location.unique())
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


##Process OMI data
omi = pd.read_csv(f'{DATA_DIR_INTERIM}/omi/omi_{STAGE}.csv')
omi['file_datetime_end']=pd.to_datetime(omi.date).dt.tz_localize(None)
omi = pd.merge(omi,df_meta)
omi=omi.sort_values(by=['file_datetime_end','grid_id']).reset_index(drop=True)

def impute_omi(df):
    df=df.sort_values(by=['file_datetime_end','grid_id']).reset_index(drop=True)
    grps = []
    for grid_id, grp in list(df.groupby(['grid_id'])):
        try:
          grp[feat_omi] = grp[feat_omi].fillna(grp[feat_omi].interpolate(limit_direction='forward',method='pchip')).ffill()
        except:
          grp[feat_omi] = grp[feat_omi].fillna(grp[feat_omi].interpolate(limit_direction='forward',method='linear')).ffill()
        grps.append(grp)
    df=pd.concat(grps).sort_values(by=['file_datetime_end','grid_id']).reset_index(drop=True)
    return df

feat_omi = ['ColumnAmountNO2','ColumnAmountNO2CloudScreened','ColumnAmountNO2Trop','ColumnAmountNO2TropCloudScreened','Weight']
omi[feat_omi[:-1]] = omi[feat_omi[:-1]] /10e15
omi = impute_omi(omi)

##Process TROPOMI data 
tropomi = pd.read_csv(f'{DATA_DIR_INTERIM}/tropomi/tropomi_{STAGE}.csv',parse_dates=['file_datetime_start','file_datetime_end']).rename(columns={'latitude':'lat_trop','longitude':'lon_trop'})
tropomi = tropomi.groupby(['grid_id','granule_id','file_datetime_start','file_datetime_end']).mean().reset_index()
tropomi = pd.merge(tropomi,df_meta)
tropomi=tropomi.sort_values(by=['file_datetime_end','grid_id']).reset_index(drop=True)

#calculate direction,distance of extracted tropomi data from grid bounds
g = Geod(ellps='WGS84')
fwd_azimuth,back_azimuth,distance = g.inv( tropomi.lon_trop,tropomi.lat_trop, tropomi.grid_minx, tropomi.grid_miny)
tropomi['grid_min_fwd_az_trop'] = fwd_azimuth
tropomi['grid_min_bck_az_trop'] = back_azimuth
tropomi['grid_min_dist_trop'] = distance ##

fwd_azimuth,back_azimuth,distance = g.inv( tropomi.lon_trop,tropomi.lat_trop, tropomi.grid_maxx, tropomi.grid_maxy)
tropomi['grid_max_fwd_az_trop'] = fwd_azimuth
tropomi['grid_max_bck_az_trop'] = back_azimuth
tropomi['grid_max_dist_trop'] = distance

##Process GEOS data
df_geos = pd.read_csv(f'{DATA_DIR_INTERIM}/geos/geos_{STAGE}.csv',parse_dates=['time']).rename(columns={'time':'data_datetime'})
df_geos = df_geos.sort_values(by=['grid_id','data_datetime']).reset_index(drop=True)

# at midday(UTC), hindcasts are generated for last 24 hours
# => data past midday is only available after the following day's midday
# So data generation time is:
df_geos['current_day_noon'] = df_geos.data_datetime.dt.floor('D')+pd.DateOffset(hours=12)
df_geos.loc[df_geos.data_datetime<df_geos.current_day_noon,'data_generation_datetime'] = df_geos.loc[df_geos.data_datetime<df_geos.current_day_noon].current_day_noon
df_geos.loc[df_geos.data_datetime>df_geos.current_day_noon,'data_generation_datetime'] = df_geos.loc[df_geos.data_datetime>df_geos.current_day_noon].current_day_noon+pd.DateOffset(hours=24)

#assert df_geos.data_generation_datetime.dt.time.nunique() ==1 
df_geos = pd.merge(df_geos,df_meta,on='grid_id').drop(columns=['tz','wkt','geometry','location'])
df_geos['no2']=df_geos['no2'].fillna(0)
df_geos = df_geos.sort_values(by=['grid_id','data_datetime']).reset_index(drop=True)

#calculate direction,distance of extracted GEOS data from grid bounds
g = Geod(ellps='WGS84') 
fwd_azimuth,back_azimuth,distance = g.inv( df_geos.lon,df_geos.lat, df_geos.grid_minx, df_geos.grid_miny)
df_geos['grid_min_fwd_az_25'] = fwd_azimuth
df_geos['grid_min_bck_az_25'] = back_azimuth
df_geos['grid_min_dist_25'] = distance

fwd_azimuth,back_azimuth,distance = g.inv( df_geos.lon,df_geos.lat, df_geos.grid_maxx, df_geos.grid_maxy)
df_geos['grid_max_fwd_az_25'] = fwd_azimuth
df_geos['grid_max_bck_az_25'] = back_azimuth
df_geos['grid_max_dist_25'] = distance


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
# print(feat_gfs)

##features list
feat_meta = ['location_code','grid_minx', 'grid_miny', 'grid_maxx', 'grid_maxy']

#for trop
feat_meta_trop = ['lat_trop','lon_trop','grid_min_fwd_az_trop', 'grid_min_bck_az_trop', 'grid_max_fwd_az_trop',
                  'grid_max_bck_az_trop','grid_min_dist_trop', 'grid_max_dist_trop']
feat_meta = feat_meta+feat_meta_trop

#for .25
feat_meta25  = ['lat','lon'] + ['grid_min_fwd_az_25', 'grid_min_bck_az_25', 'grid_max_fwd_az_25', 'grid_max_bck_az_25','grid_min_dist_25', 'grid_max_dist_25']
feat_meta = feat_meta+feat_meta25

feat_elev = ['elev_mean','elev_median','elev_min','elev_max','elev_std','elev_skew','elev_kurt']


feat_label = ['obs_datetime_start_hour','obs_datetime_start_dow','obs_datetime_start_month'] ##features extracted from teh label

feat_time = ['file_end_obs_start_diff','file_end_start_diff']#'trop_file_start_obs_start_diff']

feat_meta = feat_meta+feat_elev+feat_label+feat_time

feat_trop = ['qa_value', 'nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision', 
             'nitrogendioxide_tropospheric_column_precision_kernel', 'air_mass_factor_troposphere', 'air_mass_factor_total', 
             'tm5_tropopause_layer_index']


feat_geos = ['no2_min','no2_max','no2_mean','no2_sum']


feat_all = feat_meta+feat_omi+feat_trop+feat_geos+feat_gfs

assert len(list(set(feat_all))) == len(feat_all)
print(f'Features #{len(feat_all)}')

grid_geos = {}
for grid_id,grp in df_geos.groupby('grid_id'):
    grid_geos[grid_id]=grp

grid_gfs = {}
for grid_id,grp in df_gfs.groupby('grid_id'):
    grid_gfs[grid_id]=grp

def get_data(obs):
    """
    Extract data required to predict a single observation i.e. NO2 for a specific grid_id and datetime
    """

    data_geos = grid_geos[obs.grid_id]
    data_geos =  data_geos[(data_geos.data_generation_datetime<obs.obs_datetime_end)].tail(24).copy()
    max_gen_date = data_geos.data_generation_datetime.max()
    assert(max_gen_date<obs.obs_datetime_end)

    data_gfs = grid_gfs[obs.grid_id]
    
    #forecasts are available 3h earlier
    data_gfs = data_gfs[(data_gfs.valid_time<obs.obs_datetime_end)].tail(N_GFS_SAMPLES).copy()

    data_omi = omi[(omi.grid_id==obs.grid_id)&(omi.file_datetime_end<obs.obs_datetime_end)].tail(1).copy().reset_index()

    data = tropomi[(tropomi.grid_id==obs.grid_id)&(tropomi.file_datetime_end<obs.obs_datetime_end)].tail(1).copy().set_index('grid_id')

    data['file_end_obs_start_diff'] = (data.file_datetime_end-obs.obs_datetime_start)/np.timedelta64(1, 'h')
    data['file_end_start_diff'] = (data.file_datetime_end-data.file_datetime_start)/np.timedelta64(1, 'h')
    for c in feat_label:
        data[c] = obs[c]

    if len(data_omi) == 0:
        print(f'skipping {obs.grid_id}: {obs.obs_datetime_start}')

    for c in feat_omi:
        data[c]=data_omi[c].values[0]

    c='no2'
    data['no2_min'] = data_geos[c].min()
    data['no2_max'] = data_geos[c].max()
    data['no2_mean'] = data_geos[c].mean()
    data['no2_sum'] = data_geos[c].sum()

    #gfs stats
    gfs_sl=data_gfs.drop(columns='valid_time').groupby('grid_id')[vars_gfs_singlelevel].agg(stats_gfs_singlelevel)
    gfs_sl.columns = gfs_sl.columns.map('_'.join)

    gfs_ml = data_gfs.drop(columns='valid_time').groupby('grid_id')[vars_gfs_multilevel].agg('mean')
    gfs_ml.columns = gfs_ml.columns.map(lambda x:f'{x}_mean')

    data = pd.concat([data,gfs_sl,gfs_ml],axis=1)

    for c in feat_meta25:
        data[c]=data_geos[c].values[0]

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

##add loc_trop_mean features
feat_trop=['qa_value','nitrogendioxide_tropospheric_column','nitrogendioxide_tropospheric_column_precision','nitrogendioxide_tropospheric_column_precision_kernel','air_mass_factor_troposphere','air_mass_factor_total','tm5_tropopause_layer_index']
loc_trop = df_data.groupby(['datetime','location_code'])[feat_trop].mean().reset_index()
feat_trop_loc = [f'{c}_loc' for c in feat_trop]
loc_trop = loc_trop.rename(columns={feat_trop[i]:feat_trop_loc[i] for i in range(len(feat_trop))})
df_data=pd.merge(df_data,loc_trop,on=['datetime','location_code'])
df_data=df_data.sort_values(by=['datetime','grid_id']).reset_index(drop=True)

path = SAVE_DIR/ f'{STAGE}_tail{N_GFS_SAMPLES}.pkl'
df_data.to_pickle(path)
print(f'Saved data to {path}')