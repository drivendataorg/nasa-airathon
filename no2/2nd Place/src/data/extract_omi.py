"""
    Downloads OMI data from an s3 bucket and extracts relevant features
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
parser.add_argument("--stage", help="Stage of data to process, i.e train,test,prod", required=True) ##output file will be saved as omi_{STAGE}
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
SAVE_DIR = DATA_DIR_INTERIM/"omi"

TMP_DIR = DATA_DIR_INTERIM/'tmp/omi'/STAGE #to temporarily save downloaded files
N_JOBS = mp.cpu_count()*2
if args.n_jobs:
    N_JOBS = args.n_jobs

print(f'using {N_JOBS} processes')

df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
# df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})

df_satellite_meta = pd.read_csv(PATH_SATMETA)
df_satellite_meta = df_satellite_meta[(df_satellite_meta['product']=='omi') & (df_satellite_meta.split==STAGE)]
df_satellite_meta['s3url'] = df_satellite_meta[args.s3url]

for d in [DATA_DIR_INTERIM,TMP_DIR,SAVE_DIR]:
    os.makedirs(d,exist_ok=True)


DATAFIELDS = ['ColumnAmountNO2','ColumnAmountNO2CloudScreened','ColumnAmountNO2Trop','ColumnAmountNO2TropCloudScreened','Weight']


#map grid coordinates to the nearest pixel in OMI data 
lon = np.arange(0., 1440.0) * 0.25 - 180 + 0.125
lat = np.arange(0., 720.0) * 0.25 - 90 + 0.125
grid_ixs = []
for _,row in df_grid_meta.iterrows():
    bounds = row.geometry.bounds
    OFFx=0.01
    OFFy=0.01
    eps=0.001
    found=False
    ixs0=[]
    ixs1=[]

    while not found:
        if len(ixs0)==0:
            OFFy+=eps
            ixs0=np.where( (lat>=(bounds[1]-OFFy)) & (lat<=(bounds[3]+OFFy)) )[0]

        if len(ixs1)==0:
            OFFx+=eps
            ixs1=np.where( (lon>=(bounds[0]-OFFx)) & (lon<=(bounds[2]+OFFx)) )[0]

        if len(ixs0)>0 and len(ixs1)>0:
            found=True
            break

    grid_ixs.append([list(ixs0),list(ixs1),lat[ixs0],lon[ixs1]])


grid_ixs = np.array(grid_ixs).astype(float)
df_grid_meta['ixs0'] =  grid_ixs[:,0].astype(int)
df_grid_meta['ixs1'] =  grid_ixs[:,1].astype(int)
df_grid_meta['lat'] =  grid_ixs[:,2]
df_grid_meta['lon'] =  grid_ixs[:,3]


def extract_data(s3_path):  
    filename = s3_path.split('/')[-1]
    local_path = TMP_DIR/filename
    utils.download_s3_file(s3_path,local_path)

    date = filename.split('_')[0]
  
    DATA_DICT = {}
    for _,row in df_grid_meta.iterrows():
        DATA_DICT[row.grid_id] = [date,row.lat,row.lon]

    nc = Dataset(local_path)
    grp = nc.groups['HDFEOS'].groups['GRIDS'].groups['ColumnAmountNO2']
    all_data=[]

    for datafield in DATAFIELDS:
        var = grp.groups['Data Fields'].variables[datafield]

        var.set_auto_maskandscale(False)
        data = var[:].astype(np.float64)
        scale = var.ScaleFactor
        offset = var.Offset
        missing_value = var.MissingValue
        fill_value = var._FillValue

        title = var.Title
        units = var.Units
        data[data == missing_value] = np.nan
        data[data == fill_value] = np.nan
        data = scale * (data - offset)
        all_data.append(data)

    all_data  = np.array(all_data)
    nc.close()

    if DO_CLEANUP:
        os.remove(local_path) # delete local file

    for _,row in df_grid_meta.iterrows():
        ixs0 = int(row.ixs0)
        ixs1 = int(row.ixs1)
        d = all_data[:,ixs0,ixs1]
        DATA_DICT[row.grid_id] = DATA_DICT[row.grid_id]+list(d)

    df=pd.DataFrame(DATA_DICT).T
  
    return df


filelist = df_satellite_meta.s3url.tolist()
results = pqdm(filelist, extract_data, n_jobs=N_JOBS,argument_type=None)
df_data=pd.concat(results)

cols=['date','lat','lon']+DATAFIELDS
df_data = df_data.rename(columns={i:cols[i] for i in range(len(cols))}).reset_index().rename(columns={'index':'grid_id'})
path = SAVE_DIR/ f'omi_{STAGE}.csv'
df_data.to_csv(path,index=False)
print(f'Saved data to {path}')