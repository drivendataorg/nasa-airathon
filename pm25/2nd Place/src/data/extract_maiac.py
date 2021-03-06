"""
    Downloads MAIC data from an s3 bucket and extracts relevant features
"""
from pathlib import Path
import sys,os,re,glob,random,ast,warnings,multiprocessing,argparse
import pyproj
from pyhdf.SD import SD, SDC
import numpy as np, pandas as pd, geopandas as gpd
import multiprocessing as mp
from tqdm.auto import tqdm
from pqdm.processes import pqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--stage", help="Stage of data to process, i.e train,test,prod", required=True) 
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
SAVE_DIR = DATA_DIR_INTERIM/"maiac"

TMP_DIR = DATA_DIR_INTERIM/'tmp/maiac'/STAGE #to temporarily save downloaded files
N_JOBS = mp.cpu_count()*2
if args.n_jobs:
    N_JOBS = args.n_jobs

print(f'using {N_JOBS} processes')

df_grid_meta = gpd.read_file(DATA_DIR_RAW/ 'grid_metadata.csv')
df_grid_meta['location']=df_grid_meta.location.map({'Delhi':'dl','Los Angeles (SoCAB)':'la','Taipei':'tpe'})

df_satellite_meta = pd.read_csv(PATH_SATMETA)
df_satellite_meta = df_satellite_meta[(df_satellite_meta['product']=='maiac') & (df_satellite_meta.split==STAGE)]
df_satellite_meta['s3url'] = df_satellite_meta[args.s3url]

for d in [DATA_DIR_INTERIM,TMP_DIR,SAVE_DIR]:
    os.makedirs(d,exist_ok=True)




DATAFIELDS = ['Optical_Depth_047','Optical_Depth_055','AOD_Uncertainty','FineModeFraction',
              'Column_WV','AOD_QA','AOD_MODEL','Injection_Height','cosSZA','cosVZA',
              'RelAZ','Scattering_Angle','Glint_Angle']

# define crs using parameters from gridmetadata
sinu_crs = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext").crs
wsg84_crs = pyproj.CRS.from_epsg("4326")
# Set up transformers to project sinusoidal grid onto wgs84 grid
transformer = pyproj.Transformer.from_crs(
    sinu_crs,
    wsg84_crs,
    always_xy=True,  
)

def get_lonlat(x0,y0,x1,y1,nx,ny):
    x = np.linspace(x0, x1, nx, endpoint=False, dtype='float64')
    y = np.linspace(y0, y1, ny, endpoint=False, dtype='float64')
    xv, yv = np.meshgrid(x, y)

    lon, lat = transformer.transform(xv, yv)
    return lon,lat

keys=['[6671703.118, 3335851.559, 7783653.637667, 2223901.039333, 1200, 1200]',
 '[6671703.118, 3335851.559, 7783653.637667, 2223901.039333, 240, 240]',
 '[-11119505.196667, 4447802.078667, -10007554.677, 3335851.559, 1200, 1200]',
 '[-11119505.196667, 4447802.078667, -10007554.677, 3335851.559, 240, 240]',
 '[12231455.716333, 3335851.559, 13343406.236, 2223901.039333, 1200, 1200]',
 '[12231455.716333, 3335851.559, 13343406.236, 2223901.039333, 240, 240]',
 '[11119505.196667, 3335851.559, 12231455.716333, 2223901.039333, 1200, 1200]',
 '[11119505.196667, 3335851.559, 12231455.716333, 2223901.039333, 240, 240]']

LONLAT={}
for key in keys:
    x0,y0,x1,y1,nx,ny=ast.literal_eval(key)
    lon, lat = get_lonlat(x0,y0,x1,y1,nx,ny)
    LONLAT[key]= [lon,lat]


def extract_data(s3_path,location):
    granule_id = s3_path.split('/')[-1]
    local_path = str(TMP_DIR/granule_id)
    utils.download_s3_file(s3_path,local_path)

    hdf = SD(local_path, SDC.READ) # Read dataset
    date = local_path.split('/')[-1].split('_')[0]

    DATA_DICT = {}
    for _,row in df_grid_meta[df_grid_meta.location==location].iterrows():
        DATA_DICT[row.grid_id] = [date]
    for datafield in DATAFIELDS:
        data3D = hdf.select(datafield) #select sds
        # Read attributes for selected datafield
        attrs = data3D.attributes(full=1) 
        lna=attrs["long_name"]
        long_name = lna[0]
        vra=attrs["valid_range"]
        valid_range = vra[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        # ua=attrs["unit"]
        # units = ua[0]
        ##scale factor, offset
        scale_factor=1.
        add_offset=0.0
        if 'scale_factor' in attrs.keys():
        #except ['AOD_QA','AOD_MODEL','Injection_Height']:
            sfa=attrs["scale_factor"]
            scale_factor = sfa[0] 
            aoa=attrs["add_offset"]
            add_offset = aoa[0]
        
        orbits=[v for k,v in data3D.dimensions().items() if 'Orbits' in k][0]
        all_orbits_data=[]
        for i in range(orbits):
            data = data3D[i,:,:].astype(np.float32)
            # Apply the attributes to the data.
            invalid = np.logical_or(data < valid_range[0], data > valid_range[1])
            invalid = np.logical_or(invalid, data == _FillValue)
            data[invalid] = np.nan
            data = (data - add_offset) * scale_factor
            all_orbits_data.append(data)
        
        data=np.array(all_orbits_data)
        data=np.nanmean(data,axis=0)

        fattrs = hdf.attributes(full=1) #file attributes
        ga = fattrs["StructMetadata.0"] ##GROUP attributes
        gridmeta = ga[0]
        ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                                  (?P<upper_left_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<upper_left_y>[+-]?\d+\.\d+)
                                  \)''', re.VERBOSE)

        match = ul_regex.search(gridmeta)
        x0 = np.float64(match.group('upper_left_x'))
        y0 = np.float64(match.group('upper_left_y'))

        lr_regex = re.compile(r'''LowerRightMtrs=\(
                                  (?P<lower_right_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<lower_right_y>[+-]?\d+\.\d+)
                                  \)''', re.VERBOSE)
        match = lr_regex.search(gridmeta)
        x1 = np.float64(match.group('lower_right_x'))
        y1 = np.float64(match.group('lower_right_y'))

        nx, ny = data.shape
        key=str([x0,y0,x1,y1,nx,ny])

        lon,lat = LONLAT[key]
        
        for _,row in df_grid_meta[df_grid_meta.location==location].iterrows():
            bounds=row.geometry.bounds
            #expanding ROI to impute missing values
            for OFF in [i*0.01 for i in range(1,5)]:
                #increment area with a small offset
                ixs=np.where((lon>=bounds[0]-OFF) & (lon<=bounds[2]+OFF) & (lat>=bounds[1]-OFF) & (lat<=bounds[3]+OFF)) 
                d_mean=np.nanmean(data[ixs])
                d_median=np.nanmedian(data[ixs])
                if not np.isnan(d_mean):
                    assert(not np.isnan(d_median))
                    break

            DATA_DICT[row.grid_id].append(d_mean)
            DATA_DICT[row.grid_id].append(d_median)

    hdf.end()
    if DO_CLEANUP:
        os.remove(local_path) # delete local file

    df=pd.DataFrame(DATA_DICT).T
    return df


args = list(zip(df_satellite_meta.s3url.tolist(),df_satellite_meta.location.tolist()))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    results = pqdm(args, extract_data, n_jobs=N_JOBS,argument_type='args')

df_data=pd.concat(results).reset_index().rename(columns={'index':'grid_id'})
cols=['date']+[f'{d}_{s}' for d in DATAFIELDS for s in ['mean','median']]
df_data=df_data.rename(columns={i:cols[i] for i in range(len(cols))})

path = SAVE_DIR/ f'maiac_{STAGE}.csv'
df_data.to_csv(path,index=False)
print(f'Saved data to {path}')