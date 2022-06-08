"""
to run the script:
python extract_vars.py <path_hdf_files> <path_shapefiles> <split_type>
<path_hdf_files> should contain hdf files.
<split_type>: train or test 
So, currently, I run this script for maiac 2018, 2019, 2020 separately. 
"""


import sys
import os
from tqdm import tqdm

try:
    from hdf_utils import read_hdf, read_datafield, read_attr_and_check, get_lon_lat
except:
    from preprocessing.aod.hdf_utils import read_hdf, read_datafield, read_attr_and_check, get_lon_lat

import re
import pyproj
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

import geopandas as gpd
import pandas as pd

import shapely.wkt
from shapely.geometry import Point, Polygon

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool, freeze_support, cpu_count


def read_shape(filename: str) -> gpd.geodataframe.GeoDataFrame:
    """Read and return shape file

    Returns:
        geopandas.geodataframe.GeoDataFrame: geoDataframe 
    """
    map_ = gpd.read_file(filename)
    return map_


def getBounds(shape):
    x1 = []
    y1 = []
    for i in range(len(shape)):
        if(isinstance(shape.iloc[i].geometry, shapely.geometry.polygon.Polygon)):  
            x = shape.exterior.iloc[i].coords.xy[0]
            y = shape.exterior.iloc[i].coords.xy[1]
            x1.append(min(x))
            x1.append(max(x))
            y1.append(min(y))
            y1.append(max(y))
        else:
            for poly in shape.iloc[i].geometry:
                x = poly.exterior.coords.xy[0]
                y = poly.exterior.coords.xy[1]
                x1.append(min(x))
                x1.append(max(x))
                y1.append(min(y))
                y1.append(max(y))
    
    return x1,y1

def getBoundary(shape, nx, ny):
    x, y = getBounds(shape)
    my_lats = np.linspace(np.min(y), np.max(y), nx)
    my_lons = np.linspace(np.min(x), np.max(x), ny)
    
    return(my_lats,my_lons)

def mask(map1, geo_df, datafield):
    pol = map1.geometry
    pts = geo_df.geometry
    test = geo_df
    l,t,df = [],[],[]
    for k in range(len(pts)):
        flag = True
        for i in range(len(pol)):
            if(pol[i].contains(pts[k])):
                l.append(geo_df.latitude[k])
                t.append(geo_df.longitude[k])
                df.append(geo_df[datafield][k])
                flag = False
                break

        if(flag):
            l.append(np.nan)
            t.append(np.nan)
            df.append(np.nan)

    newdf = pd.DataFrame({'latitude':l, 'longitude': t, datafield:df})
    return newdf

class extract_fields:

    def __init__(self, locations_key, datafs, path, path_shapefiles, save_path_root):
        self.locations_key = locations_key
        self.datafs = datafs
        self.path = path
        self.path_shapefiles = path_shapefiles
        self.save_path_root = save_path_root

    def extract(self, file: str) -> None:
        """[summary]

        Args:
            file (str): [description]
        """
        
        date_time, source, loc = file.split('_')[:3]
        loc_fullname = self.locations_key[loc] 
        file_n = file.split('_')[-1].split('.')[0]


        hdf = read_hdf(f'{self.path}/{file}')

        for datafield in self.datafs:
            
            data3D = read_datafield(hdf, datafield)
            data = read_attr_and_check(data3D)
            lon, lat = get_lon_lat(hdf, data)

            # For each shapefile in this location....
            grid_ids = os.listdir(f'{self.path_shapefiles}/{loc_fullname}') # {sys.argv[2]}
            

            for grid_id in grid_ids:
                
                # Load shape file
                shp_file = f'{self.path_shapefiles}/{loc_fullname}/{grid_id}/{grid_id}.shp'
                m = read_shape(shp_file)
                

                geo_ds = pd.DataFrame({'latitude':lat.flatten(), 'longitude': lon.flatten(), f'{datafield}':data.flatten()})
                # clip data with shape file boundaries
                
                nx, ny = data.shape
                x1, y1 = getBoundary(m, nx, ny)

                temp = geo_ds[geo_ds['latitude'] >= min(x1)]
                temp = temp[temp['latitude'] <= max(x1)]
                temp = temp[temp['longitude'] >= min(y1)]
                temp = temp[temp['longitude'] <= max(y1)]
                crc = {'init':'epsg:4326'}
                
                geometry = [Point(xy) for xy in zip(temp['longitude'], temp['latitude'])]
                geo_df = gpd.GeoDataFrame(temp, crs = crc, geometry = geometry)
                geo_df.reset_index(drop=True, inplace=True)

                msk = mask(m, geo_df, datafield)

                save_path = f'{self.save_path_root}/{source}/{loc}/{grid_id}/{date_time}'
                
                
                os.makedirs(save_path, exist_ok = True)

                msk.to_csv(f'{save_path}/{datafield}_{file_n}.csv')
                                


if __name__ == "__main__":

    locations_key = {'tpe':'Taipei', 'dl':'Delhi', 'la':'Los Angeles (SoCAB)'}
    datafs = ['Optical_Depth_055','Optical_Depth_047','AOD_Uncertainty','FineModeFraction','Column_WV','AOD_QA','AOD_MODEL','Injection_Height']
    path = sys.argv[1]
    path_shapefiles = sys.argv[2]
    split_type = sys.argv[3]
    save_path_root = sys.argv[3]

    ext = extract_fields(locations_key, datafs, path, path_shapefiles, save_path_root)

    filenames = os.listdir(path)

    freeze_support()
    
    pool = Pool(cpu_count()) 

    for _ in tqdm(pool.imap_unordered(ext.extract, filenames), total=len(filenames)):
        pass

    
