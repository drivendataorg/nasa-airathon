import re
import pyproj

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyhdf
from pyhdf.SD import SD, SDC


# Make this function more coherent later
# Add descriptions to each function
def read_hdf(file_name: str) -> pyhdf.SD.SD:
    """
    Copyright (C) 2014 John Evans

    This example code illustrates how to access and visualize an LP DAAC MCD19A2
    v6 HDF-EOS2 Sinusoidal Grid file in Python.

    If you have any questions, suggestions, or comments on this example, please use
    the HDF-EOS Forum (http://hdfeos.org/forums).  If you would like to see an
    example of any other NASA HDF/HDF-EOS data product that is not listed in the
    HDF-EOS Comprehensive Examples page (http://hdfeos.org/zoo), feel free to
    contact us at eoshelp@hdfgroup.org or post it at the HDF-EOS Forum
    (http://hdfeos.org/forums).

    Usage:  save this script and run

        $python MCD19A2.A2010010.h25v06.006.2018047103710.hdf.py


    Tested under: Python 3.7.3 :: Anaconda custom (64-bit)
    Last updated: 2020-07-23
    """

    hdf = SD(file_name, SDC.READ)
    return hdf


def read_datafield(hdf: pyhdf.SD.SD, datafield_name: str) -> pyhdf.SD.SDS: 
    
    # Select layer/band
    data3D = hdf.select(datafield_name)
    
    return data3D


def read_attr_and_check(data3D: pyhdf.SD.SDS) -> np.ma.core.MaskedArray:
    
    data = data3D[0, :, :].astype(np.double)

    # Read attributes of the fieldname.
    attrs = data3D.attributes(full=1)
    lna=attrs["long_name"]
    long_name = lna[0]
    vra=attrs["valid_range"]
    valid_range = vra[0]
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    
    try:
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
    except:
        scale_factor = 1.0

    ua=attrs["unit"]
    units = ua[0]

    try:
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
    except:
        add_offset = 0

    # Apply the attributes to the data.
    invalid = np.logical_or(data < valid_range[0], data > valid_range[1])
    invalid = np.logical_or(invalid, data == _FillValue)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor
    data = np.ma.masked_array(data, np.isnan(data))

    return data

def get_lon_lat(hdf: pyhdf.SD.SD, data: np.ma.core.MaskedArray) -> tuple:
    
    # Construct the grid.  The needed information is in a global attribute
    # called 'StructMetadata.0'.  Use regular expressions to tease out the
    # extents of the grid.
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                            (?P<upper_left_x>[+-]?\d+\.\d+)
                            ,
                            (?P<upper_left_y>[+-]?\d+\.\d+)
                            \)''', re.VERBOSE)

    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x'))
    y0 = np.float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                            (?P<lower_right_x>[+-]?\d+\.\d+)
                            ,
                            (?P<lower_right_y>[+-]?\d+\.\d+)
                            \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))
            
    nx, ny = data.shape
    x = np.linspace(x0, x1, nx, endpoint=False)
    y = np.linspace(y0, y1, ny, endpoint=False)
    xv, yv = np.meshgrid(x, y)

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    
    try:
        wgs84 = pyproj.Proj("+init=EPSG:4326")
    except:
        wgs84 = pyproj.Proj('epsg:4326')

    lon, lat= pyproj.transform(sinu, wgs84, xv, yv)

    return (lon, lat)

   
   