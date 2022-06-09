# "Single Day" NO2 Prediction for a Specific Location

Username: sukantabasu

## Summary

In this README file, we document all the required steps for predicting NO2 at a specific location for a single day.   
   

## Setup

1. Install the prerequisities
     - Python version 3.9.10, h1dd9edd_2_cpython, conda-forge

2. Install the following required python packages
     - lightgbm 3.3.0, pypi_0, pypi
     - netcdf4 1.5.7, py39h4a1dd59_1
     - scikit-learn 1.0.1, pypi_0, pypi
     - shapely 1.7.1, py39h9250791_0
     - glob2 0.7, pyhd3eb1b0_0

3. Other essential packages
	 - nco 
	 ```
	 brew install nco
	 ``` 
	 - [rdams_client api](https://github.com/NCAR/rda-apps-clients) 

## Hardware

The solution was run on iMac (2020), 3,3 GHz 6-Core Intel Core i5, 80 GB 2133 MHz DDR4, AMD Radeon Pro 5300 4 GB


## Download GFS data 

Historical GFS forecasts (grid size: 0.25 degree) are available from the **[rda](https://rda.ucar.edu/datasets/ds084.1/)** website. Only 3 hourly data are archived on this website. Free registration is required prior to access the download portal. The GFS data can be downloaded via [rdams_client api](https://github.com/NCAR/rda-apps-clients). 

Let us assume that we are interested in predicting NO2 for August 24, 2021. Then, we first create a control file (ds084.1_control1.ctl) with the following information. It will request air temperature, relative humidity, and wind data for a few vertical levels.  

```
dataset=ds084.1
date=202108240000/to/202108240000
datetype=init
param=TMP/R H/U GRD/V GRD/
level=HTGL:2/10/100
oformat=netCDF
```
Next, submit the data extraction request using the following command: 

```
python rdams_client.py -submit ds084.1_control1.ctl
```
You will receive an immediate confirmation as follows. 

```
{
   "status": "ok",
   "request_duration": "10.086723 seconds",
   "code": 200,
   "messages": [],
   "result": {
      "request_id": "556831"
   },
   "request_end": "2022-04-04T07:01:25.655236",
   "request_start": "2022-04-04T07:01:15.568513"
}
```
Repeat the previous two steps with a new control file (ds084.1_control2.ctl) as shown below. This file will request data for planetary boundary layer height, sensible heat flux, and ventillation rate. 

```
dataset=ds084.1
date=202108240000/to/202108240000
datetype=init
param=HPBL/SHTFL/VRATE
oformat=netCDF
```

The data download links will arrive after some time via email. Depending on the loads on NCAR's storage system, it can take a few minutes to more than an hour. Subsequently, the GFS files can be downloaded from the rda portal. These files are included in this [repo directory](../data/singleday/raw/GFS/). In VAR1 subdirectory, the datasets contain air temperature, relative humidity, and wind data. Whereas, in the VAR2 subdirectory, the files contain planetary boundary layer height, sensible heat flux, and ventillation rate data. 


# Preprocessing

### 1. Process OMI data

The OMI dataset is in hdf5 format. The NCKS tool, from the NCO package, is used to convert these files to netcdf format. Please use the following command for processing all the OMI files in a single directory: 

```
for FILE in *.he5; do ncks $FILE ${FILE%.he5}.nc; done
```
As an illustrative example, OMI data for August 24, 2021 is included in this [repo](../data/singleday/raw/OMI/). 

Then, run **[01\_Preprocess\_SingleDay\_OMI.ipynb](../notebooks/singleday/01_Preprocess_SingleDay_OMI.ipynb)** notebook. For specific location of interest, it identifies the nearest OMI grid point. Subsequently, it extracts OMI data (both tropospheric and total column NO2 values) for that specific grid point and save it in the **[testOMI.csv](../data/singleday/processed/test/OMI/testOMI.csv)** file. 

As an example, we consider NO2 prediction for grid ID "ZZ8JF" (near Los Angeles). The extracted full column and tropospheric OMI data are as follows for August 24, 2021: 
 
```
	NO2_OMI  NO2Tr_OMI
0  6.747592   3.842983
```

### 2. Process GFS data

Run **[02\_Preprocess\_SingleDay\_GFS.ipynb](../notebooks/singleday/02_Preprocess_SingleDay_GFS.ipynb)** notebook. Based on the downloaded GFS data, the code extracts all the necessary meteorological variables and saved in **[testGFS.csv](../data/singleday/processed/test/GFS/testGFS.csv)** file. 


## Prediction

Run **[03\_Forecast.ipynb](../notebooks/singleday/03_Forecast.ipynb)** to make the forecast for the single day scenario. This notebook expects **[testOBS.csv](../data/singleday/processed/test/STN/testOBS.csv)**,  **[testOMI.csv](../data/singleday/processed/test/OMI/testOMI.csv)**, and **[testGFS.csv](../data/singleday/processed/test/GFS/testGFS.csv)** as input.

Using the trained model parameters in **[model](../model/)** directory it performs the following computations: 

* 100 forecasts using model parameters from trnOpt = 1; compute median forecasts.
* 100 forecasts using model parameters from trnOpt = 2; compute median forecasts. 
* Whenever OMI data is available, use median forecasts from trnOpt = 2. When OMI data is missing, use median forecasts from trnOpt = 1. 

The final forecast is saved as **[submission\_sukantabasu.csv](../forecast/singleday/submission_sukantabasu.csv)**.

## License

MIT license. 
