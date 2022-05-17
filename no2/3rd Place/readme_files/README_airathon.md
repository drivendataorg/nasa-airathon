# Solution - NASA Airathon: Predict Air Quality (Trace Gas Track)

Username: sukantabasu

## Summary

In this hackathon, remote-sensing data from the Ozone Monitoring Instrument (OMI) and numerical weather prediction data from the Global Forecast System (GFS) are utilized. For regression analysis, the [LightGBM](https://lightgbm.readthedocs.io/en/latest/#) model is used in conjunction with an ensemble approach. 

## Setup

1. Install the prerequisities
     - Python version 3.9.10, h1dd9edd_2_cpython, conda-forge

2. Install the following required python packages (`pip install -r requirements.txt`)
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

## Hardware

The solution was run on iMac (2020), 3,3 GHz 6-Core Intel Core i5, 80 GB 2133 MHz DDR4, AMD Radeon Pro 5300 4 GB

Training time: ~1.5 hour

Inference time: 25 sec


## Preprocessing

### 1. Process grid metadata

Run **[01\_Preprocess\_GridMetaData.ipynb](../notebooks/airathon/01_Preprocess_GridMetaData.ipynb)** notebook. This notebook reads the **[grid_metadata.csv](../data/airathon/raw/STN/grid_metadata.csv)** file (provided by the airathon organizers), and in turn, creates the **[grid_latlon.csv](../data/airathon/processed/grid_latlon.csv)** file. This processed file contains the centroid coordinates (i.e., latitude, longitude) for each grid as shown below. 

```
       ID   longitude   latitude
0   1X116  121.503307  24.998015
1   1Z2W7   77.282074  28.566368
2   3A3IE -117.911367  34.149445
3   3S31A -117.956283  33.814243
4   6EIL6   77.057495  28.566368
..    ...         ...        ...

```

### 2. Process time, grid locations, and labels for training set

Run **[02a\_Preprocess\_TrainingLabel.ipynb](../notebooks/airathon/02a_Preprocess_TrainingLabel.ipynb)** notebook. This notebook reads the **[train_labels.csv](../data/airathon/raw/STN/train_labels.csv)** file (provided by the airathon organizers) and combines it with **[grid_latlon.csv](../data/airathon/processed/grid_latlon.csv)** file to generate a new file called **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)**. The content of this combined file is as follows: 

```
                   datetime     ID   longitude   latitude        NO2
0      2019-01-01T08:00:00Z  3A3IE -117.911367  34.149445   8.695000
1      2019-01-01T08:00:00Z  3S31A -117.956283  33.814243  10.496667
2      2019-01-01T08:00:00Z  7II4T -118.046114  34.000629  37.208333
3      2019-01-01T08:00:00Z  8BOQH -118.450356  34.037858   9.791667
4      2019-01-01T08:00:00Z  A2FBI -117.417294  34.000629   4.308333
...                     ...    ...         ...        ...        ...

```
The total number of rows in **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)** is 36,131. 


### 3. Process time and grid locations for test set

In a similar manner, the processing of the test data are performed. Run **[02b\_Preprocess\_TestLabel.ipynb](../notebooks/airathon/02b_Preprocess_TestLabel.ipynb)** notebook. This notebook reads the **[submission_format.csv](../data/airathon/raw/STN/submission_format.csv)** file (provided by the airathon organizers) and combines it with **[grid_latlon.csv](../data/airathon/processed/grid_latlon.csv)** file to generate **[testOBS.csv](../data/airathon/processed/test/STN/testOBS.csv)** file. The content of this new file is as follows:

```
                   datetime     ID   longitude   latitude
0      2018-09-08T08:00:00Z  3A3IE -117.911367  34.149445
1      2018-09-08T08:00:00Z  3S31A -117.956283  33.814243
2      2018-09-08T08:00:00Z  7II4T -118.046114  34.000629
3      2018-09-08T08:00:00Z  8BOQH -118.450356  34.037858
4      2018-09-08T08:00:00Z  A2FBI -117.417294  34.000629
...                     ...    ...         ...        ...
```
The total number of rows in **[testOBS.csv](../data/airathon/processed/test/STN/testOBS.csv)** is 16,350.


### 4. Process OMI data

#### Training set

The OMI dataset is in hdf5 format. The NCKS tool, from the NCO package, is used to convert these files to netcdf format. Please use the following command for processing all the OMI files in a single directory: 

```
for FILE in *.he5; do ncks $FILE ${FILE%.he5}.nc; done
```

Then, run **[03a\_Preprocess\_TrainingOMI\_Step1.ipynb](../notebooks/airathon/03a_Preproces_TrainingOMI_Step1.ipynb)** notebook. For each 5km x 5km grid, it locates the nearest OMI grid point. Subsequently, it extracts OMI time-series data (both tropospheric and total column NO2 values) for that specific grid point and save it in a csv file. The missing OMI values are represented as NaNs. The files are named as: **[1X116\_trainOMI.csv](../data/airathon/processed/train/OMI/1X116_trainOMI.csv)**, **[1Z2W7\_trainOMI.csv](../data/airathon/processed/train/1Z2W7_trainOMI.csv)**, etc. There are a total of 68 files in **[train/OMI/](../data/airathon/processed/train/OMI/)** folder. 


Next, run **[03b\_Preprocess\_TrainingOMI\_Step2.ipynb](../notebooks/airathon/03b_Preproces_TrainingOMI_Step2.ipynb)** notebook. It combines all the gridID_trainOMI files (e.g., **[1X116\_trainOMI.csv](../data/airathon/processed/train/OMI/1X116_trainOMI.csv)**, **[1Z2W7\_trainOMI.csv](../data/airathon/processed/train/OMI/1Z2W7_trainOMI.csv)**) into a single **[trainOMI.csv](../data/airathon/processed/train/OMI/trainOMI.csv)** file. This file contains 2 variables: **NO2\_OMI** (NO2 concentration in the entire atmospheric column) and **NO2Tr\_OMI** (NO2 concentration in the tropospheric column). Overall statistics is as follows: 

```
	    NO2_OMI	    NO2Tr_OMI
count	18397	    18276
mean	7.246708	4.707578
std	    3.283868	3.390332
min	    0.007125	0.019817
25%	    5.305796	2.646509
50%	    6.472640	3.862540
75%	    8.185221	5.620449
max	    32.771667	31.05511

```

Since this is a forecast challenge, OMI data from the same day as the station observations are only considered. In other words, no future or past data are used in training and prediction. 

_Please note that the total number of rows in **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)** and **[trainOMI.csv](../data/airathon/processed/train/OMI/trainOMI.csv)** are the same. However, there are numerous missing values in **[trainOMI.csv](../data/airathon/processed/train/OMI/trainOMI.csv)** file. Approximately, there are only about 18,300 OMI samples in contrast to about 36,000 training cases._  

#### Test set 

Similar to the training set, please first convert the hdf5 files to netcdf format. 

Subsequently, run **[03c\_Preprocess\_TestOMI\_Step1.ipynb](../notebooks/airathon/03c_Preprocess_TestOMI_Step1.ipynb)** and **[03d\_Preprocess\_TestOMI\_Step2.ipynb](../notebooks/airathon/03d_Preprocess_TestOMI_Step2.ipynb)** notebooks. The structure of these notebooks are very similar to the ones described earlier under the training set category. Overall statistics of the **[testOMI.csv](../data/airathon/processed/test/OMI/testOMI.csv)** file is as follows:

```
		NO2_OMI		NO2Tr_OMI
count	8838		8759
mean	8.340485	5.873028
std		3.783766	3.910196
min		0.138372	0.000530
25%		5.950994	3.339092
50%		7.447121	4.836471
75%		9.659702	7.228231
max		32.558712	30.773388
``` 
_Please note that the total number of rows in **[testOBS.csv](../data/airathon/processed/test/STN/testOBS.csv)** and **[testOMI.csv](../data/airathon/processed/test/OMI/testOMI.csv)** are the same. However, there are numerous missing values in **[testOMI.csv](../data/airathon/processed/test/OMI/testOMI.csv)** file. Approximately, there are only about 8,800 OMI samples in contrast to about 16,300 test cases._  

### 5. Process GFS data

#### Data Download

Historical GFS forecasts (grid size: 0.25 degree) are downloaded from the **[rda](https://rda.ucar.edu/datasets/ds084.1/)** website. Only 3 hourly data are archived on this website. Free registration is required prior to access the download portal. 

During the hackathon, subsets of GFS data were *manually* downloaded via **Get a Subset** option. Due to the sheer size of the GFS dataset, this portal does not allow downloading of many files at a time. For this reason, each required meteorological variable (e.g., 10-m wind speed, 2-m air temperature, planetary boundary layer height) is downloaded separately for each region (i.e., Delhi, Los Angeles, Taipei). These files in *.tar format are included in this **[repo](../data/airathon/raw/GFS/)**. 

*Please note that each tar file contains more than 100k files.* They should be extracted in individual directories prior to running the processing notebooks (described below). You can do this with `bash extract_gfs.sh`.

Since this is a forecasting challenge, only forecast data from GFS are used. For example, if the forecast day of interest is March 1, 2021, then forecast fields from the GFS run initialized at 0 UTC of March 1, 2021 are utilized. The tar.gz files contain up to 48 h of forecast values. Due to distinct time zones of Delhi, Los Angeles, and Taipei, different forecast hours are extracted for each region. Each forecast horizon approximately ranges from (local) midnight (0 h) to following day's midnight (0 h) for all the locations. For example, at Los Angeles, midnight to midnight corresponds to 8 UTC to 32 UTC. Since archived GFS data are only availble every 3 hours, 6 h to 30 h ahead forecasts are considered here representing a full diurnal cycle.   

```
Delhi: 18 - 42 h ahead forecasts 
Los Angeles: 6 - 30 h ahead forecasts
Taipei: 15 - 39 h ahead forecasts 
```  

Essentially, for each grid location, 9 forecasted values of a specific meteorological variable are extracted per day. Let us consider 2-m air temperature (T2) as an example. For a given day, T2(t ~ 0 h), T2(t ~ 3 h), T2(t ~ 6 h), ..., T2(t ~ 24 h) forecasts are extracted for a specific location.      


#### Training set

Run **[04a\_Preprocess_TrainingGFS.ipynb](../notebooks/airathon/04a_Preprocess_TrainingGFS.ipynb)** notebook. Please set the **GFS\_DIR** properly. The notebook assumes that the GFS data have been extracted inside individual sub-directories of **GFS\_DIR** as follows: 

```
.
├── DEL
│   ├── PBLH
│   ├── RH
│   ├── SHFX
│   ├── T100
│   ├── T2
│   ├── U
│   ├── V
│   └── VENT
├── LAX
│   ├── PBLH
│   ├── RH
│   ├── SHFX
│   ├── T100
│   ├── T2
│   ├── U
│   ├── V
│   └── VENT
└── TAI
    ├── PBLH
    ├── RH
    ├── SHFX
    ├── T100
    ├── T2
    ├── U
    ├── V
    └── VENT

```
Please be aware of the large number of files in **GFS\_DIR**. 

The notebook will extract the following meteorological variables: 

- planetary boundary layer height (PBLH)
- 10-m wind speed (M10)
- 100-m wind speed (M100)
- 100-m wind direction (X100) and corresponding cosine and sine values
- power-law exponent computed between 10 m and 100 m (alpha)
- wind directional shear (beta)
- 2-m air temperature (T2)
- air temperature gradient between 2 m and 100 m (dT)
- surface relative humidity (RH)
- sensible heat flux (SHFX)
- gradient Richardson number (Rig) - not used
- ventillation rate (VENT)  

In addition, it will also extract week day and Julian day (cosine and sine values). All these variables are saved in **[trainGFS.csv](../data/airathon/processed/train/GFS/trainGFS.csv)**. The total number of rows of this file is the same as the **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)** file. 

#### Test set 

The notebook **[04b\_Preprocess_TestGFS.ipynb](../notebooks/airathon/04b_Preprocess_TestGFS.ipynb)** is used to generate the **[testGFS.csv](../data/airathon/processed/test/GFS/testGFS.csv)** file. The total number of rows of this file is the same as the **[testOBS.csv](../data/airathon/processed/test/STN/testOBS.csv)** file. The overall extraction process is identical to the generation of the **[trainGFS.csv](../data/airathon/processed/train/GFS/trainGFS.csv)** file and will not be repeated here for brevity. 


## Training

The **[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)** model is used for training. Please check **[05\_Train.ipynb](../notebooks/airathon/05_Train.ipynb)** notebook. 

The training process is performed twice by changing the trnOpt parameter. 

* trnOpt = 1: the model uses **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)** and **[trainGFS.csv](../data/airathon/processed/train/GFS/trainGFS.csv)** as input.  
* trnOpt = 2: the model uses **[trainOBS.csv](../data/airathon/processed/train/STN/trainOBS.csv)**,  **[trainOMI.csv](../data/airathon/processed/train/OMI/trainOMI.csv)**, and **[trainGFS.csv](../data/airathon/processed/train/GFS/trainGFS.csv)** as input.

Since **[trainOMI.csv](../data/airathon/processed/train/OMI/trainOMI.csv)** contains significant amount of missing data, trnOpt = 2 makes use of lesser number of samples. On the other hand, trnOpt = 1 has lesser number of input features (due to lack of OMI data). 

During training the following strategies are used: 

* extreme NO2 events (higher than 99.9th percentile) are removed from input data to avoid overfitting. Since MSE loss function is used with gradient boosting, such extreme values tend to have significant negative impact on model performance.   

* hyper-parameter tuning using 10-fold cross validation in conjunction with **[halving random grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html)**. The following hyper-parameters are tuned: 

```
'num_leaves':2**np.arange(2,10,1)
'max_depth':np.arange(1,11,1)
'learning_rate':np.array([0.005,0.01,0.025])
'reg_lambda':np.arange(0,3.01,0.5)
'reg_alpha':np.arange(0,3.01,0.5)
'subsample':np.arange(0.1,1.01,0.1) 
'colsample_bytree':np.arange(0.1,1.01,0.1)
```  

* with optimized hyper-parameters, 100 different tuned models are created. For each case, the overall training data is randomly split into 90% and 10% segments for training and validation, respectively. To avoid any temporal information leakage, the samples of the 10% validation segment are consecutive in time. Early stopping is used as regularization. 


Trained model parameters for both the trnOpt cases are saved in **[model](../model/)** directory using pickle.  


## Prediction

Run **[06\_Forecast.ipynb](../notebooks/airathon/06_Forecast.ipynb)** to make the forecast for all the test cases. This notebook expects **[testOBS.csv](../data/airathon/processed/test/STN/testOBS.csv)**,  **[testOMI.csv](../data/airathon/processed/test/OMI/testOMI.csv)**, and **[testGFS.csv](../data/airathon/processed/test/GFS/testGFS.csv)** as input.

Using the trained model parameters in **[model](../model/)** directory it performs the following computations: 

* 100 forecasts using model parameters from trnOpt = 1; compute median forecasts.
* 100 forecasts using model parameters from trnOpt = 2; compute median forecasts. 
* Whenever OMI data is available, use median forecasts from trnOpt = 2. When OMI data is missing, use median forecasts from trnOpt = 1. 

The final forecast is saved as **[submission\_sukantabasu.csv](../forecast/airathon/submission_sukantabasu.csv)**. This particular forecast leads to public and private leaderboard R2 scores of 0.4862 and 0.4419, respectively.    

## License

MIT license. 
