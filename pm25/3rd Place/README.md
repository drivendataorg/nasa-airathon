# 3rd Place Solution - NASA Airathon: Predict Air Quality (Particulate Track)

Username: Katalip

## Summary

The solution focuses on fusing approved Aerosop Optical Depth (AOD) data from [MCD19A2 product](https://cmr.earthdata.nasa.gov/search/concepts/C1000000505-LPDAAC_ECS.html) and meteorological data from [GFS Forecasts products](https://rda.ucar.edu/datasets/ds084.1/#metadata/detailed.html?_do=y). AOD data is in HDF format and appropriate preprocessing steps outlined by organizers were followed: layer extraction, adding offset and scaling, constructing the grid in WGS84(EPSG:4326) coordinate system. The main problem with AOD data were missing values. Several approaches for time series imputation were tested among which linear interpolation performed the best. Data was interpolated for each grid id separately while ensuring proper placement of data points (sorting by date). GFS data is distributed in grib2 format at 0.25°, and main features were selected based on research articles (details can be found in the documentation). The GFS data was aggregated and grid coordinates at a finer resolution were rounded to the nearest GFS grid to join with AOD values. Additional data engineering was performed. The final model was an ensemble of tuned Random Forest Regressor and generalized Gradient Boosting Regressor from 'sklearn' library. The former model was tuned using 'Optuna' hyperparameter tuning framework, while default hyperprameters without fixing the random seed for the latter were used. Some experimentation showed such an approach more effective for increasing overall generalization perfomance of the pipeline. 


# Setup
An important note here, some python packages for preprocessing geospatial data (`geopandas`, `pyhdf`, `pyproj`) are not pure python libraries, and initialy data preparation was performed by using separate environments to avoid any package conflicts under python 3.10. To simplify the process, the whole pipeline was tested in one environment. Following setup was tested on Windows, Linux platforms under python 3.9 and python 3.7 respectively. Please, follow the appropriate instructions for your platform.

**For Windows** (tested with conda 4.11.0 and python 3.9):
1. Create the conda environment
```
conda create --name air_3rd python=3.9
```
2. Activate the environment and install the required python packages
```
conda activate air_3rd
bash create_env.sh
```
If you're facing any issues, please, refer to other [installation options](additional_inst_options.md)

**For Linux** (tested with Ubuntu distribution with conda 4.10.3 python 3.7 (necessary for `geopandas`)):
1. Create the conda environment
```
conda create --name air_3rd python=3.7
```
2. Activate the environment and install the required python packages
```
conda activate air_3rd
bash create_env.sh
```
If you're facing any issues, please, refer to other [installation options](link here)

The structure of the directory before running training or inference should be:
```
air_3rd_place
├── data
│   ├── processed  <- The final, canonical data sets for modeling
│   ├── raw        <- The original, immutable data dump  
|   |   ├──aod
|   |   |  ├──test
|   |   |  ├──train
|   |   |  ├── submission_format.csv
│   |   |  ├── grid_metadata.csv
│   |   |  ├── train_labels.csv
│   |   |  └── pm25_satellite_metadata.csv
|   |   ├──gfs
|   |   |  ├──donwloaded_files
|   |   |  └──metadata_selected_variables   <- Contains product specifiations for each feature from GFS 
|   └── interm     <- Directories necessary for intermediate preprocessing steps 
|       ├──gfs
|       ├──maiac
|       └──shape_files        
├── models             <- Trained and serialized models, saved encodings, model predictions
|   ├──predictions 
|   └──saved_encodings    
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── create_cfg.py
│   ├── delete_data_dirs.py
│   ├── prepare_data.py
|   ├── train.py
|   ├── predict.py   
|   └── preprocessing  <- Contains preprocessing scripts
├── cfg   <- Config files for training/predicting on new date periods
├── nbs   <- Additional Jupyter Notebooks for data requesting, hyperparameter tuning
|   ├── rda-apps-clients <- For requesting GFS data    
│   ├── modelling_optuna.ipynb <- Hyperprameter tuning for Random Forest Regressor
├── README.md              <- The top-level README for developers using this project.
├── create_env.sh          <- Script for installing necessary packages
├── environment_linux.yml  <- Alternate installation option for Linux users
└── environment_win.yml    <- Alternate installation option for Windows users
```

# Hardware

The solution, originally, was run on Windows 10 Pro(Version	10.0.19044 Build 19044)
- Number of CPUs: 4
- Processor: Intel® Core™ i5-9300H 2.4-4.1 Ghz
- Memory: 16 GB

Data preparation time: For training data (35.1Gb): ~12-15 hours (no multiprocessing), 4-6 hours (with multiprocessing)
For testing data (20.3Gb): ~8-12 hours (no multiprocessing), 3-4 hours (with multiprocessing)

Training time: ~3 minutes

Inference time: ~2 minutes

# Run Inference
To make predictions for a new period, first create a config file for it:
```
python src/create_cfg.py <CONFIG_NAME> # Names 'train','test_comp_period' are not allowed 
```
Necessary empty folders for required data files, intermediate preprocessing steps and model outputs will be generated.

Next, the following data preparation step is required
# Data Preparation For Inference
Path to the raw data (folder in appropriate format) should be specified or put in the directories 
of the repository (recommended) created for the <CONFIG_NAME>.

**For MAIAC AOD Data (MCD19A2):**
**Important:** HDF **file names** should follow the **same** naming conventions used in the competition files: " `{time_end}_{product}_{location}_{number}.ext`, where `time_end` is formatted as `YYYYMMDDTHHmmss`. In rare cases, locations and times may have more than one associated file. When this occurs, file number is denoted by `number`. For instance, `20191230T194148_misr_la_0.nc` represents first data file from the Los Angeles South Coast Air Basin collected by MISR on December 30, 2019 at 7:41pm UTC."

**Important:** The raw data files should be separated by years. For example, if new testing period spans over 2018 and 2019:
```
...
├── raw        <- The original, immutable data dump  
|      ├──aod
|         ├──test
|         |  ├──<CONFIG_NAME> 
|               ├── maiac
|                  ├──2018
|                  |  ├──20180201T024000_maiac_tpe_0.hdf
|                  |  ├──20180201T042000_maiac_tpe_0.hdf 
|                  |   ....
|                  └──2019
|                     ├──20190201T024000_maiac_tpe_0.hdf
|                     ├──20190201T042000_maiac_tpe_0.hdf 
|                      .... 
|         ├──train
|                    
...       
```

**For GFS Forecast Data:**
Data was requested using the following [Jupyter Notebook](nbs/rda-apps-clients/src/python/requesting_gfs_ds0841.ipynb). It was downloaded and put in the appropriate directories manually (only 6 files) due to implicit data preparation time from NCEP. They provide archived files, which can be just extracted in the respective folders.

Requested features were split in 2 groups: group_1, group_2. Specification of features present in each category are provided here (link).

Data should be put in the folders in the next way:

```
│   ├── raw        <- The original, immutable data dump  
|   |   ...
|   |   ├──gfs
|   |   |  ├──donwloaded_files
|   |   |     ├──dl
|   |   |     |   ├──<CONFIG_NAME>
|       |     |      ├──group_1
|       |     |         ├──gfs.0p25.2017010100.f024.grib2
|       |     |         ├──gfs.0p25.2017010106.f024.grib2
|       |     |         ...
|       |     |      ├──group_2
|       |     |         ├──gfs.0p25.2017010100.f024.grib2
|       |     |         ├──gfs.0p25.2017010106.f024.grib2
|       |     |         ...
|       |     ├──la
|       |     |    ├──<CONFIG_NAME>
|       |     |       ├──group_1
|       |     |       └──group_2
|       |     └──tp
|       |          ├──<CONFIG_NAME>
|       |             ├──group_1
|       |             └──group_2
|   |   └──metadata_selected_variables    
```

Specify the path to the **'submission_format.csv'** in `cfg/<CONFIG_NAME>.yml` ('path_labels' parameter). Expected in the same format as the one provided in the competition. This file shold contain dates, grid_ids for which predictions should be made. The same input is expected for a case of only one day and one grid id.
Default: submission_format.csv from competition page

In case new **new grid cells** are provided in the training data, update the **grid_metadata.csv** or provide a path to a new version in the 'cfg/<config_name>.yml'


To make a prediction on test files from competition (final model trained on competition training data is used):
1. Predict
```
python src/predict.py test_comp_period.yml 
```
Predictions are saved in models/pred_test_comp_period.csv


To make a prediction for a new period with <CONFIG_NAME> (final model trained on competition training data is used):
1. Prepare the data
```
python src/prepare_data.py <CONFIG_NAME>.yml 
```
2. Predict
```
python src/predict.py <CONFIG_NAME>.yml 
```
Predictions are saved in models/pred_<CONFIG_NAME>.csv

- How much space will any interim processed data files require?
- What files will be saved out during inference and what is each used for? Where will the final submission be saved out to by default?
- Are there any common pitfalls we should be aware of for troubleshooting?

# Run Training
# Data Preparation For Training
To train the model, path to the raw data (folder in appropriate format) should be specified or put in the directories 
of the repository (recommended)

**For MAIAC AOD Data (MCD19A2):**
**Important:** HDF **file names** should follow the **same** naming conventions used in the competition files: " `{time_end}_{product}_{location}_{number}.ext`, where `time_end` is formatted as `YYYYMMDDTHHmmss`. In rare cases, locations and times may have more than one associated file. When this occurs, file number is denoted by `number`. For instance, `20191230T194148_misr_la_0.nc` represents first data file from the Los Angeles South Coast Air Basin collected by MISR on December 30, 2019 at 7:41pm UTC."

**Important:** The raw data files should be separated by years. In this case:
```
...
├── raw        <- The original, immutable data dump  
|      ├──aod
|         ├──test
|         ├──train
|         ├── maiac
|              ├──2018
|                 ├──20180201T024000_maiac_tpe_0.hdf
|                 ├──20180201T042000_maiac_tpe_0.hdf 
|                     ....
|                 ├──2019
|                 ├──2020           
...       
```

**For GFS Forecast Data:**
Data was requested using the following [Jupyter Notebook](nbs/rda-apps-clients/src/python/requesting_gfs_ds0841.ipynb). It was downloaded and put in the appropriate directories manually (only 6 files) due to implicit data preparation time from NCEP. They provide archived files, which can be just extracted in the respective folders.

Requested features were split in 2 groups: group_1, group_2. Specification of features present in each category are provided here (link).

Data should be put in the folders in the next way:

```
│   ├── raw        <- The original, immutable data dump  
|   |   ...
|   |   ├──gfs
|   |   |  ├──donwloaded_files
|   |   |     ├──dl
|   |   |         ├──train
|       |            ├──group_1
|       |               ├──gfs.0p25.2017010100.f024.grib2
|       |               ├──gfs.0p25.2017010106.f024.grib2
|       |               ...
|       |            ├──group_2
|       |               ├──gfs.0p25.2017010100.f024.grib2
|       |               ├──gfs.0p25.2017010106.f024.grib2
|       |               ...
|       |      ├──la
|       |           ├──train
|       |              ├──group_1
|       |              ├──group_2
|       |      ├──tp
|       |           ├──train
|       |              ├──group_1
|       |              ├──group_2
|   |   └──metadata_selected_variables
```

Specify the path to the **'train_labels.csv'** in ```cfg/train.yml``` ('path_labels' parameter). Expected in the same format as the one provided in the competition. 
Default: train_lables.csv from competition page

In case new **new grid cells** are provided in the training data, update the **grid_metadata.csv** or provide a path to a new version in the 'cfg/train.yml'

## To train the model only with precomputed final views (csv files). Skipping the data preparation part
```
python src/train.py train.yml
```
## To train the model from raw data files
1. Prepare the data
```
python src/prepare_data.py train.yml 
```
2. Train
```
python src/train.py train.yml 
```

Models, necessary encodings are serialized and saved in 'models' folder.
Pickled Random Forest Regrossor weighs around 30 Mb, while GBR 200 Kb.


**Clarifications related to the ensurement of correct time usage**
From the competition page: 
```A note on time: Keep in mind, estimates should only factor in data available at the time of inference (training data excluded). You may only use data that would be available up through the day of estimation when generating predictions. Use of any future data after the day of estimation, excluding training data, is prohibited.```

All categorical encodings and computed features are available at the time of inference for the given data. 
 From the competition page reagrding datetime in train_labels.csv/submission_format.csv: 
 ```
The label’s datetime represents the start of a 24 period over which the air quality is averaged. For example, a label with datetime 2019-01-31T08:00:00Z represents an average taken from 2019-01-31T08:00:00Z to 2019-02-01T07:59:00Z (inclusive).

Therefore, you can use satellite data with an endtime on or before 2019-02-01T07:59:00Z for a label with datetime 2019-01-31T08:00:00Z . Note that this is 11:59pm local time (pacific time).
 ```
Satellite data with an endtime on or before the given datetime in the train_labels.csv/submission_format was used for joining data. For GFS data, 24 hour forecast products were used, which satisfy the requirements above.



