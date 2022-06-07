# nasa-airathon

[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/nasa-aq-banner-web.jpg)

# NASA Airathon: Predict Air Quality - Particulate Track

## Goal of the Competition
The goal of this track of the competition was to use remote sensing data and other geospatial data sources to develop models for estimating daily levels of PM2.5 with high spatial resolution. Successful models could provide critical data to help the public take action to reduce their exposure to air pollution.

## What's in this Repository

This repository contains code from winning competitors in the [NASA Airathon: Predict Air Quality - Particulate Track](https://www.drivendata.org/competitions/88/competition-air-quality-pm/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | vstark21 | 0.759 | 0.806 | Used MAIAC and MISR satellite data, GFS forecast data, and NASADEM elevation data to train XGBoost, CatBoost, and LGBM models that were ensembled together using 5-fold linear regression. This was done for individual locations in one pipeline, and for all the locations together in another; the results were then ensembled together for final results. The mean and variance of each cell were used, and gridwise mean imputation was used to fill in missing values. Optuna, a hyperparameter tuning framework, was used to tune hyperparameters and both R-squared and RMSE were used to evaluate the model.
2   | karelds | 0.772 | 0.798 | Used an ensemble of 45 LGBM models trained on MAIAC satellite data and GFS forecasts. GFS variables included those related to air humidity, soil temperature, soil humidity, air temperature, wind velocity, wind direction and rainfall/ precipitation. GFS forecasts from up to 3 days preceding the relevant date with different lookback periods were used. There were separate models trained for each location, and 5-fold cross-validation was used instead of time-based splits. 45 models (from 3 datasets, 5 folds, and 3 locations) were used in the final ensemble. RMSE was used to optimize.
3   | Katalip | 0.728 | 0.772 | Used an ensemble of a random forest regressor and a generalized gradient boosting regressor trained on MAIAC and MISR satellite data and GFS forecasts. GFS variables were selected based on a literature review and include those related to rainfall, precipitation, wind speed, humidity, wind direction, atmospheric stability, relative humidity, and temperature. The mean, 95th percentile, min, max, standard deviation, variance of AOD values were extracted for each grid ID, and data points were interpolated for each grid ID separately where missing. Optuna was used to tune the random forest regressor.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["How to Estimate Surface-Level PM2.5 Using MAIAC Aerosol Optical Depth Data"](https://www.drivendata.co/blog/predict-pm25-benchmark/)**
