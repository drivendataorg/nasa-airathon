[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/nasa-aq-banner-web.jpg)

# NASA Airathon: Predict Air Quality - Trace Gas Track

## Goal of the Competition
Air pollution is one of the greatest environmental threats to human health. It can result in heart and chronic respiratory illness, cancer, and premature death. Currently, no single satellite instrument provides ready-to-use, high resolution information on surface-level air pollutants. This gap in information means that millions of people cannot take daily action to protect their health.

The goal of this track of the competition was to use remote sensing data and other geospatial data sources to develop models for estimating daily levels of NO2 with high spatial resolution. Successful models can provide critical data to help the public take action to reduce their exposure to air pollution.

## What's in this Repository

This repository contains code from winning competitors in the [NASA Airathon: Predict Air Quality - Track Gas Track](https://www.drivendata.org/competitions/91/competition-air-quality-no2/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | [\_NQ\_](https://www.drivendata.org/users/_NQ_/) | 0.525 | 0.477 | Used an ensembled LGBM and neural net, with purged k-fold cross validation with a 30-day gap, and fold and parameter rotation across each training iteration. Parameters were selected intra-fold across every regularization option. The most important parameter was `linear_tree`, which fits a linear regression inside each leaf of the tree. This worked well for this competition given the strong (~0.4-0.6 in many cases) feature correlations with the target and limited set of data points. Models were trained independently for all locations, and features were extracted using concentric circles around each grid point (0.05, 0.1, ... 5 degree radius). A final ensemble with a neural network added an additional 0.5% to the model's performance.
2   | [karelds](https://www.drivendata.org/users/karelds/) | 0.513 | 0.469 | Used an ensemble of 30 LGBM models using OMI and TROPOMI satellite data, GFS forecasts, and GEOS-CF NO2 hindcasts. GFS variables were selected based on literature and availability and included those related to air humidity, soil temperature, soil humidity, air temperature, wind velocity, wind direction, and rainfall/precipitation. GFS forecasts from up to 3 days preceding the date of interest were used with different lookback periods. Separate models were trained for each location using 5-fold (without shuffle) cross-validation instead of time-based splits. 
3   | [sukantabasu](https://www.drivendata.org/users/sukantabasu/) | 0.486 | 0.445 | Used an ensemble of LGBM models with OMI satellite data and GFS forecasts. GFS variables include those related to wind speed and direction, air temperature, planetary boundary layer height, heat flux, and ventilation rate. After extreme NO2 events were excluded, parameters were tuned using 10-fold cross validation and halving random grid search. Overall training data was split into 90% and 10% segments for training and validation respectively, and to avoid temporal leakage, the validation segment samples were selected to be consecutive in time. Using these segments and early stopping as regularization, 100 different tuned models were created and ensembled.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["How to Estimate Surface-Level NO2 Using OMI Column NO2 Data"](https://www.drivendata.co/blog/predict-no2-benchmark/)**
