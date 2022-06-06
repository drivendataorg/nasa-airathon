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
1   | [\_NQ\_](https://www.drivendata.org/users/_NQ_/) | 0.525 | 0.477 | Used an ensembled LGBM and neural net, with purged k-fold cross validation with a 30-day gap.
2   | [karelds](https://www.drivendata.org/users/karelds/) | 0.513 | 0.469 | Used an ensemble of 30 LGBM models using OMI, TOPOMI, and GFS data and 5-fold cross-validation without shuffling.
3   | [sukantabasu](https://www.drivendata.org/users/sukantabasu/) | 0.486 | 0.445 | Used an ensemble of LGBM models with OMI and GFS data and 10-fold cross-validation with halving random grid search.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["How to Estimate Surface-Level NO2 Using OMI Column NO2 Data"](https://www.drivendata.co/blog/predict-no2-benchmark/)**
