# Solution - NASA Airathon: Predict Air Quality (Trace Gas Track)

Username: sukantabasu

## Repo organization
```
.
├── data
│   ├── airathon
│   │   ├── processed
│   │   │   ├── test
│   │   │   │   ├── GFS
│   │   │   │   ├── OMI
│   │   │   │   └── STN
│   │   │   └── train
│   │   │       ├── GFS
│   │   │       ├── OMI
│   │   │       └── STN
│   │   └── raw
│   │       ├── GFS
│   │       │   ├── DEL
│   │       │   ├── LAX
│   │       │   └── TAI
│   │       └── STN
│   └── singleday
│       ├── processed
│       │   └── test
│       │       ├── GFS
│       │       ├── OMI
│       │       └── STN
│       └── raw
│           ├── GFS
│           │   ├── VAR1
│           │   └── VAR2
│           └── OMI
├── forecast
│   ├── airathon
│   └── singleday
├── model
├── notebooks
│   ├── airathon
│   └── singleday
└── readme_files

```
## Quick Overview

**README**: This repo contains two readme files inside the **[readme_files](readme\_files)** folder: **[README\_airathon.md](readme_files/README_airathon.md)** and **[README\_singelday.md](readme_files/README_singleday.md)**. The first file describes all the necessary steps to reproduce the results of the NASA airathon. Whereas, the second file summarizes the required steps to forecast NO2 at a given location for **one** specific day.

**Data**: The **[data](data)** directory contains both raw and processed data. There are separate sub-directories for GFS, OMI, and station (STN) data. 

**Notebooks**: Several Jupyter notebooks are included in the **[notebooks](notebooks)** folder. The filename of each notebook starts with a number. As such, the notebooks should be run sequentially. 

**Models**: The LightGBM model is utilized in this hackathon. The pickle package is used to save the model parameters in the model sub-directory. 

**Forecast**: The **[forecast](forecast)** directory contains both airathon and single day forecasts. 

## License 
MIT license 
