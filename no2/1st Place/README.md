# NASA Airathon - Winning Solution

## Getting started

Install requirements with `pip install -r requirements.txt`.

Save the competition data files to the `data_tg` directory. This includes:
- `grid_metadata.csv`
- `submission_format.csv`
- `no2_satellite_metadata_0AF3h09`
- `train_labels.csv`

It's recommended to run on c6i.4xlarge or larger AWS instance (i.e. 16+ cores and most importantly, 1Gbps or higher network speed); consider going even higher if aiming to replicate the entire training set.

Notes on file organization:
- `tg` stands for "trace gas"
- `data_tg` contains static competition files
- `clfs_tg` contains trained classifiers
- `submissions_tg` contains submissions from `Train.py` and `TrainEnsemble.py`
- `nn1` contains all NN model iterations
- `inference` contains downloaded features from `RunFeatures.py`
- `inference_tg` contains outputted prediction files from `Inference.py`


## Feature creation

Fill in `secure.txt` with:
```
username=[your email username]
password=[your password]
```
for this server: https://rda.ucar.edu/datasets/ds084.1/#!access

Fill in `infer.txt` with the desired time period. To download all the data needed for the competition, use:
```
start=2018,9,1
end=2021,9,1
```

Then run `python RunFeatures.py`. This will download the data to the `inference` folder as follows:
```
inference/
  assim/
    co_dl_2018_08.pkl
    ...
  gfs-5/
    gfs.0p25.2018082200.f000.grib2
    ...
  ifs/
    ec.oper.fc.sfc.128_057_uvb.regn1280sc.20180822.nc
    ...
  tropomi-fine/
    20180908T054431_tropomi_tpe_0.nc
    ...
```

All data will be loaded into tar files in the `cache/` folder for training if a complete donwload (2018-2021) occurs.

Expect ~15 minute run time for one week of data (bandwidth constrained, with minimal cpu use).

## Training

First, ensure [`assim.tar`, `tropomi-fine.tar`, `ifs.tar`, `gfs-5.tar`] are in the `cache` directory. This will be the case if you ran `RunFeatures.py` with an `infer.txt` file that spans from 2018 to 2021.

For model training, you have two options.

### LightGBM Model

Run `python Train.py` which will train the LightGBM + ElasticNet model and output predictions for the competition. This script outputs trained classifiers to the `clfs_tg` directory.

Allow an hour for training if running on r6i.4xlarge (16-core / 128 GB) or comparable.

The features will be cached in `cache/all_data_tg.pkl` and `cache/submission.pkl` for use in neural network training.

A pure LightGBM submission will be saved in `submission_tg/new.csv` [expect 0.470 - 0.471 private leaderboard score, aka slightly winning, and ~0.999 correlation with new.csv provided]

OR

### Ensemble Model

Run the following three training scripts:

1. `python Train.py`: trains LightGBM + ElasticNet model

See details above in [LightGBM](#lightgbm-model)

2. `python TrainNN.py`: trains neural network 

Ensure `all_data_tg.pkl` and `submission_tg.pkl` are in the `cache directory`.

All models were trained across hundreds of r6i.large spot instances. Distributed training across systems is supported (>100 aggregate hours is sufficient)

Each fold set is roughly 15 minutes, for ~600 models
   [3 locations * ~12 parameter choices * 20 fold rotations ]

Expect 150 hours of training time on a single CPU instance (invariant to core count, may be moderately faster on GPU; add 'accelerator="gpu"' to Trainer(...) to enable GPU). 

3. `python TrainEnsemble.py`: trains ensemble of the previous two models

Ensure `all_data_tg.pkl`, `submission_tg.pkl` are in cache, and that the `clfs_tg` and `nn1` directories exist from the previous two steps.

Predictions will be stored in `submissions_tg/stack1.csv`. 

Expect ~0.477 private leaderboard score. ~0.999 correlation between replication runs.

## Out of sample inference

Ensure `infer.txt` is a subset of the features that were previously pulled. If this is not the case, follow the steps in the [feature creation](#feature-creation) section. Be sure to set `infer.txt` to the desired new time period.

Then run `python Inference.py` which will output the submissions to `inference_tg` folder.

The inference script outputs predictions for two different models. The LightGBM model is faster and simpler but does not generalize as well as the ensembled model.
```
inference_tg/
	new.csv -> predictions from LightGBM + ElasticNet model
	stack1.csv -> predictions from ensemble model (LightGBM + ElasticNet and NeuralNet)
```	

Predictions will be output within a few minutes, ~0.999 correlation across runs.

## Consideration of time

#### Assim

All data is converted from UTC to local time, and compiled up up through the relevant local day (see https://community.drivendata.org/t/clarification-on-features-dates-used-for-prediction/7215/3)

```
df.index = df.index.tz_localize('UTC').tz_convert(tz).floor('1d')
```

Labels are also converted to local time (exactly midnight), and then matched to data sources

```
t = pd.to_datetime(labels.datetime).dt.tz_convert( tz_dict[location] )
t = t.dt.floor('1d').dt.tz_localize(None)
```

#### Tropomi

ibid.

```
df.datetime = pd.to_datetime(df.datetime).dt.tz_convert(tz).dt.floor('1d')
```

#### GFS

Data is maintained for the original forecast times, in UTC; each label timestamp is rounded down to the nearest 6-hour window in UTC, and then the snapshots for 12 hours and 18 hours into the day are merged with labels

```
ldo = pd.to_datetime(labels.datetime).dt.floor('6h').dt.tz_localize(None)
df =  ....gfs.reindex([ldo
		   + datetime.timedelta(seconds = 60 * 60 * 12), ....
	  .... gfs.reindex([ldo
			+ datetime.timedelta(seconds = 60 * 60 * 18), ...
```

#### IFS

Each label timestamp is rounded down to the nearest 12-hour window, and then snapshots for 0 hrs and 12 hrs into the day are used, as above

```
ldo = pd.to_datetime(labels.datetime).dt.floor('12h').dt.tz_localize(None)
df = ...ifs.reindex([ldo + datetime.timedelta(seconds = 60 * 60 * 12), 
	 ...ifs.reindex([ldo + datetime.timedelta(seconds = 60 * 60 * 0)...
```

Most data sources use a ~3-day exponential moving average; inference performes near-identially by downloading ~10 days of trailing data.