# NASA Airathon - Winning Solution

## Data Processing - Training or Inference

Run on c6i.4xlarge or larger AWS instance (i.e. 16+ cores and most importantly, 1Gbps or higher network speed); consider going even higher if aiming to replicate the entire training set.

Use `rm -r inference` for a clean download and inference run.

All data will be loaded into tar files in the `cache/` folder for training if a complete donwload (2018-2021) occurs.

Fill in secure.txt with:
username=[your email username]
password=[your password]
for this server: https://rda.ucar.edu/datasets/ds084.1/#!access

Fill in infer.txt with:
start=2018,9,1
end=2021,9,1
to download all historical data.

Or, e.g.
start=2021,3,1
end=2021,3,7
to download a specifc date window for live inference.

```
pip install -r requirements.txt
python RunFeatures.py
```

~15 minute run time for one week of data (bandwidth constrained, with minimal cpu use).


## Inference

Ensure infer.txt is a subset of that used for data processing.

```
pip install -r requirements.txt
python Inference.py
```

[ predictions will be stored in inference_tg/stack1.csv within a few minutes, ~0.999 correlation with existing predictions ]


## Model Training - LightGBM

Run on r6i.4xlarge (16-core / 128 GB) or comparable

`rm -r clfs_tg` if you prefer a clean replication.

Ensure ['grid_metadata.csv', 'train_labels.csv', '.ipynb_checkpoints', 'submission_format.csv', 'no2_satellite_metadata_0AF3h09.csv'] in data_tg

Ensure ['assim.tar', 'tropomi-fine.tar', 'ifs.tar', 'gfs-5.tar'] are in cache

```
pip install -r requirements.txt
python Train.py
```

Allow an hour for training.

Models will be stored in clfs_tg

The features will be cached in cache [all_data_tg.pkl, submission.pkl] for use in neural network training.

A pure LightGBM submission will be saved in submission_tg/new.csv [expect 0.470 - 0.471 private leaderboard score, aka slightly winning, and ~0.999 correlation with new.csv provided]


## Model Training - Neural Network

`rm -r nn1` if you prefer a clean replication.

Ensure ['all_data_tg.pkl', 'submission_tg.pkl'] are in cache

```
pip install -r requirements.txt
python TrainNN.py 
```

All models were trained across hundreds of r6i.large spot instances. Distributed training across systems is supported (>100 aggregate hours is sufficient)

Each fold set is roughly 15 minutes, for ~600 models
   [3 locations * ~12 parameter choices * 20 fold rotations ]

Expect 150 hours of training time on a single CPU instance (invariant to core count, may be moderately faster on GPU; add 'accelerator="gpu"' to Trainer(...) to enable GPU). 


## Model Training - Ensemble

Ensure ['all_data_tg.pkl', 'submission_tg.pkl'] are in cache, and that clfs_tg and nn1 exist from model training

```
pip install -r requirements.txt
python TrainEnsemble.py 
```

Predictions will be stored in submissions_tg/stack1.csv
Expect ~0.477 private leaderboard score. 
~0.999 correlation between replication runs.


## Data Sources/Timing

Assim -- all data is converted from UTC to local time, and compiled up up through the relevant local day (see https://community.drivendata.org/t/clarification-on-features-dates-used-for-prediction/7215/3)

```
df.index = df.index.tz_localize('UTC').tz_convert(tz).floor('1d')
```

labels are also converted to local time (exactly midnight), and then matched to data sources

```
t = pd.to_datetime(labels.datetime).dt.tz_convert( tz_dict[location] )
t = t.dt.floor('1d').dt.tz_localize(None)
```

Tropomi -- ibid.

```
df.datetime = pd.to_datetime(df.datetime).dt.tz_convert(tz).dt.floor('1d')
```

GFS -- data is maintained for the original forecast times, in UTC; each label timestamp is rounded down to the nearest 6-hour window in UTC, and then the snapshots for 12 hours and 18 hours into the day are merged with labels

```
ldo = pd.to_datetime(labels.datetime).dt.floor('6h').dt.tz_localize(None)
df =  ....gfs.reindex([ldo
		   + datetime.timedelta(seconds = 60 * 60 * 12), ....
	  .... gfs.reindex([ldo
			+ datetime.timedelta(seconds = 60 * 60 * 18), ...
```

IFS -- each label timestamp is rounded down to the nearest 12-hour window, and then snapshots for 0 hrs and 12 hrs into the day are used, as above

```
ldo = pd.to_datetime(labels.datetime).dt.floor('12h').dt.tz_localize(None)
df = ...ifs.reindex([ldo + datetime.timedelta(seconds = 60 * 60 * 12), 
	 ...ifs.reindex([ldo + datetime.timedelta(seconds = 60 * 60 * 0)...
```

Most data sources use a ~3-day exponential moving average; inference performes near-identially by downloading ~10 days of trailing data.	