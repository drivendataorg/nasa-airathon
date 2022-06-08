# NASA Airathon Predict Air Quality Particulate Track

Install requirements in a Python 3 environment.

Run [train.ipynb](train.ipynb) to train and [test.ipynb](test.ipynb) to generate predictions.

Processed data and pretrained models are provided with the code so you may skip the preprocessing steps and directly run the training/ inference scripts.

[prod_demo.ipynb](prod_demo.ipynb) show an example of how to generate predictions for a single location on a specific date.

You will need an account to download GFS data. You can sign up for one at https://rda.ucar.edu/

## About time

Please refer to [src/data/create_dataset.py](src/data/create_dataset.py) for context 

```python

#Label datetime indicates the start of a 24-hour period
#So observation datatime start and end is:
df_labels['obs_datetime_start']=pd.to_datetime(df_labels.datetime).dt.tz_localize(None)
df_labels['obs_datetime_end'] = df_labels['obs_datetime_start'] + pd.DateOffset(hours=24)

#Given an observation(obs) for a particular grid_id and datetime, 
#the following filters ensure only forecasts available before observation end time 
#are used for modeling or to generate NO2 predictions.

#Satellite data (MAIAC)
data_maiac = maiac[(maiac.grid_id==obs.grid_id)&(maiac.file_datetime_end<obs.obs_datetime_end)]

#GFS data
data_gfs = data_gfs[(data_gfs.valid_time<obs.obs_datetime_end)]

```

## License
[MIT](LICENSE)



