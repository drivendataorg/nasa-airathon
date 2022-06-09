# NASA Airathon Predict Air Quality Trace Gas Track

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

#Satellite data (OMI)
data_omi = omi[(omi.grid_id==obs.grid_id)&(omi.file_datetime_end<obs.obs_datetime_end)].tail(1)

#Satellite data (TROPOMI)
data = tropomi[(tropomi.grid_id==obs.grid_id)&(tropomi.file_datetime_end<obs.obs_datetime_end)].tail(1)

#GFS data
data_gfs = data_gfs[(data_gfs.valid_time<obs.obs_datetime_end)]

#GEOS data
#At midday (UTC), hindcasts are generated for last 24 hours.
#Data past midday is only available after the following day's midday
#So data generation time is:
df_geos['current_day_noon'] = df_geos.data_datetime.dt.floor('D')+pd.DateOffset(hours=12)
df_geos.loc[df_geos.data_datetime<df_geos.current_day_noon,'data_generation_datetime']=\
df_geos.loc[df_geos.data_datetime<df_geos.current_day_noon].current_day_noon
df_geos.loc[df_geos.data_datetime>df_geos.current_day_noon,'data_generation_datetime'] =\
df_geos.loc[df_geos.data_datetime>df_geos.current_day_noon].current_day_noon+pd.DateOffset(hours=24)

#and data available for an observation is:
data_geos =  data_geos[(data_geos.data_generation_datetime<obs.obs_datetime_end)]
```



