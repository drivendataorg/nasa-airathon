import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;sns.set()
import os
import datetime
from collections import defaultdict
from loguru import logger

OUTPUT_PATH = 'submission'
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_PATH = os.path.join(OUTPUT_PATH, TIMESTAMP)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

logger.add(os.path.join(OUTPUT_PATH, 'logs.log'))

files = [
    # "D:/Repositories/NASA_Airathon/logs/2022-03-07-19-47/submission_2022-03-07-19-47.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-12-10-21/submission_2022-03-12-10-21.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-08-17-19/submission_2022-03-08-17-19.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-02-23-12-59/submission_2022-02-23-12-59.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-14/submission_2022-03-13-19-14.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-31/submission_2022-03-13-19-31.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-40/submission_2022-03-13-19-40.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-23-28/submission_2022-03-13-23-28.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-10-56/submission_2022-03-14-10-56.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-11-19/submission_2022-03-14-11-19.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-11-41/submission_2022-03-14-11-41.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-22-37/submission_2022-03-14-22-37.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-22-53/submission_2022-03-14-22-53.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-26/submission_2022-03-15-19-26.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-44/submission_2022-03-15-19-44.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-58/submission_2022-03-15-19-58.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-21-21/submission_2022-03-15-21-21.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-23-13/submission_2022-03-15-23-13.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-00-38/submission_2022-03-16-00-38.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-11-08/submission_2022-03-16-11-08.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-12-31/submission_2022-03-16-12-31.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-13-37/submission_2022-03-16-13-37.csv",
    # "D:/Repositories/NASA_Airathon/submission.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-17-13-52/submission_2022-03-17-13-52.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-17-14-45/submission_2022-03-17-14-45.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-17-17-36/submission_2022-03-17-17-36.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-17-14-45/submission_2022-03-17-14-45.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-13-37/submission_2022-03-16-13-37.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-01-36/submission_2022-03-18-01-36.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-08-00/submission_2022-03-18-08-00.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-09-48/submission_2022-03-18-09-48.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-09-48/submission_avgoof.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-12-57/submission_2022-03-18-12-57.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-18-12-57/l0avg_submission_2022-03-18-12-57.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-19-14-04/submission_2022-03-19-14-04.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-19-14-04/l0avg_submission_2022-03-19-14-04.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-19-14-44/submission_2022-03-19-14-44.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-19-14-44/l0avg_submission_2022-03-19-14-44.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-19-14-44/xgb_tuned_none.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-10-41/submission_2022-03-20-10-41.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-11-39/submission_2022-03-20-11-39.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-14-48/submission_2022-03-20-14-48.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-17-42/submission_2022-03-20-17-42.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-18-01/submission_2022-03-20-18-01.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-18-10/submission_2022-03-20-18-10.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-20-18-37/submission_2022-03-20-18-37.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-21-15-43/submission_2022-03-21-15-43.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-22-02-13/submission_2022-03-22-02-13.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-22-02-31/submission_2022-03-22-02-31.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-25-02-04/submission_2022-03-25-02-04.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-26-13-05/submission_2022-03-26-13-05.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-26-15-15/submission_2022-03-26-15-15.csv",
    "D:/Repositories/NASA_Airathon/submission/2022-03-26-17-31/submission_2022-03-26-17-31.csv",
    # "D:/Repositories/NASA_Airathon/submission/2022-03-22-03-33/submission_2022-03-22-03-33.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-28-20-18/submission_2022-03-28-20-18.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-28-22-08/submission_2022-03-28-22-08.csv",
    "D:/Repositories/NASA_Airathon/submission/2022-03-28-23-36/submission_2022-03-28-23-36.csv",
]

scores = [
    # 0.6967,
    # 0.6866,
    # 0.6954,
    # 0.6963,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # 0.7453,
    # "N/A",
    # "N/A",
    # 0.7429,
    # "N/A",
    # "N/A",
    # 0.7385,
    # "N/A",
    # "N/A",
    # 0.6890,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # 0.7366,
    # 0.7321,
    # 0.7156,
    # "N/A",
    # "N/A",
    # 0.7416,
    # "N/A",
    # "N/A",
    # 0.7391,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # 0.7224,
    # "N/A",
    # 0.7381,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # 0.7413,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    "N/A",
    "N/A",
]

preds = defaultdict()
fig, axes = plt.subplots(1, 2)
for i, filename in enumerate(files):
    logger.info(f"Reading {filename} which has score {scores[i]}...")
    sub = pd.read_csv(filename)
    sns.distplot(
        sub['value'], 
        label=f"{os.path.basename(filename)}_{str(scores[i])}",
        hist=False,
        ax=axes[0]
    )
    preds[f"{os.path.basename(filename)}_{str(scores[i])}"] = sub['value'].tolist()

preds = pd.DataFrame(preds)
sns.heatmap(preds.corr(), annot=True, ax=axes[1])

sub['value'] = preds.mean(axis=1)
sub.to_csv(os.path.join(OUTPUT_PATH, f'submission_{TIMESTAMP}.csv'), index=False)

sns.distplot(
    sub['value'], 
    label="Average",
    hist=False, 
    ax=axes[0])
axes[0].legend()
plt.show()
