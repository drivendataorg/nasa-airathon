import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('train_features.csv')

for col in train_data.columns:
    sns.histplot(train_data[col])
    plt.show()
