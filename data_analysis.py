import pickle
import sys

import torch
from datetime import datetime, timedelta
import copy
# import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import seaborn as sns
# import watermark as watermark
from pylab import rcParams
import matplotlib.pyplot as plt
# from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Reading sensor data time series
    ts_df = pd.read_parquet('sensor_data_ts', engine='pyarrow')

    class_names = ['Normal', 'GasAlarm', 'GasHigh', 'GasLow', 'GasSTEL', 'GasTWA']

    ts_df = ts_df.rename(columns={'message_code_name': 'target'})
    ts_df.loc[ts_df['target'] == 'SensorAlarmMsg', 'target'] = class_names[1]
    ts_df.loc[ts_df['target'] == 'GasHighAlarmMsg', 'target'] = class_names[2]
    ts_df.loc[ts_df['target'] == 'GasLowAlarmMsg', 'target'] = class_names[3]
    ts_df.loc[ts_df['target'] == 'GasSTELAlarmMsg', 'target'] = class_names[4]
    ts_df.loc[ts_df['target'] == 'GasTWAAlarmMsg', 'target'] = class_names[5]

    ts_df = ts_df.sort_values(by=['datetime'])
    ts_df = ts_df.set_index(pd.DatetimeIndex(ts_df['datetime']), drop=True)

    n_train_days = 30
    ts_train_range = datetime(2019, 1, 1) + timedelta(days=n_train_days)
    print('training for {0} days of sensor data: '.format(n_train_days))
    ts_df = ts_df[ts_df['datetime'] < ts_train_range]

    # original data:  28 < latitude < 30 and  -99 < longitude < -98
    # Narrow down to one operation site

    ts_df = ts_df[ts_df['latitude'] < 28.7]
    ts_df = ts_df[ts_df['longitude'] > -98.8]
    print('No. records for San Antonio site: ', ts_df['target'].count())

    ts_df = ts_df.drop(labels=['datetime', 'latitude', 'longitude', 'message_code_id'],
                       axis=1)
    ts_columns = ts_df.columns

    print(ts_columns)
    print(ts_df.head())
    # print(ts_df.dtypes)

    # Data Preliminary Analysis

    # count number of distinct target values (must be less than or equal to 6)
    print('No. of distinct records for each target class: \n', ts_df['target'].value_counts())

    ax = sns.countplot(x=ts_df.target)
    ax.set_xticklabels(class_names)
    # fig1 = ax.get_figure()
    # fig1.savefig('target_class_counts.svg')

    normal_df = ts_df[ts_df['target'] == 'Normal'].drop(labels='target', axis=1)
    anomaly_df = ts_df[ts_df['target'] != 'Normal'].drop(labels='target', axis=1)

    test = normal_df[(normal_df['LEL'] != 0) | (normal_df['H2S'] != 0)]
    print(test.head(50))
    # print(anomaly_df.head())
    # with open('train_variables_{0}days.pkl'.format(n_train_days), 'wb') as f:
    #     pickle.dump([normal_df, anomaly_df], f)

    num_count_total = ts_df.target.count()
    # print('No. of records in orgiinal timeseries: ', num_count_total)
    smooth_df = ts_df.sample(int(num_count_total / 3))
    # print('sampled: ', smooth_df.target.count())
    # print('original: ', num_count_total)
    # print(smooth_df.head())

    cols_plot = ['CO', 'H2S', 'LEL', 'O2', 'target']
    feature_colors = {'GasHigh': 'red', 'GasLow': 'yellow', 'Normal': 'green', 'GasAlarm': 'red'}
    fig2, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 6), sharex='all', dpi=80)

    for i, cls in enumerate(cols_plot):
        if cls == cols_plot[-1]:
            axs[i].scatter(x=ts_df.index, y=ts_df[cls], c=ts_df[cls].map(feature_colors))
        else:
            axs[i].plot(ts_df[cls], linewidth=2)
            axs[i].set_ylabel(cls)

    # fig2.savefig('data_analysis_timeseries.svg')
    # plt.show()

    ### find null values
    df = ts_df
    columns = df.columns

    for col in columns:
        print("{0} values contains NULL: ".format(col), df[col].isnull().values.any())


if __name__ == '__main__':
    # input = sys.argv[1]
    main()
