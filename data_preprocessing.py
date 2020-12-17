from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def main():
    # Reading sensor data time series
    ts_df = pd.read_parquet('sensor_data_ts', engine='pyarrow')

    class_names = ['Normal', 'GasAlarm', 'GasHigh', 'GasLow', 'GasSTEL', 'GasTWA']

    ts_df = ts_df.rename(columns={'message_code_name': 'target'})

    ts_df = ts_df.sort_values(by=['datetime'])
    ts_df = ts_df.set_index(pd.DatetimeIndex(ts_df['datetime']), drop=True)

    # original data:  28 < latitude < 30 and  -99 < longitude < -98
    # Narrow down to one operation site

    ts_df = ts_df[ts_df['latitude'] < 28.7]
    ts_df = ts_df[ts_df['longitude'] > -98.8]
    print('No. records for San Antonio site: ', len(ts_df.index))

    ts = ts_df.drop(labels=['datetime', 'latitude', 'longitude'], axis=1)

    ts_resampled = ts.groupby(pd.Grouper(freq='15T')).max().dropna()

    ts_resampled.loc[ts_resampled['message_code_id'] == 64, 'target'] = 'Normal'
    ts_resampled.loc[ts_resampled['message_code_id'] == 10, 'target'] = 'Normal'
    ts_resampled.loc[ts_resampled['message_code_id'] == 66, 'target'] = 'GasLow'
    ts_resampled.loc[ts_resampled['message_code_id'] == 67, 'target'] = 'GasHigh'
    ts_resampled.loc[ts_resampled['message_code_id'] == 68, 'target'] = 'GasTWA'
    ts_resampled.loc[ts_resampled['message_code_id'] == 69, 'target'] = 'GasSTEL'
    ts_resampled.loc[ts_resampled['message_code_id'] == 100, 'target'] = 'GasAlarm'
    print(ts_resampled.head(30))
    ts_resampled = ts_resampled.drop(labels=['message_code_id'], axis=1)

    ts = ts.drop(labels=['message_code_id'], axis=1)

    print(len(ts_resampled.index))
    print(len(ts.index))
    print(ts.loc[datetime(2019, 1, 1, 21, 40):].head(30))

    ts.to_parquet('timeseries_main.parquet')
    ts_resampled.to_parquet('timeseries_resampled.parquet')

    # n_train_days = 30
    # ts_train_range = datetime(2019, 1, 1) + timedelta(days=n_train_days)
    # print('data time range for {0} days of sensor data: '.format(n_train_days))
    # ts = ts.loc[ts.index < ts_train_range]
    #
    # ts_columns = ts.columns
    #
    # print(ts_columns)
    # print(ts.head())
    # print(ts_df.dtypes)

    # Data Preliminary Analysis

    # count number of distinct target values (must be less than or equal to 6)
    print('No. of distinct records for each target class: \n', ts['target'].value_counts())
    print('No. of distinct records for each target class in resampled time series: \n',
          ts_resampled['target'].value_counts())

    ax = sns.countplot(x=ts.target)
    ax.set_xticklabels(class_names)

    normal_df = ts[ts['target'] == 'Normal'].drop(labels='target', axis=1)
    anomaly_df = ts[ts['target'] != 'Normal'].drop(labels='target', axis=1)

    test = normal_df[(normal_df['LEL'] != 0) | (normal_df['H2S'] != 0)]

    cols_plot = ['CO', 'H2S', 'LEL', 'O2', 'target']
    feature_colors = {'GasHigh': 'red', 'GasLow': 'yellow', 'Normal': 'green', 'GasAlarm': 'red', 'GasSTEL': 'red',
                      'GasTWA': 'red'}
    fig2, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 6), sharex='all', dpi=80)

    for i, cls in enumerate(cols_plot):
        if cls == cols_plot[-1]:
            print(cls)
            axs[i].scatter(x=ts_resampled.index, y=ts_resampled[cls], c=ts_resampled[cls].map(feature_colors))
        else:
            axs[i].plot(ts_resampled[cls], linewidth=2)
            axs[i].set_ylabel(cls)

    plt.show()
    ### find null values
    # df = ts
    # columns = df.columns
    #
    # for col in columns:
    #     print("{0} values contains NULL: ".format(col), df[col].isnull().values.any())


if __name__ == '__main__':
    # input = sys.argv[1]
    main()
