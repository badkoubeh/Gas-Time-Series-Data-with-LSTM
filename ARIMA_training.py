import sys
import warnings
from datetime import datetime, timedelta
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

warnings.simplefilter('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def main():
    ts = pd.read_parquet('timeseries_resampled.parquet')

    # print(test_stationarity(ts['O2']))
    # ts = ts.groupby(pd.Grouper(freq='15T')).max()
    # ts_train = ts_train.resample('30T').mean().ffill()
    train_end_date = datetime(2019, 7, 1)
    ts_train = ts[:train_end_date]

    # remove outliers
    # ts_train = ts_train[ts_train.between(ts_train.quantile(.05), ts_train.quantile(.95))]

    # normal_df = ts_train[ts_train['target'] == 'Normal'].drop(labels='target', axis=1)
    # anomaly_df = ts_train[ts_train['target'] != 'Normal'].drop(labels='target', axis=1)

    # x_train, x_val = train_test_split(ts_train, test_size=0.15, random_state=RANDOM_SEED)

    x_train = ts_train['H2S'].loc[:datetime(2019, 6, 1)]
    x_val = ts_train['H2S'].loc[datetime(2019, 6, 1):]

    print(len(ts_train.index))
    print(len(x_train.index))
    print(len(x_val.index))
    # print(x_train.tail(30))

    # stepwise_fit = auto_arima(x_train, start_p=1, start_q=1,
    #                           max_p=3, max_q=3, m=12,
    #                           start_P=0, seasonal=True,
    #                           d=None, D=1, trace=True,
    #                           error_action='ignore',  # we don't want to know if an order does not work
    #                           suppress_warnings=True,  # we don't want convergence warnings
    #                           stepwise=True)  # set to stepwise
    #
    # # To print the summary
    # stepwise_fit.summary()
    model_fit = ARIMA(x_train, order=(3, 0, 2)).fit(disp=0)



    # model_fit = Holt(ts_col, initialization_method="estimated").fit()
    # model_fit = ExponentialSmoothing(ts_col, initialization_method="estimated").fit()

    # print(model_fit.summary())
    num_forecast_steps = len(x_val.index)
    forecast_res, _, conf_int = model_fit.forecast(num_forecast_steps, alpha=0.05)
    # # forecast_res = ARIMAResults(model_fit).forecast(num_forecast_steps)
    # # print(model_fit.predict(num_forecast_steps))
    #
    print(forecast_res)
    #
    forecast_series = pd.Series(forecast_res, index=x_val.index)
    # print(forecast_res)
    lower_series = pd.Series(conf_int[:, 0], index=x_val.index)
    upper_series = pd.Series(conf_int[:, 1], index=x_val.index)
    output = pd.DataFrame({'H2S': forecast_series})

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(x_train, label='training')
    plt.plot(x_val, label='actual')
    plt.plot(forecast_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    # result = seasonal_decompose(x_train,)
    # result.plot()
    # plt.show()


if __name__ == '__main__':
    main()
