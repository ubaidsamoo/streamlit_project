import streamlit as st
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
import traceback

warnings.filterwarnings('ignore')


st.set_page_config(page_title='Time Series Analysis & Forecasting', layout='wide')


def load_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    data.index = pd.to_datetime(data.index)
    return data




def plot_line(ax, series, label=None):
    ax.plot(series.index, series.values, label=label)


def run_app():
    st.title('Time Series Analysis and Forecasting')
    st.markdown('Analyze and forecast stock Close/Volume using ARIMA and decomposition.')

    col1, col2 = st.columns([1, 3])

    with col1:
        ticker = st.selectbox('Select ticker', ['GOOGL', 'AAPL', 'AMZN', 'TSLA', 'MSFT'])
        period = st.selectbox('Period', ['1y', '2y', '3y', '5y', '10y'])
        st.button('Load Data')

    # Load data
    data = load_data(ticker, period=period)

    if data.empty:
        st.error('No data downloaded. Try a different ticker or period.')
        return

    # Part A: EDA
    st.header(' Exploratory Data Analysis')
    st.subheader('Data Preview & Summary')
    st.dataframe(data.tail())
    st.write('Data types:')
    st.write(data.dtypes)
    st.write('Summary statistics:')
    st.write(data.describe())

    st.subheader('Close over time')
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_line(ax, data['Close'], label='Close')
    ax.set_title(f'{ticker} Close Price')
    ax.legend()
    st.pyplot(fig)

    st.subheader('Histograms')
    fig2, axes = plt.subplots(1, 3, figsize=(15, 3))
    data['Open'].hist(ax=axes[0], bins=30)
    axes[0].set_title('Open')
    data['Close'].hist(ax=axes[1], bins=30)
    axes[1].set_title('Close')
    data['Volume'].hist(ax=axes[2], bins=30)
    axes[2].set_title('Volume')
    st.pyplot(fig2)

    # Part B: Moving Average
    st.header(' Moving Average and Trend')
    st.write('Purpose: A moving average smooths short-term fluctuations and highlights longer-term trends.')
    data['MA7'] = data['Close'].rolling(window=7).mean()
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(data.index, data['Close'], label='Close', alpha=0.6)
    ax3.plot(data.index, data['MA7'], label='7-day MA', color='orange')
    ax3.set_title('Close and 7-day Moving Average')
    ax3.legend()
    st.pyplot(fig3)

    # Part C: Decomposition
    st.header(' Time Series Decomposition')
    st.write('Decompose Close into Trend / Seasonal / Residual components using additive model.')
    # seasonal_decompose requires at least two full cycles (2 * period) in the data.
    close_ts = data['Close'].asfreq('D').interpolate()
    n_obs = close_ts.dropna().shape[0]
    preferred_period = 365
    # Choose a safe period: if there aren't 2 full years, lower the period to half the series length.
    if n_obs >= 2 * preferred_period:
        period = preferred_period
    else:
        period = max(2, n_obs // 2)

    if period < 2 or n_obs < 4:
        st.warning('Series too short to perform seasonal decomposition reliably. Skipping decomposition.')
    else:
        try:
            decomposition = seasonal_decompose(close_ts, model='additive', period=period)
            fig4 = plt.figure(figsize=(12, 8))
            ax1 = fig4.add_subplot(411)
            ax1.plot(decomposition.observed); ax1.set_ylabel('Observed')
            ax2 = fig4.add_subplot(412)
            ax2.plot(decomposition.trend); ax2.set_ylabel('Trend')
            ax3 = fig4.add_subplot(413)
            ax3.plot(decomposition.seasonal); ax3.set_ylabel('Seasonal')
            ax4 = fig4.add_subplot(414)
            ax4.plot(decomposition.resid); ax4.set_ylabel('Residual')
            st.pyplot(fig4)

            st.markdown(f'''
            - Trend: long-term movement in the series.
            - Seasonality (period={period}): repeating patterns captured by the decomposition.
            - Residual: remaining noise after removing trend and seasonality.
            ''')
        except ValueError as e:
            st.warning(f'Seasonal decomposition not possible: {e}. Skipping decomposition.')

    # Part D: Stationarity check and ARIMA
    st.header(' Stationarity Check and ARIMA Forecasting')
    st.subheader('ADF Test on Volume')
    vol_series = data['Volume'].asfreq('D').interpolate()
    adf_result = adfuller(vol_series.dropna())
    adf_output = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Used Lag': adf_result[2],
        'Number of Observations': adf_result[3]
    }
    st.write(adf_output)
    if adf_result[1] < 0.05:
        st.success('Volume series is likely stationary (reject null at 5% level).')
    else:
        st.info('Volume series is likely non-stationary (fail to reject null at 5% level).')

    # ARIMA modeling. We'll fit ARIMA(1,0,0) on Close as requested, and separately on Volume for forecasting Volume.
    st.subheader('ARIMA Modeling')
    st.write('Fitting ARIMA(1,0,0) on Close (as requested).')
    close_train = data['Close'].dropna()
    try:
        if len(close_train) < 3:
            st.warning('Not enough Close observations to fit ARIMA(1,0,0). Skipping Close ARIMA fit.')
            model_close = None
        else:
            model_close = ARIMA(close_train.astype(float).values, order=(1, 0, 0)).fit()
            st.write(model_close.summary())
    except Exception as e:
        st.error(f'ARIMA on Close failed: {e}')

    # Forecasting Volume: we will perform a simple backtest (last 30 days) to compute MAE, then forecast next 30 days.
    st.subheader('Forecasting Volume (30 days)')
    n_forecast = 30
    vol = data['Volume'].dropna()
    if len(vol) < 60:
        st.warning('Not enough Volume data to perform reliable train/test split for backtesting.')
        train_vol = vol
        test_vol = pd.Series(dtype=float)
    else:
        train_vol = vol[:-n_forecast]
        test_vol = vol[-n_forecast:]

    def to_array(x):
        return x.values if hasattr(x, 'values') else np.asarray(x)

    try:
        if train_vol.empty or len(train_vol) < 3:
            st.warning('Not enough Volume observations to fit ARIMA(1,0,0). Skipping Volume ARIMA fit.')
            model_vol = None
        else:
            model_vol = ARIMA(train_vol.astype(float).values, order=(1, 0, 0)).fit()
        # Forecast for test period
        mae = None
        if (model_vol is not None) and (not test_vol.empty):
            pred_test = model_vol.forecast(steps=n_forecast)
            pred_arr = to_array(pred_test)
            mae = mean_absolute_error(test_vol.values, pred_arr)
            st.write(f'MAE on last {n_forecast} days: {mae:.2f}')
        else:
            st.write('Skipping backtest MAE due to small sample size or skipped fit.')

        # Retrain on full volume series and forecast next 30 days
        if vol.empty or len(vol) < 3:
            st.warning('Not enough full Volume data to produce a forecast.')
            model_vol_full = None
            future_forecast = pd.Series(dtype=float)
        else:
            model_vol_full = ARIMA(vol.astype(float).values, order=(1, 0, 0)).fit()
            future_forecast = model_vol_full.forecast(steps=n_forecast)

        # Plot historical and forecast
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        ax5.plot(vol.index, vol.values, label='Historical Volume')
        future_index = pd.date_range(start=vol.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
        future_arr = to_array(future_forecast)
        ax5.plot(future_index, future_arr, label='Forecast (30 days)', color='red')
        ax5.set_title('Volume: Historical and 30-day Forecast')
        ax5.legend()
        st.pyplot(fig5)

        if mae is not None:
            st.write('Model evaluation:')
            st.write('- MAE on holdout shown above. Lower is better.')
            if (mae > vol.mean()).all():
                st.warning('MAE is large relative to average Volume — model may be underfitting or data is highly volatile.')
            else:
                st.success('MAE is reasonable relative to average Volume.')

    except Exception as e:
        st.error(f'ARIMA modeling on Volume failed: {e}')
        st.text('Full traceback:')
        st.text(traceback.format_exc())

    st.markdown('''
    - Comment: ARIMA(1,0,0) is a simple AR(1) model. If the series shows seasonality or non-stationarity, consider differencing (d>0) or SARIMA models.
    - Overfitting vs Underfitting: If residuals remain structured and MAE is high, the model underfits; if in-sample fit is excellent but out-of-sample error is poor, it may be overfitting.
    ''')

    st.header('Done')
    st.write('You can change the `ticker` and `period` and reload to explore different data.')


if __name__ == '__main__':
    run_app()
