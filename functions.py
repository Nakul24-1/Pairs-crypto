import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from stqdm import stqdm

maxlag = 18


@st.cache_data
def start_preprocess(stockA, stockB, start_date, end_date, lookback_window, n_periods):
    # convert date column to datetime
    stockA["date"] = pd.to_datetime(stockA["date"]).dt.date
    stockB["date"] = pd.to_datetime(stockB["date"]).dt.date
    # select rows with date on or after 2020-01-01
    stockA = stockA[stockA["date"] >= start_date]
    stockB = stockB[stockB["date"] >= start_date]
    # select rows with date before 2023-04-01
    stockA = stockA[stockA["date"] < end_date]
    stockB = stockB[stockB["date"] < end_date]
    # make date as index
    stockA.set_index("date", inplace=True)
    stockB.set_index("date", inplace=True)
    # select 3 columns which are date and symbol and close columns
    stockA = stockA[["symbol", "close"]]
    stockB = stockB[["symbol", "close"]]
    stockA, stockB = compute_rolling_returns(stockA, stockB, n_periods, lookback_window)
    return stockA, stockB


# Compute the rolling N-day returns for each stock
@st.cache_data
def compute_rolling_returns(stockA, stockB, n_periods, lookback_window):
    n = 1
    stockA[f"returns_{n}d"] = (stockA["close"] - stockA["close"].shift(n)) / stockA[
        "close"
    ].shift(n)
    stockB[f"returns_{n}d"] = (stockB["close"] - stockB["close"].shift(n)) / stockB[
        "close"
    ].shift(n)
    # Compute the trailing N-day means and standard deviations for each stock
    stockA[f"stddev_{n}d"] = (
        stockA[f"returns_{n}d"]
        .rolling(window=(20 if n == 1 else lookback_window))
        .std(ddof=1)
        .shift(1)
    )
    stockB[f"stddev_{n}d"] = (
        stockB[f"returns_{n}d"]
        .rolling(window=(20 if n == 1 else lookback_window))
        .std(ddof=1)
        .shift(1)
    )
    for n in n_periods:
        stockA[f"returns_{n}d"] = (stockA["close"] - stockA["close"].shift(n)) / stockA[
            "close"
        ].shift(n)
        stockB[f"returns_{n}d"] = (stockB["close"] - stockB["close"].shift(n)) / stockB[
            "close"
        ].shift(n)
        # Compute the trailing N-day means and standard deviations for each stock
        stockA[f"stddev_{n}d"] = (
            stockA[f"returns_{n}d"]
            .rolling(window=(20 if n == 1 else lookback_window))
            .std(ddof=1)
            .shift(1)
        )
        stockB[f"stddev_{n}d"] = (
            stockB[f"returns_{n}d"]
            .rolling(window=(20 if n == 1 else lookback_window))
            .std(ddof=1)
            .shift(1)
        )
        # Compute the z-scores for each stock for each N-day period
        stockA[f"zscore_{n}d"] = stockA[f"returns_{n}d"] / stockA[f"stddev_{n}d"]
        stockB[f"zscore_{n}d"] = stockB[f"returns_{n}d"] / stockB[f"stddev_{n}d"]
    return stockA, stockB


@st.cache_data
def make_new_df(stockA, stockB, n_periods):
    new_df = pd.DataFrame()
    for n in n_periods:
        new_df[f"zdiff_{n}d"] = stockA[f"zscore_{n}d"] - stockB[f"zscore_{n}d"]

    new_df["stddev_1d_BTC"] = stockA["stddev_1d"]
    new_df["stddev_1d_ETH"] = stockB["stddev_1d"]
    new_df["diff_1d"] = stockA["returns_1d"] - stockB["returns_1d"]
    return new_df


# find inv volatility based returns
@st.cache_data
def inv_volatility_returns1d(new_df, stockA, stockB):
    new_df["wtBTC"] = (
        new_df[["stddev_1d_BTC", "stddev_1d_ETH"]].max(axis=1) / new_df["stddev_1d_BTC"]
    )
    new_df["wtETH"] = (
        new_df[["stddev_1d_BTC", "stddev_1d_ETH"]].max(axis=1) / new_df["stddev_1d_ETH"]
    )
    # add the sum of wt
    new_df["wt_sum"] = new_df["wtBTC"] + new_df["wtETH"]
    # normalize the weights
    new_df["wtBTC"] = new_df["wtBTC"] / new_df["wt_sum"]
    new_df["wtETH"] = new_df["wtETH"] / new_df["wt_sum"]
    # drop the wt_sum column
    new_df.drop("wt_sum", axis=1, inplace=True)
    # Calculate future returns
    new_df["current_ret"] = (
        stockA["returns_1d"] * new_df["wtBTC"].shift(1)
        - stockB["returns_1d"] * new_df["wtETH"].shift(1)
    ).shift(-1)
    # replace NA values with 0
    new_df.fillna(0, inplace=True)
    return new_df


@st.cache_data
def calculate_positions(new_df, n, LongCap, ShortCap, age_limit):
    new_df[f"Long_Cap{n}"] = 0
    new_df[f"Short_Cap{n}"] = 0

    # Create column for position and age and aged for each lookback period
    new_df[f"position{n}"] = 0
    new_df[f"age{n}"] = 0
    new_df[f"aged{n}?"] = 0
    for i in stqdm(range(1, len(new_df))):
        # Check for a long signal
        if (new_df[f"zdiff_{n}d"].iloc[i] > LongCap) & (
            new_df[f"position{n}"].iloc[i - 1] == 1
        ):
            new_df[f"Long_Cap{n}"].iloc[i] = 1
        else:
            new_df[f"Long_Cap{n}"].iloc[i] = 0

        # Check for a short signal
        if (new_df[f"zdiff_{n}d"].iloc[i] < ShortCap) & (
            new_df[f"position{n}"].iloc[i - 1] == -1
        ):
            new_df[f"Short_Cap{n}"].iloc[i] = 1
        else:
            new_df[f"Short_Cap{n}"].iloc[i] = 0

        # Calculate Age of signal
        if (
            (
                (new_df[f"position{n}"].iloc[i - 1] == 0)
                and (new_df[f"signal_{n}"].iloc[i] != 0)
            )
            or (
                (new_df[f"position{n}"].iloc[i - 1] * new_df[f"signal_{n}"].iloc[i])
                == -1
            )
            or (
                (new_df[f"age{n}"].iloc[i - 1] == age_limit)
                and (new_df[f"signal_{n}"].iloc[i] != 0)
            )
        ):
            new_df[f"age{n}"].iloc[i] = 1

        elif (
            (
                (new_df[f"position{n}"].iloc[i - 1] == 0)
                and (new_df[f"signal_{n}"].iloc[i] == 0)
            )
            or (
                (new_df[f"Long_Cap{n}"].iloc[i] != 0)
                or (new_df[f"Short_Cap{n}"].iloc[i] != 0)
            )
            or ((new_df[f"age{n}"].iloc[i - 1] == age_limit))
        ):
            new_df[f"age{n}"].iloc[i] = 0
        else:
            new_df[f"age{n}"].iloc[i] = new_df[f"age{n}"].iloc[i - 1] + 1
        # Check if the signal is aged
        if new_df[f"age{n}"].iloc[i - 1] == age_limit:
            new_df[f"aged{n}?"].iloc[i] = 1
        else:
            new_df[f"aged{n}?"].iloc[i] = 0
        # Calculate the position
        # Excel formula =IF(AND(AE73=0,AF73=0,AG73=0,AI73=0),AJ72,AE73)
        if (
            (new_df[f"Long_Cap{n}"].iloc[i] == 0)
            & (new_df[f"Short_Cap{n}"].iloc[i] == 0)
            & (new_df[f"aged{n}?"].iloc[i] == 0)
            & (new_df[f"signal_{n}"].iloc[i] == 0)
        ):
            new_df[f"position{n}"].iloc[i] = new_df[f"position{n}"].iloc[i - 1]
        else:
            new_df[f"position{n}"].iloc[i] = new_df[f"signal_{n}"].iloc[i]

    return new_df


# Calculate the returns by multiplying the position by the returns
@st.cache_data
def calculate_returns(new_df, n):
    new_df[f"returns_{n}"] = new_df[f"current_ret"] * new_df[f"position{n}"]
    return new_df


# Make an equity curve starting from 100
@st.cache_data
def make_equity_curve(new_df, n):
    new_df[f"equity_curve{n}"] = 100 * (1 + new_df[f"returns_{n}"]).cumprod()
    return new_df


@st.cache_data
def zdiff_calculations(
    new_df, n_periods, thresholds, threshold_short, threshold_long, signal_reverse
):
    # loop over the lookback periods
    for period in n_periods:
        # get the zdiff and threshold values for the current period
        zdiff = new_df[f"zdiff_{period}d"]
        short_threshold, long_threshold = thresholds.get(
            period, (threshold_short, threshold_long)
        )
        # create a boolean mask based on the zdiff and threshold values
        short_signal = zdiff > short_threshold
        long_signal = zdiff < long_threshold

        if signal_reverse:
            # set the signal value where the condition is met
            new_df.loc[short_signal, f"signal_{period}"] = 1
            new_df.loc[long_signal, f"signal_{period}"] = -1
        else:
            new_df.loc[short_signal, f"signal_{period}"] = -1
            new_df.loc[long_signal, f"signal_{period}"] = 1

    return new_df


@st.cache_data
def drawdown(return_series):
    drawdown = (return_series.cummax() - return_series) / return_series.cummax()
    drawdown.max(), drawdown.idxmax()
    # print max drawdown and duration
    st.write(f"Max drawdown: {drawdown.max()}")
    # plot the drawdown
    st.line_chart(drawdown)


@st.cache_data
def sharpe_ratio(returns):
    return np.sqrt(365 * 24) * returns.mean() / returns.std()


@st.cache_data
def adf_test(series, title=""):
    st.write(f"Augmented Dickey-Fuller Test: {title}")
    result = adfuller(
        series.dropna(), autolag="AIC"
    )  # dropna() handles differenced data
    labels = [
        "ADF Test Statistic",
        "p-value",
        "# Lags Used",
        "Number of Observations Used",
    ]
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f"Critical Value ({key})"] = value
    st.write(out)  # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        st.write(
            "Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary"
        )
    else:
        st.write(
            "Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary "
        )


@st.cache_data
def grangers_causation_matrix(data, variables, test="ssr_chi2test", verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(
        np.zeros((len(variables), len(variables))), columns=variables, index=variables
    )
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(
                data[[r, c]], maxlag=maxlag, verbose=False
            )
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f"Y = {r}, X = {c}, P Values = {p_values}")
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "_y" for var in variables]
    return df
