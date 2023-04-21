import streamlit as st
import pandas as pd
import numpy as np

# perform dickey-fuller test on stockA with StockB
from statsmodels.tsa.stattools import adfuller


def adf_test(series, title=""):
    print(f"Augmented Dickey-Fuller Test: {title}")
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
    print(out.to_string())  # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        print(
            "Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary"
        )
    else:
        print(
            "Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary "
        )


# Perform Granger Causality Test to check if Stock A causes Stock B
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 18


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


new_df = pd.DataFrame()

# Load the stock data into two separate dataframes, stockA and stockB
stockB = pd.read_csv("D:\Downloads\Gemini_ETHUSD_1h.csv")
stockA = pd.read_csv("D:\Downloads\Gemini_BTCUSD_1h.csv")
# convert date column to datetime
stockA["date"] = pd.to_datetime(stockA["date"])
stockB["date"] = pd.to_datetime(stockB["date"])
# select rows with date after 2020-01-01
stockA = stockA[stockA["date"] > "2020-01-01"]
stockB = stockB[stockB["date"] > "2020-01-01"]
# make date as index
stockA.set_index("date", inplace=True)
stockB.set_index("date", inplace=True)
# select 3 columns which are date and symbol and close columns
stockA = stockA[["symbol", "close"]]
stockB = stockB[["symbol", "close"]]

# take the lookback window size as input and the list of N-day periods to compute z-scores

lookback_window = st.number_input(
    "Lookback window size", min_value=1, max_value=100, value=60, step=1
)
n_periods = [1, 5, 10, 20]

# Compute the rolling N-day returns for each stock
for n in n_periods:
    stockA[f"returns_{n}d"] = (stockA["close"] - stockA["close"].shift(n)) / stockA[
        "close"
    ].shift(n)
    stockB[f"returns_{n}d"] = (stockB["close"] - stockB["close"].shift(n)) / stockB[
        "close"
    ].shift(n)

# Compute the trailing N-day means and standard deviations for each stock
for n in n_periods:
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
for n in n_periods:
    stockA[f"zscore_{n}d"] = stockA[f"returns_{n}d"] / stockA[f"stddev_{n}d"]
    stockB[f"zscore_{n}d"] = stockB[f"returns_{n}d"] / stockB[f"stddev_{n}d"]

for n in n_periods:
    new_df[f"zdiff_{n}d"] = stockA[f"zscore_{n}d"] - stockB[f"zscore_{n}d"]

new_df["stddev_1d_BTC"] = stockA["stddev_1d"]
new_df["stddev_1d_ETH"] = stockB["stddev_1d"]
new_df["diff_1d"] = stockA["returns_1d"] - stockB["returns_1d"]


df = pd.DataFrame()
df["stockA"] = stockA["close"]
df["stockB"] = stockB["close"]
# drop na values
df.dropna(inplace=True)
# take difference of log of close price
df["diff"] = np.log(df["stockA"]) - np.log(df["stockB"])
# Take ratio of log of close price
df["ratio"] = np.log(df["stockA"]) / np.log(df["stockB"])

# check if the difference of log of close price is stationary
# adf_test(df["diff"])

# check if the ratio of log of close price is stationary
# adf_test(df["ratio"])

# check if stockA causes stockB
# grangers_causation_matrix(df, variables=["stockA", "stockB"])


# define the thresholds for each lookback period
thresholds = {5: (1, -1), 10: (1, -1), 20: (1, -1)}

# loop over the lookback periods
for period in n_periods:
    # get the zdiff and threshold values for the current period
    zdiff = new_df[f"zdiff_{period}d"]
    short_threshold, long_threshold = thresholds.get(period, (1, -1))

    # create a boolean mask based on the zdiff and threshold values
    short_signal = zdiff > short_threshold
    long_signal = zdiff < long_threshold

    # set the signal value where the condition is met
    new_df.loc[short_signal, f"signal_{period}"] = -1
    new_df.loc[long_signal, f"signal_{period}"] = 1


# print the number of signals for each period
for period in n_periods:
    print(
        f'Number of signals for period {period}: {new_df[f"signal_{period}"].count()}'
    )

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
new_df["current_ret"] = stockA["returns_1d"] * new_df["wtBTC"].shift(1) - stockB[
    "returns_1d"
] * new_df["wtETH"].shift(1)

st.dataframe(new_df)
