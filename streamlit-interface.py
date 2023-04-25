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


col1, col2 = st.columns(2)

new_df = pd.DataFrame()

# Load the stock data into two separate dataframes, stockA and stockB
stockB = pd.read_csv("D:\Downloads\Gemini_ETHUSD_1h.csv")
stockA = pd.read_csv("D:\Downloads\Gemini_BTCUSD_1h.csv")

with col1:
    st.header("Choose Start Date for Stock A and Stock B")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    st.write(type(start_date))

with col2:
    st.header("Inputs")
    # Set the lookback window and the list of N-day periods to compute z-scores for
    # lookback_window = 60
    lookback_window = st.number_input(
        "Lookback Window", min_value=1, max_value=240, value=60, step=1
    )
    n_periods = st.multiselect("Choose Periods", [5, 10, 20, 40, 60], [5])
    # Add one to n_periods
    n_periods = [1] + n_periods

stockA["date"] = pd.to_datetime(stockA["date"]).dt.date
stockB["date"] = pd.to_datetime(stockB["date"]).dt.date
st.write(type(stockA["date"][1]))
# select rows with date after 2020-01-01
stockA = stockA[stockA["date"] > start_date]
stockB = stockB[stockB["date"] > start_date]
# make date as index
stockA.set_index("date", inplace=True)
stockB.set_index("date", inplace=True)
# select 3 columns which are date and symbol and close columns
stockA = stockA[["symbol", "close"]]
stockB = stockB[["symbol", "close"]]

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

# Take inputs of the thresholds for each lookback period and store as a tuple

threshold_short = st.number_input(
    "Short Threshold", min_value=-3.0, max_value=3.0, value=1.0, step=0.1
)
threshold_long = st.number_input(
    "Long Threshold", min_value=-3.0, max_value=3.0, value=-1.0, step=0.1
)

thresholds = {
    5: (threshold_short, threshold_long),
    10: (threshold_short, threshold_long),
    20: (threshold_short, threshold_long),
}

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

    # set the signal value where the condition is met
    new_df.loc[short_signal, f"signal_{period}"] = 1
    new_df.loc[long_signal, f"signal_{period}"] = -1


# print the number of signals for each period
for period in n_periods:
    st.write(
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
new_df["current_ret"] = (
    stockA["returns_1d"] * new_df["wtBTC"].shift(1)
    - stockB["returns_1d"] * new_df["wtETH"].shift(1)
).shift(-1)
st.dataframe(new_df.tail())

# replace NA values with 0
new_df.fillna(0, inplace=True)
pd.options.mode.chained_assignment = None

col3, col4 = st.columns(2)

with col3:
    LongCap = st.number_input(
        "Long Cap", min_value=-3.00, max_value=3.00, value=1.00, step=0.05
    )
    ShortCap = st.number_input(
        "Short Cap", min_value=-3.00, max_value=3.00, value=-1.00, step=0.05
    )
    age_limit = st.number_input(
        "Age Limit", min_value=1, max_value=480, value=60, step=1
    )

# Create columns for long and short signals for each lookback period
for period in n_periods:
    new_df[f"Long_Cap{period}"] = 0
    new_df[f"Short_Cap{period}"] = 0

    # Create column for position and age and aged for each lookback period
    new_df[f"position{period}"] = 0
    new_df[f"age{period}"] = 0
    new_df[f"aged{period}?"] = 0

# Loop over each row in the dataframe user iterrows
# for index, row in df.iterrows():

for i in range(1, len(new_df)):
    # Check for a long signal
    if (new_df["zdiff_5d"].iloc[i] > LongCap) & (new_df["position5"].iloc[i - 1] == 1):
        new_df["Long_Cap5"].iloc[i] = 1
    else:
        new_df["Long_Cap5"].iloc[i] = 0

    # Check for a short signal
    if (new_df["zdiff_5d"].iloc[i] < ShortCap) & (
        new_df["position5"].iloc[i - 1] == -1
    ):
        new_df["Short_Cap5"].iloc[i] = 1
    else:
        new_df["Short_Cap5"].iloc[i] = 0

    # Calculate Age of signal

    if (
        ((new_df["position5"].iloc[i - 1] == 0) and (new_df["signal_5"].iloc[i] != 0))
        or ((new_df["position5"].iloc[i - 1] * new_df["signal_5"].iloc[i]) == -1)
        or (
            (new_df["age5"].iloc[i - 1] == age_limit)
            and (new_df["signal_5"].iloc[i] != 0)
        )
    ):
        new_df[f"age5"].iloc[i] = 1

    elif (
        ((new_df["position5"].iloc[i - 1] == 0) and (new_df["signal_5"].iloc[i] == 0))
        or ((new_df["Long_Cap5"].iloc[i] != 0) or (new_df["Short_Cap5"].iloc[i] != 0))
        or ((new_df["age5"].iloc[i - 1] == age_limit))
    ):
        new_df[f"age5"].iloc[i] = 0
    else:
        new_df[f"age5"].iloc[i] = new_df[f"age5"].iloc[i - 1] + 1

    # Check if the signal is aged
    if new_df[f"age5"].iloc[i - 1] == age_limit:
        new_df[f"aged5?"].iloc[i] = 1
    else:
        new_df[f"aged5?"].iloc[i] = 0

    # Calculate the position
    # Excel formula =IF(AND(AE73=0,AF73=0,AG73=0,AI73=0),AJ72,AE73)

    if (
        (new_df[f"Long_Cap5"].iloc[i] == 0)
        & (new_df[f"Short_Cap5"].iloc[i] == 0)
        & (new_df[f"aged5?"].iloc[i] == 0)
        & (new_df[f"signal_5"].iloc[i] == 0)
    ):
        new_df[f"position5"].iloc[i] = new_df[f"position5"].iloc[i - 1]
    else:
        new_df[f"position5"].iloc[i] = new_df["signal_5"].iloc[i]


with col4:
    # print values related to the signal 5
    print(new_df["signal_5"].value_counts())
    print(new_df["Long_Cap5"].value_counts())
    print(new_df["Short_Cap5"].value_counts())

    print(new_df["aged5?"].value_counts())
    print(new_df["position5"].value_counts())

# Calculate the returns by multiplying the position by the returns
new_df["returns"] = new_df["current_ret"] * new_df["position5"]

# Make an equity curve starting from 100
new_df["equity_curve"] = 100 * (1 + new_df["returns"]).cumprod()
# plot the curve
st.pyplot(new_df["equity_curve"].plot(figsize=(9, 6)), use_container_width=False)
