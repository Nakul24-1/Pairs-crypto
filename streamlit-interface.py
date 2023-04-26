import streamlit as st
import pandas as pd
import numpy as np

# perform dickey-fuller test on stockA with StockB
from statsmodels.tsa.stattools import adfuller

# Perform Granger Causality Test to check if Stock A causes Stock B
from statsmodels.tsa.stattools import grangercausalitytests

from functions import *

with st.form("my_form"):
    st.header("Pairs Trading Strategy")
    st.subheader("Choose Stock A and Stock B")
    stockA_name = st.selectbox("Choose Stock A", ["BTC", "ETH", "LTC", "MATIC"])
    stockB_name = st.selectbox(
        "Choose Stock B",
        ["BTC", "ETH", "LTC", "MATIC"],
    )
    # Load the stock data into two separate dataframes, stockA and stockB
    stockB = pd.read_csv(f"Gemini_{stockB_name}USD_1h.csv")
    stockA = pd.read_csv(f"Gemini_{stockA_name}USD_1h.csv")

    col11, col12, col13 = st.columns(3)

    with col11:
        st.subheader("Choose Start and End Dates for Stock A and Stock B")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-04-01"))

    with col12:
        st.header("Inputs")
        # Set the lookback window and the list of N-day periods to compute z-scores for
        # lookback_window = 60
        lookback_window = st.number_input(
            "Lookback Window", min_value=1, max_value=240, value=60, step=1
        )
        n_periods = st.multiselect(
            "Choose Periods", [5, 10, 20, 40, 60, 80, 100, 120], [5]
        )
        # Add one to n_periods
        n_periods = n_periods

    with col13:
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

    button_run = st.form_submit_button("Run")
    if st.session_state.get("button_run") != True:
        st.session_state["button_run"] = button_run  # Saved the state
    if st.session_state["button_run"] == True:
        stockA, stockB = start_preprocess(
            stockA, stockB, start_date, end_date, lookback_window, n_periods
        )

        stockA, stockB = compute_rolling_returns(
            stockA, stockB, n_periods, lookback_window
        )
        new_df = make_new_df(stockA, stockB, n_periods)

        new_df = zdiff_calculations(
            new_df, n_periods, thresholds, threshold_long, threshold_short
        )
        st.success("Done")

        df = pd.DataFrame()
        df["stockA"] = stockA["close"]
        df["stockB"] = stockB["close"]
        # drop na values
        df.dropna(inplace=True)
        # take difference of close price
        df["diff"] = (df["stockA"]) - (df["stockB"])
        # Take ratio of close price
        df["ratio"] = (df["stockA"]) / (df["stockB"])

        # check if the difference of log of close price is stationary
        # adf_test(df["diff"])

        # check if the ratio of log of close price is stationary
        # adf_test(df["ratio"])

        # check if stockA causes stockB
        # grangers_causation_matrix(df, variables=["stockA", "stockB"])

        # print the number of signals for each period
        for period in n_periods:
            st.write(
                f'Number of signals for period {period}: {new_df[f"signal_{period}"].count()}'
            )

        new_df = inv_volatility_returns1d(new_df, stockA, stockB)
        st.dataframe(new_df)

pd.options.mode.chained_assignment = None
st.divider()
st.header("Calculate Positions")
LongCap = st.number_input(
    "Long Cap", min_value=-3.00, max_value=3.00, value=1.00, step=0.05
)
ShortCap = st.number_input(
    "Short Cap", min_value=-3.00, max_value=3.00, value=-1.00, step=0.05
)
age_limit = st.number_input("Age Limit", min_value=1, max_value=480, value=60, step=1)

positions_button = st.button("positions_button")
if st.session_state.get("positions_button") != True:
    st.session_state["positions_button"] = positions_button  # Saved the state

if st.session_state["positions_button"] == True:
    for n in stqdm(n_periods):
        new_df = calculate_positions(new_df, n, LongCap, ShortCap, age_limit)

    col21, col22, col23, col24, col25 = st.columns(5)

    with col21:
        for n in n_periods:
            st.bar_chart(new_df[f"signal_{n}"].value_counts(), width=0, height=0)

    with col22:
        for n in n_periods:
            st.bar_chart(new_df[f"aged{n}?"].value_counts(), width=0, height=0)

    with col23:
        for n in n_periods:
            st.bar_chart(new_df[f"Long_Cap{n}"].value_counts(), width=0, height=0)

    with col24:
        for n in n_periods:
            st.bar_chart(new_df[f"Short_Cap{n}"].value_counts(), width=0, height=0)

    with col25:
        for n in n_periods:
            st.bar_chart(new_df[f"position{n}"].value_counts(), width=0, height=0)

if st.button("Plot equity curves"):
    for n in n_periods:
        new_df = calculate_returns(new_df, n)
        new_df = make_equity_curve(new_df, n)
        st.line_chart(new_df[f"equity_curve{n}"], width=0, height=0)
