import streamlit as st
import pandas as pd
import numpy as np
from functions import *

st.set_page_config(
    page_title="Augmented Dickey Fuller test", page_icon="📈", layout="wide"
)

with st.form("my_form1"):
    # select stock pair
    st.header("Augmented Dickey Fuller test")
    st.subheader("Choose Stock A and Stock B")
    stockA_name = st.selectbox(
        "Choose Stock A", ["BTC", "ETH", "LTC", "AAVE", "DOGE", "SOL", "MATIC"]
    )
    stockB_name = st.selectbox(
        "Choose Stock B", ["BTC", "ETH", "LTC", "AAVE", "DOGE", "SOL", "MATIC"]
    )

    st.subheader("Choose Start and End Dates for Stock A and Stock B")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-04-01"))

    if st.form_submit_button("Start ADF test"):
        # Load the stock data into two separate dataframes, stockA and stockB
        stockB = pd.read_csv(f"Gemini_{stockB_name}USD_1h.csv")
        stockA = pd.read_csv(f"Gemini_{stockA_name}USD_1h.csv")
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
        # make a df which contains just close price of stockA and stockB
        df = pd.DataFrame()
        df["stockA"] = stockA["close"]
        df["stockB"] = stockB["close"]
        print(len(stockA["close"]), len(stockB["close"]))
        # drop na values
        df.dropna(inplace=True)
        # take difference of close prices
        df["diff"] = (df["stockA"]) - (df["stockB"])
        # Take ratio of close prices
        df["ratio"] = (df["stockA"]) / (df["stockB"])

        # perform dickey-fuller test on stockA with StockB
        st.subheader("Dickey-Fuller test for difference of close prices")
        adf_test(df["diff"], title="Difference")
        st.subheader("Dickey-Fuller test for ratio of close prices")
        adf_test(df["ratio"], title="Ratio")
