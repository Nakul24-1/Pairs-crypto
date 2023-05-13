import pandas as pd
import numpy as np
import streamlit as st
import pickle
import graphviz as graphviz

# Import roc auc score
from sklearn.metrics import roc_auc_score

# Train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree
# Print accuracy score
from sklearn.metrics import classification_report

# Print classification report
# Import confusion matrix
from sklearn.metrics import confusion_matrix

# Import time series train test split function
from sklearn.model_selection import TimeSeriesSplit

# Import roc curve
from sklearn.metrics import roc_curve

# Plot ROC curve
import matplotlib.pyplot as plt
import altair as alt


df = pd.read_csv("pairs_crypto.csv", index_col=0, parse_dates=True, skiprows=6)
# drop unnamed columns
df.drop(
    df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
)
# drop columns with no data
df.dropna(axis=1, how="all", inplace=True)
df = df.dropna()

df["spread"] = df["ret1PEP"] - df["ret1KO"]

with st.form("classifier_form"):
    # Create 2 columns
    # Instructions for the user
    st.write(
        "Probability > long threshold = long signal and probability < short threshold = short signal"
    )
    col1, col2 = st.columns(2)
    with col1:
        # take threshold values as input
        threshold_long = st.number_input("Enter long threshold", value=0.50)
        threshold_short = st.number_input("Enter short threshold", value=0.50)
        num = st.number_input("Enter number of cross validation splits", value=5)
        # Take classifier as input
        classifier = st.selectbox(
            "Choose a classifier", ["Random Forest", "Logistic Regression", "XGBoost"]
        )
    with col2:
        # Take number of estimators as input
        n_estimators = st.number_input("Enter number of estimators", value=100)
        # Take max depth as input
        max_depth = st.number_input("Enter max depth", value=8)
        # Take min samples leaf as input
        min_samples_leaf = st.number_input("Enter min samples leaf", value=120)
        # Take random state as input
        random_state = st.number_input("Enter random state", value=2)
        # make submit button
    submit_button = st.form_submit_button(label="Run Cross Validation")

# if submit button is clicked
if submit_button:
    # if classifier is random forest clf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=120, random_state=2)\
    if classifier == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif classifier == "Logistic Regression":
        clf = LogisticRegression(random_state=random_state)
    elif classifier == "XGBoost":
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            colsample_bytree=0.7,
            random_state=random_state,
        )
    # check if spread is positive or negative and assign 1 or 0
    df["spread_signal"] = np.where(df["spread"] > 0, 1, 0)
    # Shift the signal back by one day
    df["spread_signal"] = df["spread_signal"].shift(-1)
    df["spread_signal"] = df["spread_signal"].fillna(0)

    # Select X variables for decision tree
    X = df[
        [
            "closePEP",
            "ret1PEP",
            "PEP_ret5",
            "PEP_ret10",
            "PEP_ret20",
            "zPEP5",
            "zPEP10",
            "zPEP20",
            "closeKO",
            "ret1KO",
            "KO_ret5",
            "KO_ret10",
            "KO_ret20",
            "zKO5",
            "zKO10",
            "zKO20",
            "zdiff5",
            "zdiff10",
            "zdiff20",
            "vol20PEP",
            "vol20KO",
        ]
    ]

    # Select y variable for decision tree
    y1 = df["spread_signal"]
    df2 = df[["spread_signal", "spread"]]

    tscv = TimeSeriesSplit(n_splits=num)
    # Split the data into training and test sets
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y1.iloc[train_index], y1.iloc[test_index]
        # Print the length of the train and test variables
        st.write(
            "Length of train:", len(train_index), "Length of test:", len(test_index)
        )
        clf = clf
        # Fit the classifier
        clf.fit(X_train, y_train)
        # Create predictions
        y_pred = clf.predict(X_test)
        st.write("Accuracy score:", accuracy_score(y_test, y_pred))
        # Print confusion matrix
        st.write("Confusion matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        # Create predictions for the test set
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        # Calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        # Plot ROC curve
        fig, ax = plt.subplots()
        # Make the plot smaller
        fig.set_size_inches(5, 4)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams["font.size"] = 8
        plt.title("ROC curve for spread signal classifier")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.grid(True)
        st.pyplot(fig)

        # Labels for axis
        # Print AUC score
        st.write("ROC AUC score:", roc_auc_score(y_test, y_pred_proba))
        df_test = df2.iloc[test_index]
        df_test["y_pred_proba"] = y_pred_proba
        # if probability > long threshold = long signal and probability < short threshold = short signal
        df_test["position"] = np.where(
            df_test["y_pred_proba"] > threshold_long,
            1,
            np.where(df_test["y_pred_proba"] < threshold_short, -1, 0),
        )
        # calcuate future returns of the spread based on predicted signal
        df_test["strategy"] = df_test["position"].shift(1) * df_test["spread"]
        # replace NaN values with 0
        df_test["strategy"] = df_test["strategy"].fillna(0)
        # calculate cumulative returns
        df_test["creturns"] = df_test["strategy"].cumsum()
        # calculate the sharpe ratio
        sharpe = np.sqrt(365 * 24) * (
            df_test["strategy"].mean() / df_test["strategy"].std()
        )
        st.write("Sharpe ratio:", sharpe)
        # calculate the equity curve like this df['equity_curve'] = 100*(1+df['strat_ret']).cumprod()
        df_test["equity_curve"] = 100 * (1 + df_test["strategy"]).cumprod()
        # plot the equity curve using st.plot
        st.line_chart(df_test["equity_curve"])
        import pickle

        # save the model
        with open("model.pkl", "wb") as f:
            pickle.dump(clf, f)

# Add a subheader
st.subheader("Testing on unseen data")
st.write("The final model tested on data between April 2023 and mid May 2023")

tab1, tab2 , tab3 = st.tabs(["📈 Test", "🗃 Data","🔥Model Explainability"])

with tab1:
    # load the model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    # load the data
    df_test = pd.read_csv(
        "Assignment_PAIRS_data_crypto_test.csv",
        index_col=0,
        parse_dates=True,
        skiprows=6,
    )
    # drop unnamed columns
    df_test.drop(
        df_test.columns[df_test.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    # drop columns with no data
    df_test.dropna(axis=1, how="all", inplace=True)
    df_test = df_test.dropna()

    df_test["spread"] = df_test["ret1PEP"] - df_test["ret1KO"]
    df_test["spread_signal"] = np.where(df_test["spread"] > 0, 1, 0)
    # Shift the signal back by one day
    df_test["spread_signal"] = df_test["spread_signal"].shift(-1)
    df_test["spread_signal"] = df_test["spread_signal"].fillna(0)
    # Select X variables for decision tree
    X = df_test[
        [
            "closePEP",
            "ret1PEP",
            "PEP_ret5",
            "PEP_ret10",
            "PEP_ret20",
            "zPEP5",
            "zPEP10",
            "zPEP20",
            "closeKO",
            "ret1KO",
            "KO_ret5",
            "KO_ret10",
            "KO_ret20",
            "zKO5",
            "zKO10",
            "zKO20",
            "zdiff5",
            "zdiff10",
            "zdiff20",
            "vol20PEP",
            "vol20KO",
        ]
    ]

    # Select y variable for decision tree
    y1 = df_test["spread_signal"]

    # Create predictions
    y_pred = model.predict(X)
    st.write("Accuracy score:", accuracy_score(y1, y_pred))
    # Print confusion matrix
    st.write("Confusion matrix:")
    st.write(confusion_matrix(y1, y_pred))
    # Create predictions for the test set
    y_pred_proba = model.predict_proba(X)[:, 1]
    # Calculate roc curve
    fpr, tpr, thresholds = roc_curve(y1, y_pred_proba)
    # Plot ROC curve
    fig, ax = plt.subplots()
    # Make the plot smaller
    fig.set_size_inches(5, 4)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams["font.size"] = 8
    plt.title("ROC curve for spread signal classifier")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.grid(True)
    st.pyplot(fig)

    # Labels for axis
    # Print AUC score
    st.write("ROC AUC score:", roc_auc_score(y1, y_pred_proba))
    df_test["y_pred_proba"] = y_pred_proba
    # if probability > long threshold = long signal and probability < short threshold = short signal
    df_test["position"] = np.where(
        df_test["y_pred_proba"] > threshold_long,
        1,
        np.where(df_test["y_pred_proba"] < threshold_short, -1, 0),
    )
    # calcuate future returns of the spread based on predicted signal
    df_test["strategy"] = df_test["position"].shift(1) * df_test["spread"]
    # replace NaN values with 0
    df_test["strategy"] = df_test["strategy"].fillna(0)
    # calculate cumulative returns
    df_test["creturns"] = df_test["strategy"].cumsum()
    # calculate the sharpe ratio
    sharpe = np.sqrt(365 * 24) * (
        df_test["strategy"].mean() / df_test["strategy"].std()
    )
    st.write("Sharpe ratio:", sharpe)
    # calculate the equity curve like this df['equity_curve'] = 100*(1+df['strat_ret']).cumprod()
    df_test["equity_curve"] = 100 * (1 + df_test["strategy"]).cumprod()
    # plot the equity curve using st.plot
    st.line_chart(df_test["equity_curve"])

with tab2:
    st.write("Full Test data")
    st.dataframe(df_test)
    st.write("Tail of the Test data")
    st.write(df_test.tail())

with tab3:
    st.write("Feature Importance")
    # Calculate feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    # Plot feature importances
    fig, ax = plt.subplots()
    feature_importances.nlargest(20).plot(kind='barh')
    plt.title("Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    st.pyplot(fig)

    dot_data  = tree.export_graphviz(model.estimators_[0],feature_names=X.columns , out_file=None,max_depth=3, filled=True)
    st.graphviz_chart(dot_data)

    