# AP_Projet_code.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from scripts.data_processing import preprocess_all_data, calculate_returns, calculate_volatility

# Define start and end dates for the weekly data
start_date = pd.to_datetime('2011-01-09')
end_date = pd.to_datetime('2023-12-24')
keep_columns = ['Date', 'Dernier Prix']

# List of all tickers
all_data_ticker = ['AMAZON', 'APPLE', 'google', 'TESLA',
                 'GOLD', 'CL1 COMB Comdty', 'NG1 COMB Comdty', 'CO1 COMB Comdty', 
                 'DowJones', 'Nasdaq', 'S&P', 'Cac40', 'ftse', 'NKY',
                 'EURR002W', 'DEYC2Y10', 'USYC2Y10', 'JPYC2Y10', 'TED SPREAD JPN', 'TED SPREAD US', 'TED SPREAD EUR',
                 'renminbiusd', 'yenusd', 'eurodollar' ,'gbpusd',
                 'active_address_count', 'addr_cnt_bal_sup_10K', 'addr_cnt_bal_sup_100K', 'miner-revenue-native-unit', 'miner-revenue-USD', 'mvrv', 'nvt', 'tx-fees-btc', 'tx-fees-usd']

# Preprocess data
try:
    merged_df = preprocess_all_data(all_data_ticker, start_date, end_date, keep_columns)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Calculate returns and volatilities
dataset_returns = calculate_returns(merged_df)
dataset_volatility = calculate_volatility(merged_df)

# Streamlit interface
st.set_page_config(page_title='Financial Analysis and Prediction App', layout='wide')

tabs = st.tabs(['Home', 'Prices', 'Returns', 'Volatility', 'Predictive Models', 'Investment Strategy', 'News'])

# Home tab
with tabs[0]:
    st.title('Financial Analysis and Prediction App')
    st.write("""
        Welcome to the Financial Analysis and Prediction App. This application allows you to analyze and predict financial data using various models.
        You can explore different financial metrics, apply predictive models, and devise investment strategies based on predicted and actual prices.
    """)

# Prices tab
with tabs[1]:
    st.header('Price')
    features = merged_df.columns.tolist()
    selected_features = st.multiselect('Select Features', features, key='price_features')
    if selected_features:
        try:
            selected_price = [f"{feature}" for feature in selected_features]
            price = merged_df[selected_price]
            st.line_chart(price)
        except KeyError as e:
            st.error(f"Error selecting price columns: {e}")

# Returns tab
with tabs[2]:
    st.header('Returns')
    selected_features = st.multiselect('Select Features', merged_df.columns.tolist(), key='returns_features')
    if selected_features:
        try:
            selected_returns = [f"{feature}_returns" for feature in selected_features]
            returns = dataset_returns[selected_returns]
            st.line_chart(returns)
        except KeyError as e:
            st.error(f"Error selecting returns columns: {e}")

# Volatility tab
with tabs[3]:
    st.header('Volatility')
    selected_features = st.multiselect('Select Features', merged_df.columns.tolist(), key='volatility_features')
    if selected_features:
        try:
            selected_volatility = [f"{feature}_volatility" for feature in selected_features]
            volatility = dataset_volatility[selected_volatility]
            st.line_chart(volatility)
        except KeyError as e:
            st.error(f"Error selecting volatility columns: {e}")

# Predictive Models tab
with tabs[4]:
    st.header('Predictive Models')
    model_choice = st.selectbox('Select Model', ['Random Forest', 'SARIMA', 'LSTM'], key='model_choice')
    if model_choice == 'Random Forest':
        st.write('Random Forest model details and predictions...')
    elif model_choice == 'SARIMA':
        st.write('SARIMA model details and predictions...')
    elif model_choice == 'LSTM':
        st.write('LSTM model details and predictions...')

# Investment Strategy tab
with tabs[5]:
    st.header('Investment Strategy')
    strategy_choice = st.selectbox('Select Strategy', ['Predicted Bitcoin Prices', 'Actual Bitcoin Prices'], key='strategy_choice')
    if strategy_choice == 'Predicted Bitcoin Prices':
        st.write('Investment strategy based on predicted Bitcoin prices using Recursive Features Elimination')
    elif strategy_choice == 'Actual Bitcoin Prices':
        st.write('Investment strategy based on actual Bitcoin prices...')

# Bitcoin News tab
with tabs[6]:
    st.header('Latest Bitcoin News')
    st.write('Here are the latest news about Bitcoin and cryptocurrencies in general')
    # Placeholder for news integration (could be an API call to a news service)

# General Financial Markets News tab
with tabs[6]:
    st.header('Latest Financial Markets News')
    st.write('Here are the latest news about the global financial world.')
    # Placeholder for news integration (could be an API call to a news service)




