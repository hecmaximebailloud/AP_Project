# app.py
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
all_data_ticker = ['btc', 'AMAZON', 'APPLE', 'google', 'TESLA',
                 'GOLD', 'CL1 COMB Comdty', 'NG1 COMB Comdty', 'CO1 COMB Comdty', 
                 'DowJones', 'Nasdaq', 'S&P', 'Cac40', 'ftse', 'NKY',
                 'EURR002W', 'DEYC2Y10', 'USYC2Y10', 'JPYC2Y10', 'TED SPREAD JPN', 'TED SPREAD US', 'TED SPREAD EUR',
                 'renminbiusd', 'yenusd', 'eurodollar' ,'gbpusd',
                 'active_address_count', 'addr_cnt_bal_sup_10K', 'addr_cnt_bal_sup_100K', 'miner-revenue-native-unit', 'miner-revenue-USD','mvrv','nvt','tx-fees-btc', 'tx-fees-usd']

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
st.title('Data Analysis and Visualization')

# Sidebar for feature selection
st.sidebar.header('Feature Selection')
features = merged_df.columns.tolist()
selected_features = st.sidebar.multiselect('Select Features', features)

if selected_features:
    st.header('Price')
    selected_price = [f"{feature}" for feature in selected_features]
    try:
        price = merged_df[selected_price]
        st.line_chart(price)
    except KeyError as e:
        st.error(f"Error selecting price columns: {e}")

    st.header('Returns')
    selected_returns = [f"{feature}_returns" for feature in selected_features]
    try:
        returns = dataset_returns[selected_returns]
        st.line_chart(returns)
    except KeyError as e:
        st.error(f"Error selecting returns columns: {e}")

    st.header('Volatility')
    selected_volatility = [f"{feature}_volatility" for feature in selected_features]
    try:
        volatility = dataset_volatility[selected_volatility]
        st.line_chart(volatility)
    except KeyError as e:
        st.error(f"Error selecting volatility columns: {e}")
else:
    st.write("Please select at least one feature to display.")

# Picture display section
st.sidebar.header('Picture Selection')
image_dir = 'path/to/your/images'  # Change to your image directory
images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
selected_image = st.sidebar.selectbox('Select an Image', images)

if selected_image:
    image_path = os.path.join(image_dir, selected_image)
    st.image(image_path, caption=selected_image)



