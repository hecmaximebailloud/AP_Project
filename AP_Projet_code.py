import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Import preprocessing functions
from scripts.data_processing import preprocess_all_data, calculate_returns, calculate_volatility

# Define start and end dates for the weekly data
start_date = pd.to_datetime('2011-09-01')
end_date = pd.to_datetime('2023-12-24')
keep_columns = ['Date', 'Dernier Prix']

# List of all tickers
all_data_ticker = ['btc', 'AMAZON', 'APPLE', 'google', 'TESLA',
                 'GOLD', 'CL1 COMB Comdty', 'NG1 COMB Comdty', 'CO1 COMB Comdty', 
                 'DowJones', 'Nasdaq', 'S&P', 'Cac40', 'ftse', 'NKY',
                 'EURR002W', 'DEYC2Y10', 'USYC2Y10', 'JPYC2Y10', 'TED SPREAD JPN', 'TED SPREAD US', 'TED SPREAD EUR',
                 'renminbiusd', 'yenusd', 'eurodollar' ,'gbpusd',
                 'active_address_count', 'addr_cnt_bal_sup_10K', 'addr_cnt_bal_sup_100K' , 'miner-revenue-native-unit','miner-revenue-USD','mvrv','nvt','tx-fees-btc', 'tx-fees-usd']

# Preprocess data
try:
    merged_df = preprocess_all_data(all_data_ticker, start_date, end_date, keep_columns)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Debug: Check the date range in merged_df
st.write(f"Data date range: {merged_df.index.min()} to {merged_df.index.max()}")

# Calculate returns and volatilities
dataset_returns = calculate_returns(merged_df)
dataset_volatility = calculate_volatility(merged_df)

# Streamlit interface
st.title('Bitcoin Price Prediction and Analysis')

# Sidebar for feature selection
features = merged_df.columns.tolist()
selected_features = st.sidebar.multiselect('Select Features', features)

# Display selected features returns and volatility
if selected_features:
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

# Placeholder for model predictions
def load_model_predictions(model_name, data):
    # Placeholder model logic (replace with actual model predictions)
    if model_name == 'Random Forest':
        # Load the model
        model = joblib.load('scripts/random_forest_model.pkl')
        predictions = model.predict(data)
    elif model_name == 'SARIMA':
        # Example for SARIMA (replace with actual implementation)
        model = SARIMAX(data['btc_Dernier Prix'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        predictions = results.predict(start=len(data), end=len(data) + 50)
    elif model_name == 'LSTM':
        # Example for LSTM (replace with actual implementation)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Assume data is already preprocessed and split
        predictions = model.predict(data)
    else:
        predictions = np.random.randn(len(data))
    return predictions

# Select model for prediction
model_options = ['Random Forest', 'SARIMA', 'LSTM']
selected_model = st.sidebar.selectbox('Select Model', model_options)

if selected_model:
    st.header(f'BTC Price Predictions using {selected_model}')
    # Assume 'features' variable contains the feature columns for prediction
    try:
        features_for_prediction = merged_df[selected_features]
        predictions = load_model_predictions(selected_model, features_for_prediction)
        prediction_df = pd.DataFrame({'Date': merged_df.index, 'Predicted_Price': predictions})
        st.line_chart(prediction_df.set_index('Date'))
    except KeyError as e:
        st.error(f"Error selecting prediction columns: {e}")
