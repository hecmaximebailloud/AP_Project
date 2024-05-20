import streamlit as st
import pandas as pd
import numpy as np
from scripts.data_processing import load_data, preprocess_data, load_all_data, preprocess_all_data, merge_datasets
from scripts.model_training import train_random_forest, train_rf_model
from scripts.model_evaluation import evaluate_model, mae, rmse

# Load and preprocess data
btc_file = 'data/btc.csv'
tickers = ['AMAZON', 'APPLE', 'google', 'TESLA', 'GOLD', 'CL1 COMB Comdty', 'NG1 COMB Comdty', 'CO1 COMB Comdty', 
           'DowJones', 'Nasdaq', 'S&P', 'Cac40', 'ftse', 'NKY', 'EURR002W', 'DEYC2Y10', 'USYC2Y10', 'JPYC2Y10', 
           'TED SPREAD JPN', 'TED SPREAD US', 'TED SPREAD EUR', 'renminbiusd', 'yenusd', 'eurodollar', 'gbpusd', 
           'active_address_count', 'addr_cnt_bal_sup_10K', 'addr_cnt_bal_sup_100K', 'miner-revenue-native-unit', 
           'miner-revenue-USD', 'mvrv', 'nvt', 'tx-fees-btc', 'tx-fees-usd']
file_paths = [f'data/{ticker}.csv' for ticker in tickers]

btc = load_data(btc_file)
btc = preprocess_data(btc)

all_data = [btc] + load_all_data(tickers, file_paths)
all_data = preprocess_all_data(all_data, pd.to_datetime('09/01/2011'))
merged_df = merge_datasets(all_data)

# Extract features and labels for Random Forest
dataset_prices = merged_df.set_index('Date')
features = dataset_prices.drop(columns=['btc_Dernier Prix'])
labels = dataset_prices['btc_Dernier Prix']

# Train Random Forest and get the best hyperparameters
best_params = train_random_forest(features, labels)
rf_model = train_rf_model(features, labels, best_params)
rf_predictions, rf_rmse, rf_mae = evaluate_model(rf_model, features, labels)

# Display results on Streamlit
st.title("Bitcoin Price Predictions and Forecasts")
st.markdown("""
This app displays Bitcoin price predictions using different machine learning models (Random Forest, SARIMA, LSTM) and concludes with an investment strategy.
""")

st.header("Data Overview")
st.write("Bitcoin Price Data")
st.write(btc.head())

st.header("Model Predictions")

st.subheader("Random Forest Predictions")
rf_df = pd.DataFrame({'Date': dataset_prices.index, 'Predicted Price': rf_predictions})
fig_rf = px.line(rf_df, x='Date', y='Predicted Price', title='Random Forest Predictions')
st.plotly_chart(fig_rf)

# You can add similar sections for SARIMA and LSTM predictions

st.header("Comparison of Predictions")
fig_combined = make_subplots(rows=1, cols=1)
fig_combined.add_trace(go.Scatter(x=dataset_prices.index, y=dataset_prices['btc_Dernier Prix'], mode='lines', name='Actual Price'))
fig_combined.add_trace(go.Scatter(x=rf_df['Date'], y=rf_df['Predicted Price'], mode='lines', name='RF Prediction'))
# Add SARIMA and LSTM traces here
fig_combined.update_layout(title='Model Predictions vs Actual Price', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_combined)

st.header("Investment Strategy")
st.markdown("""
The Moving Average Crossover Strategy based on the SARIMA and Random Forest models analyzes the crossover points of short-term and long-term moving averages to make investment decisions. 
This strategy demonstrates the practical application of the model predictions.
""")

st.title("Streamlit App with Code Display")

st.header("Displaying Python Code")

# Use st.echo to show the code and execute it
st.subheader("Using st.echo")
with st.echo():
    st.write("This block of code is both shown and executed.")
    st.write("Streamlit makes it easy to create interactive apps.")
