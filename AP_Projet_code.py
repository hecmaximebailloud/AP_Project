# AP_Projet_code.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.data_processing import preprocess_all_data, calculate_returns, calculate_volatility, fetch_latest_news, get_countdown, halving_details, next_halving_date



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

feature_groups = {
        "Equity_Indices": [
            "DowJones", "Nasdaq", "S&P", "Cac40", "ftse", "NKY"
        ],
        "Individual_Stocks": [
            "AMAZON", "APPLE", "google", "TESLA"
        ],
        "Commodity_Prices": [
            "GOLD", "CL1 COMB Comdty", "NG1 COMB Comdty", "CO1 COMB Comdty"
        ],
        "Interest_Rates_Yields": [
            "USYC2Y10", "TED SPREAD EUR", "TED SPREAD US", "TED SPREAD JPN", "EURR002W", "JPYC2Y10","DEYC2Y10"
        ],
       
        "Blockchain_Cryptocurrency_Metrics": [
            "active_address_count", "addr_cnt_bal_sup_10K", "addr_cnt_bal_sup_100K", "miner-revenue-native-unit", "miner-revenue-USD", "mvrv", "nvt", "tx-fees-btc", "tx-fees-usd"
        ],
        "Foreign_Exchange_Rates": [
            "eurodollar", "gbpusd", "renminbiusd", "yenusd"
        ]
         }

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

tabs = st.tabs(['Home', 'Prices', 'Returns', 'Volatility', 'Correlation', 'Groups Analysis', 'Predictive Models', 'Investment Strategy', 'Bitcoin Halving', 'Crypto News'])

# Home tab
with tabs[0]:
  # Title and Introduction
 st.title("Impact of Economic Indicators on Bitcoin Price Predictions and Investment Strategies")
 st.write("""
 Welcome to the Bitcoin Price Prediction and Investment Strategy Dashboard App in the context of the Advanced Programming project.

 This project aims to develop a framework for predicting Bitcoin prices using various machine-learning models and economic indicators. I have used Random Forest, SARIMA, and LSTM models to analyze the data and develop investment strategies. Explore the different sections to learn more about the methodology, results, and insights.
""")


 st.write("## Key Insights")
 st.markdown("""
 - **Model Performance**: The Random Forest model with feature selection (RFE) provided more accurate predictions compared to models using all features.
 - **Feature Importance**: Economic indicators such as equity indices, commodity prices, and blockchain metrics significantly impact Bitcoin prices.
 - **Investment Strategy**: The Moving Average Crossover Strategy based on SARIMA and Random Forest predictions shows practical application for making investment decisions.
 - **Challenges**: Prediction accuracy varies during significant market events such as Bitcoin halving.
""")

 st.write("## Project Details")
 st.markdown("""
 For detailed information about the project, methodology, and results, please refer to the following sections or visit the [GitHub repository](https://github.com/hecmaximebailloud/AP_Project.git).

 - [Prices](#prices)
 - [Returns](#returns)
 - [Volatility](#volatility)
 - [Correlation](#correlation)
 - [Groups Analysis](#groups)
 - [Predictive Models](#models)
 - [Investment Strategy](#investment)
 - [Bitcoin Halving](#halving)
 - [Crypto News](#news)
 """)
   



# Prices tab
with tabs[1]:
    st.header('Prices of Features')
    features = merged_df.columns.tolist()
    selected_features = st.multiselect('Select one or more Features:', features, key='price_features')
    if selected_features:
        st.write("### Select Date Range")
        min_date = pd.to_datetime(merged_df.index.min())
        max_date = pd.to_datetime(merged_df.index.max())
        start_date, end_date = st.date_input("Date range:", [min_date, max_date], key='date_range')
        
        if start_date > end_date:
            st.error("Error: End date must be greater than start date.")
        else:
            try:
                filtered_df = merged_df.loc[start_date:end_date, selected_features]
                
                st.write("### Customize Plot")
                chart_type = st.selectbox("Select Chart Type", ['Line Chart', 'Area Chart'], key='chart_type')
                show_moving_average = st.checkbox("Show Moving Average", value=False, key='moving_average')
                st.write("##### Choose a Moving average window between a 10-week and a 50-week period")
                ma_window = st.slider("Moving Average Window", min_value=1, max_value=30, value=5, key='ma_window')

                st.write("### Price Chart")
                if chart_type == 'Line Chart':
                    if show_moving_average:
                        for feature in selected_features:
                            filtered_df[f"{feature}_MA"] = filtered_df[feature].rolling(window=ma_window).mean()
                        st.line_chart(filtered_df)
                    else:
                        st.line_chart(filtered_df)
                elif chart_type == 'Area Chart':
                    st.area_chart(filtered_df)

                st.write("### Plot Details")
                st.markdown(f"**Selected Features:** {', '.join(selected_features)}")
                st.markdown(f"**Date Range:** {start_date} to {end_date}")

            except KeyError as e:
                st.error(f"Error selecting price columns: {e}")


# Returns tab
with tabs[2]:
    st.header('Returns of Features')
    selected_features = st.multiselect('Select one or more Features:', merged_df.columns.tolist(), key='returns_features')
    if selected_features:
        try:
            selected_returns = [f"{feature}_returns" for feature in selected_features]
            
            st.write("### Select Date Range")
            min_date = pd.to_datetime(dataset_returns.index.min())
            max_date = pd.to_datetime(dataset_returns.index.max())
            start_date, end_date = st.date_input("Date range:", [min_date, max_date], key='returns_date_range')
            
            if start_date > end_date:
                st.error("Error: End date must be greater than start date.")
            else:
                filtered_returns = dataset_returns.loc[start_date:end_date, selected_returns]
                
                st.write("### Customize Plot")
                chart_type = st.selectbox("Select Chart Type", ['Line Chart', 'Area Chart', 'Bar Chart'], key='returns_chart_type')
                show_cumulative_returns = st.checkbox("Show Cumulative Returns", value=False, key='cumulative_returns')

                if show_cumulative_returns:
                    cumulative_returns = (1 + filtered_returns).cumprod() - 1
                    plot_data = cumulative_returns
                else:
                    plot_data = filtered_returns

                st.write("### Returns Chart")
                if chart_type == 'Line Chart':
                    st.line_chart(plot_data)
                elif chart_type == 'Area Chart':
                    st.area_chart(plot_data)
                elif chart_type == 'Bar Chart':
                    st.bar_chart(plot_data)

                st.write("### Plot Details")
                st.markdown(f"**Selected Features:** {', '.join(selected_features)}")
                st.markdown(f"**Date Range:** {start_date} to {end_date}")

        except KeyError as e:
            st.error(f"Error selecting returns columns: {e}")
          
# Volatility tab
with tabs[3]:
    st.header('Volatility of Features')
    selected_features = st.multiselect('Select one or more Features:', merged_df.columns.tolist(), key='volatility_features')
    if selected_features:
        try:
            selected_volatility = [f"{feature}_volatility" for feature in selected_features]
            
            # Date range selector
            st.write("### Select Date Range")
            min_date = pd.to_datetime(dataset_volatility.index.min())
            max_date = pd.to_datetime(dataset_volatility.index.max())
            start_date, end_date = st.date_input("Date range:", [min_date, max_date], key='volatility_date_range')
            
            if start_date > end_date:
                st.error("Error: End date must be greater than start date.")
            else:
                filtered_volatility = dataset_volatility.loc[start_date:end_date, selected_volatility]
                
                st.write("### Customize Plot")
                chart_type = st.selectbox("Select Chart Type", ['Line Chart', 'Area Chart'], key='volatility_chart_type')
                show_rolling_volatility = st.checkbox("Show Rolling Volatility", value=False, key='rolling_volatility')
                rolling_window = st.slider("Rolling Window Size", min_value=1, max_value=30, value=5, key='rolling_window')

                if show_rolling_volatility:
                    rolling_volatility = filtered_volatility.rolling(window=rolling_window).std()
                    plot_data = rolling_volatility
                else:
                    plot_data = filtered_volatility

                st.write("### Volatility Chart")
                if chart_type == 'Line Chart':
                    st.line_chart(plot_data)
                elif chart_type == 'Area Chart':
                    st.area_chart(plot_data)

                st.write("### Plot Details")
                st.markdown(f"**Selected Features:** {', '.join(selected_features)}")
                st.markdown(f"**Date Range:** {start_date} to {end_date}")

        except KeyError as e:
            st.error(f"Error selecting volatility columns: {e}")


# Correlation tab
with tabs[4]:
    st.header('Correlation')
    st.write(f'First, you need to choose which features you want to add in the correlation matrix. Then, You can customize the heatmap as you wish (color, size):')  
    features = dataset_returns.columns.tolist()
    selected_features = st.multiselect('Select Features', features, key='correlation_features')
    if selected_features:
        try:
            correlation_matrix = dataset_returns[selected_features].corr()
            st.write("Customize Heatmap")
            cmap_option = st.selectbox('Select Color Map', ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
            annot_option = st.checkbox('Show Annotations', value=True)
            figsize_width = st.slider('Figure Width', min_value=5, max_value=15, value=10)
            figsize_height = st.slider('Figure Height', min_value=5, max_value=15, value=6)
            
            # Display heatmap
            st.write("Correlation Heatmap")

            fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
            sns.heatmap(correlation_matrix, annot=annot_option, cmap=cmap_option, ax=ax)
            st.pyplot(fig)
        except KeyError as e:
            st.error(f"Error selecting features for correlation: {e}")
          
# Groups tab
with tabs[5]:
  st.header('Groups Analysis')
  group_choice = st.selectbox('Select what you want to see in details about groups', ['Groups Overview', 'Groups Importance', 'Importance Evolution'], key = 'group_choice')
  if group_choice == 'Groups Overview':
    st.header('Features and Groups')
    st.write('Select a group and you will see the features that belong to this group')
    for group, features in feature_groups.items():
        with st.expander(group):
             st.markdown("\n".join([f"- {feature}" for feature in features]))

  elif group_choice == 'Groups Importance':
    st.header('Importance of each group in the Random Forest model')
    st.image("Images/Group's Importance.png", caption = 'Importance of each group for the Random Forest predictions', use_column_width = False)

  elif group_choice == 'Importance Evolution':
    st.header('Evolution of the two most important groups')
    st.image('Images/Evolution of groups importance BCM and EI .png', caption = 'Evolution of their importance over time', use_column_width = False)
    st.markdown("""
        **Key Points:**
        -  Importance seems to be traveling in opposite directions in a seesaw fashion for blockchain metrics and Equity Indices—an inverse relationship. An upsurge in the importance of one group infers a decline in the other group that outlines varied market dynamics.
        """)



# Predictive Models tab
with tabs[6]:
    st.header('Predictive Models')
    model_choice = st.selectbox('Select Model', ['Random Forest', 'SARIMA', 'LSTM'], key='model_choice')
    if model_choice == 'Random Forest':
        st.subheader('Random Forest model details and predictions.')
        st.write('Here, you can see the comparison of the predicted prices between Bitcoin actual prices, a Random Forest using all features (34) and a Random Forest using the 5 most explicative features (selected with Recursive Features Elimination).')
        st.write('The top features are Google, Tesla, Nasdaq, S&P500, and miner revenue.')
        st.write('You will find below the accuracy comparison between both Random Forest models.')
        st.image('Images/Screen Shot 2024-05-16 at 8.42.15 pm.png', caption='Random Forest Model', use_column_width=True)
        st.image('Images/Accuracy Comparison between RFE and all features .png', caption = 'Accuracy of the predicted prices over time', use_column_width = False)
        st.write("### Insights")
        st.write("""
        The Random Forest model with all features shows a broad view of how different economic indicators influence Bitcoin prices. 
        Using Recursive Feature Elimination, the model was refined to focus on the most significant features, resulting in improved prediction accuracy.
        """)
        st.markdown("""
        **Key Points:**
        - The full-featured model includes 34 features, capturing various economic indicators.
        - The RFE model narrows down to 5 key features: Google, Tesla, Nasdaq, S&P500, and miner revenue.
        - The accuracy comparison highlights the improvement achieved through feature selection.
        """)
    elif model_choice == 'SARIMA':
        st.subheader('SARIMA model details and predictions')
        st.image('Consolidated BTC prices comparison.png', caption = 'SARIMA model', use_column_width = False)
        st.write("### Insights")
        st.write("""
        The SARIMA model is particularly effective in capturing seasonal trends and time-series patterns in Bitcoin prices.
        The model uses historical price data to identify recurring patterns and predict future prices.
        """)
        st.markdown("""
        **Key Points:**
        - SARIMA stands for Seasonal Autoregressive Integrated Moving Average.
        - It is well-suited for data with strong seasonal effects.
        - The model effectively captures both trend and seasonality in Bitcoin prices.
        """)
    elif model_choice == 'LSTM':
        st.subheader('LSTM model details and predictions')
        st.write('As you can see below, the overall predicted price is quite good, but the forecasted price does not look good. I would advise you not to pay attention to this if you want to invest in Bitcoin...')  
        st.write("### Predictions and Comparisons")
        st.image('Images/Screen Shot 2024-05-18 at 5.35.41 pm.png', caption='LSTM Model Predictions', use_column_width=True)
        
        st.write("### Insights")
        st.write("""
        While the LSTM model shows promise in capturing complex temporal dependencies, its performance on forecasted prices can vary.
        The model's predictions highlight areas where deep learning techniques can be further optimized for financial time-series forecasting.
        """)
        st.markdown("""
        **Key Points:**
        - LSTM stands for Long Short-Term Memory, a recurrent neural network (RNN) type.
        - LSTMs are designed to handle long-term dependencies in sequential data.
        - The model's performance depends on the quality and amount of training data.
        """)

metrics_actual_strategy = {
    "Mean Returns": 0.530479,
    "Total Return": 5.481621,
    "Standard Deviation Annualized": 0.572757,
    "Sharpe Ratio": 0.926186
}

metrics_actual_benchmark = {
    "Mean Returns": 0.507176,
    "Total Return": 5.621199,
    "Standard Deviation Annualized": 0.569447,
    "Sharpe Ratio": 0.890646
}

metrics_predicted_strategy = {
    "Mean Returns": 0.212736,
    "Total Return": 2.216004,
    "Standard Deviation Annualized": 0.54922,
    "Sharpe Ratio": 0.387342
}

metrics_predicted_benchmark = {
    "Mean Returns": 0.447542,
    "Total Return": 4.997550,
    "Standard Deviation Annualized": 0.52828,
    "Sharpe Ratio": 0.847167
}

df_metrics_actual = pd.DataFrame({
    "Metric": ["Mean Returns", "Total Return", "Standard Deviation Annualized", "Sharpe Ratio"],
    "Strategy": [metrics_actual_strategy["Mean Returns"], metrics_actual_strategy["Total Return"], metrics_actual_strategy["Standard Deviation Annualized"], metrics_actual_strategy["Sharpe Ratio"]],
    "Benchmark": [metrics_actual_benchmark["Mean Returns"], metrics_actual_benchmark["Total Return"], metrics_actual_benchmark["Standard Deviation Annualized"], metrics_actual_benchmark["Sharpe Ratio"]]
})

df_metrics_predicted = pd.DataFrame({
    "Metric": ["Mean Returns", "Total Return", "Standard Deviation Annualized", "Sharpe Ratio"],
    "Strategy": [metrics_predicted_strategy["Mean Returns"], metrics_predicted_strategy["Total Return"], metrics_predicted_strategy["Standard Deviation Annualized"], metrics_predicted_strategy["Sharpe Ratio"]],
    "Benchmark": [metrics_predicted_benchmark["Mean Returns"], metrics_predicted_benchmark["Total Return"], metrics_predicted_benchmark["Standard Deviation Annualized"], metrics_predicted_benchmark["Sharpe Ratio"]]
})

# Investment Strategy tab
with tabs[7]:
    st.header('Investment Strategy')
    st.subheader('Moving-Average Crossover Strategy')
    st.write('The strategy selected is the Moving-Average Crossover Strategy. This strategy involves taking long and short positions based on the crossover points of short-term and long-term moving averages, we buy when we forecast a price increase (positive signal) and go short when we forecast a decrease (negative signal).')  
    st.write('Here you can choose whether the performance of the strategy, based on my predictions, or the performance with the actual prices.') 
    strategy_choice = st.selectbox('Select Strategy', ['Predicted Bitcoin Prices', 'Actual Bitcoin Prices'], key='strategy_choice')
    if strategy_choice == 'Predicted Bitcoin Prices':
        st.subheader('Investment strategy based on predicted Bitcoin prices using Recursive Features Elimination.')
        st.write(' RFE output was the 5 most explicative features concerning Bitcoin prices. The top features are Google, Tesla, Nasdaq, S&P500, and the miner revenue.')
        st.write('Following the computation of the Moving-Averages, you will find the performance of the portfolio, with a benchmark that is "Long" every period.')
        st.image('Images/MA RFE.png', caption = 'Short and Long-term Moving Averages on predicted and forecasted prices', use_column_width = False)
        st.write("### Performance Metrics (Strategy vs. Benchmark)")
        st.table(df_metrics_predicted)
        st.image('Images/Strat perf RFE.png', caption = 'Performance of the strategy and the benchmark', use_column_width = False)

    elif strategy_choice == 'Actual Bitcoin Prices':
        st.subheader('Investment strategy based on actual Bitcoin prices')
        st.write('Following the computation of the Moving-Averages, you will find the performance of the portfolio, with a benchmark that is "Long" every period.')
        st.image('Images/MA actual prices.png', caption = 'Short and Long-term Moving Averages on actual and forecasted prices', use_column_width = False)
        st.write("### Performance Metrics (Strategy vs. Benchmark)")
        st.table(df_metrics_actual)
        st.image('Images/Strat perf actual prices.png', caption = 'Performance of the strategy and the benchmark', use_column_width = False)

    st.write("## Strategy Comparison")
    st.write("""
    The table below compares the key performance metrics of the strategy based on predicted Bitcoin prices and actual Bitcoin prices.
    """)
    comparison_metrics = {
        "Metric": ["Mean Returns", "Total Return", "Standard Deviation Annualized", "Sharpe Ratio"],
        "Predicted Strategy": [metrics_predicted_strategy["Mean Returns"], metrics_predicted_strategy["Total Return"], metrics_predicted_strategy["Standard Deviation Annualized"], metrics_predicted_strategy["Sharpe Ratio"]],
        "Predicted Benchmark": [metrics_predicted_benchmark["Mean Returns"], metrics_predicted_benchmark["Total Return"], metrics_predicted_benchmark["Standard Deviation Annualized"], metrics_predicted_benchmark["Sharpe Ratio"]],
        "Actual Strategy": [metrics_actual_strategy["Mean Returns"], metrics_actual_strategy["Total Return"], metrics_actual_strategy["Standard Deviation Annualized"], metrics_actual_strategy["Sharpe Ratio"]],
        "Actual Benchmark": [metrics_actual_benchmark["Mean Returns"], metrics_actual_benchmark["Total Return"], metrics_actual_benchmark["Standard Deviation Annualized"], metrics_actual_benchmark["Sharpe Ratio"]]
    }
    st.table(pd.DataFrame(comparison_metrics))



# Countdown to the next halving
countdown = get_countdown(next_halving_date)

# Halving tab
with tabs[8]:
    st.title('Bitcoin Halving')

    st.markdown("""
    Bitcoin halving is an event that occurs approximately every four years, reducing the reward for mining new blocks by half. This event decreases the rate at which new bitcoins are created and reduces the total supply of bitcoins. Based on the current Bitcoin halving cycle and schedule, 100% of all Bitcoin will be mined sometime around the year 2140.
    """)

    st.header("Historical Halving Events")
    for detail in halving_details:
        st.subheader(detail['event'])
        st.write(f"**Date:** {detail['date']}")
        st.write(f"**Block Number:** {detail['block_number']}")
        st.write(f"**Block Reward Change:** {detail['block_reward']}")
        st.write("---")

    st.header("Next Expected Halving Date")
    st.write(f"The next Bitcoin halving is expected to occur around **20th April 2028**.")

    st.header("Countdown to Next Halving")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Days", countdown.days)
    col2.metric("Hours", countdown.seconds // 3600)
    col3.metric("Minutes", (countdown.seconds // 60) % 60)
    col4.metric("Seconds", countdown.seconds % 60)

    st.header("Learn More About Bitcoin Halving")
    st.video("https://youtu.be/XqB2WoFmOKc?si=ztDfb211rpN3bk-0")


# Bitcoin News tab
with tabs[9]:
    st.header('Latest Bitcoin and Cryptocurrencies News')
    api_key = 'e2542da4e232487f8a2b6e1702e8db2f'
    news_articles = fetch_latest_news(api_key)
    if news_articles:
        for article in news_articles:
            st.subheader(article['title'])
            st.write(article['summary'])
            st.markdown(f"[Read more]({article['link']})")
    else:
        st.write("Failed to fetch the latest news.")
    
    

    




