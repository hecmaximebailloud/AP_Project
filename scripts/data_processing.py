import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

def preprocess_data(btc):
    start_date = pd.to_datetime('2011-09-01')
    end_date = pd.to_datetime('2023-12-24')

    # Ensure 'Date' column is in datetime format
    btc['Date'] = pd.to_datetime(btc['Date'], errors='coerce')

    btc_range = btc[(btc['Date'] >= start_date) & (btc['Date'] <= end_date)]
    btc = btc_range.reset_index(drop=True)

    # Rename 'Dernier Prix' to 'btc_Dernier Prix'
    btc.rename(columns={'Dernier Prix': 'btc_Dernier Prix'}, inplace=True)

    # Calculate returns
    btc['btc_Dernier Prix_returns'] = btc['btc_Dernier Prix'].pct_change()

    # Calculate volatility (e.g., rolling window of 4 periods)
    btc['btc_Dernier Prix_volatility'] = btc['btc_Dernier Prix'].rolling(window=4).std()

    return btc

def load_all_data(tickers, file_paths):
    all_data = []
    for ticker, file_path in zip(tickers, file_paths):
        data = load_data(file_path)
        # Rename 'Dernier Prix' to include the ticker name
        data.rename(columns={'Dernier Prix': f'{ticker}_Dernier Prix'}, inplace=True)
        all_data.append(data)
    return all_data

def preprocess_all_data(all_data, start_date):
    for i, element in enumerate(all_data):
        element['Date'] = pd.to_datetime(element['Date'], errors='coerce')
        element.drop(element[element['Date'] < start_date].index, inplace=True)
    return all_data

def merge_datasets(all_data):
    common_dates_all = set(all_data[0]['Date'])
    for element in all_data:
        common_dates_all = common_dates_all.intersection(set(element['Date']))
        element = element[element['Date'].isin(common_dates_all)].copy()
        element.sort_values(by='Date', inplace=True)
    
    merged_df = pd.concat(all_data, axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    return merged_df

def calculate_returns(df):
    # Ensure the DataFrame has numeric columns
    numeric_cols = df.columns.difference(['Date'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    returns_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        # Debug: Check the column values before calculation
        print(f"Calculating returns for column: {col}")
        print(df[col].head())

        # Calculate returns: (current value / previous value) - 1
        returns_df[col + '_returns'] = (df[col] / df[col].shift(1) - 1).replace([np.inf, -np.inf, np.nan], 0)

    return returns_df

def calculate_volatility(df, window):
    numeric_cols = df.columns.difference(['Date'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    volatility_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        volatility_df[col + '_volatility'] = df[col].rolling(window=window).std()

    volatility_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    volatility_df.fillna(0, inplace=True)

    return volatility_df

def calculate_z_score(returns_df, volatility_df, prices_df):
    numeric_cols = prices_df.columns.difference(['Date'])
    prices_df[numeric_cols] = prices_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    z_score_df = pd.DataFrame(index=prices_df.index)
    for col in numeric_cols:
        z_score_df[col + '_z_score'] = returns_df[col + '_returns'] / volatility_df[col + '_volatility']
    
    z_score_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    z_score_df.fillna(0, inplace=True)

    return z_score_df

