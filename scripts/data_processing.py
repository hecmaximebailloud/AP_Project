import numpy as np

import pandas as pd

def load_and_preprocess_data(ticker, start_date, end_date, keep_columns):
    try:
        df = pd.read_csv(f'data/{ticker}.csv')
    except FileNotFoundError:
        raise FileNotFoundError(f"File for ticker '{ticker}' not found.")
    
    if 'Date' not in df.columns:
        raise ValueError(f"'Date' column not found in the dataset for ticker '{ticker}'.")

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error converting 'Date' column to datetime for ticker '{ticker}': {e}")
    
    df = df[df['Date'].between(start_date, end_date)]
    df = df[keep_columns].copy()
    return df

def preprocess_all_data(tickers, start_date, end_date, keep_columns):
    all_data = []
    for ticker in tickers:
        df = load_and_preprocess_data(ticker, start_date, end_date, keep_columns)
        df.columns = [f"{ticker}_{col}" if col != "Date" else "Date" for col in df.columns]
        all_data.append(df)

    # Find common dates among all datasets
    common_dates_all = set(all_data[0]['Date'])
    for df in all_data[1:]:
        common_dates_all.intersection_update(set(df['Date']))

    # Filter each dataset to include only common dates
    for i, df in enumerate(all_data):
        all_data[i] = df[df['Date'].isin(common_dates_all)].copy()
        
    # Merge all datasets into a single DataFrame
    merged_df = pd.concat(all_data, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # Remove duplicate columns
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df.set_index('Date', inplace=True)
    
    return merged_df

def calculate_returns(df):
    returns_df = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    return returns_df.add_suffix('_returns')

def calculate_volatility(df, window=4):
    volatility_df = df.rolling(window=window).std().replace([np.inf, -np.inf], np.nan).fillna(0)
    return volatility_df.add_suffix('_volatility')
