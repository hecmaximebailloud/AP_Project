import pandas as pd
import numpy as np
import datetime

def load_and_preprocess_data(ticker, start_date, end_date, keep_columns):
    try:
        df = pd.read_csv(f'data/{ticker}.csv')
    except FileNotFoundError:
        raise FileNotFoundError(f"File for ticker '{ticker}' not found.")
    
    if 'Date' not in df.columns:
        raise ValueError(f"'Date' column not found in the dataset for ticker '{ticker}'.")

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
    
    df = df[df['Date'].between(start_date, end_date)]
    df = df[keep_columns].copy()
    return df

def preprocess_all_data(tickers, start_date, end_date, keep_columns):
    btc = load_and_preprocess_data('btc', start_date, end_date, keep_columns)
    all_data = [btc]

    for ticker in tickers:
        df = pd.read_csv(f'data/{ticker}.csv')
        df = df[keep_columns].copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
        df = df[df['Date'].between(start_date, end_date)]
        all_data.append(df)

    common_dates_all = set(btc['Date'])
    for i, df in enumerate(all_data):
        common_dates_all = set(df['Date'])

    all_dates = pd.DataFrame({'Date': sorted(common_dates_all)})
    all_datasets_filled = [btc]
    for dataset in all_data[1:]:
        dataset_filled = pd.merge(all_dates, dataset, on='Date', how='left')
        dataset_filled = dataset_filled.fillna(method='ffill')
        all_datasets_filled.append(dataset_filled)

    for i, (df, ticker) in enumerate(zip(all_datasets_filled, ['btc'] + tickers)):
        df.columns = [f"{ticker}_{col}" if col != "Date" else "Date" for col in df.columns]

    merged_df = pd.concat(all_datasets_filled, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.set_index('Date', inplace=True)
    
    return merged_df

def calculate_returns(df):
    returns_df = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    return returns_df.add_suffix('_returns')

def calculate_volatility(df, window=4):
    volatility_df = df.rolling(window=window).std().replace([np.inf, -np.inf], np.nan).fillna(0)
    return volatility_df.add_suffix('_volatility')

from newsapi import NewsApiClient

def fetch_latest_news(api_key):
    newsapi = NewsApiClient(api_key=api_key)
    
    # Fetch top headlines about Bitcoin and Cryptocurrency
    all_articles = newsapi.get_everything(q='bitcoin OR cryptocurrency', language='en', sort_by='publishedAt')
    
    articles = all_articles['articles']
    news_list = []
    
    for article in articles:
        news_item = {
            'title': article['title'],
            'link': article['url'],
            'summary': article['description']
        }
        news_list.append(news_item)
    
    return news_list

def get_countdown(target_date):
    now = datetime.datetime.now()
    countdown = target_date - now
    return countdown

# Additional information for each halving
halving_details = [
    {
        "event": "First Halving",
        "date": "28th November 2012",
        "block_number": 210000,
        "block_reward": "50 BTC to 25 BTC"
    },
    {
        "event": "Second Halving",
        "date": "9th July 2016",
        "block_number": 420000,
        "block_reward": "25 BTC to 12.5 BTC"
    },
    {
        "event": "Third Halving",
        "date": "11th May 2020",
        "block_number": 630000,
        "block_reward": "12.5 BTC to 6.25 BTC"
    },
    {
        "event": "Fourth Halving",
        "date": "20th April 2024",
        "block_number": 840000,
        "block_reward": "6.25 BTC to 3.125 BTC"
    }
]

next_halving_date = datetime.datetime(2028, 4, 20)













