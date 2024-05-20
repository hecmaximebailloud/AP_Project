import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(btc):
    start_date = pd.to_datetime('09/01/2011')
    end_date = pd.to_datetime('24/12/2023')

    btc_range = btc[(btc['Date'] >= start_date) & (btc['Date'] <= end_date)]
    btc = btc_range.reset_index(drop=True)
    return btc

def load_all_data(tickers, file_paths):
    all_data = []
    for ticker, file_path in zip(tickers, file_paths):
        data = load_data(file_path)
        all_data.append(data)
    return all_data

def preprocess_all_data(all_data, start_date):
    keep_columns = ['Date', 'Dernier Prix']
    for i, element in enumerate(all_data):
        element = element[keep_columns].copy()
        element['Date'] = pd.to_datetime(element['Date'])
        element.drop(element[element['Date'] < start_date].index, inplace=True)
        all_data[i] = element
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
