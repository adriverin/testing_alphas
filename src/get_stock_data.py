import os
import pandas as pd
import yfinance as yf


def get_stock_data(tickers, start_date, end_date, cache_path='stock_data.parquet'):
    """
    Downloads and processes stock data with robust, intelligent caching.
    This version includes a tolerance for the end date to account for API behavior
    and non-trading days.
    
    Returns:
        pd.DataFrame: A DataFrame containing the processed stock data with columns for 
                      'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 
                      'sector', and 'cap'.
    """
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    final_df = None
    should_download = False 

    if not os.path.exists(cache_path):
        print("No cache file found. A new download is required.")
        should_download = True
    else:
        print(f"Loading cached data from '{cache_path}' to check its date range...")
        cached_df = pd.read_parquet(cache_path)
        last_cached_date = cached_df.index.get_level_values('date').max()
        first_cached_date = cached_df.index.get_level_values('date').min()
        
        # Check if the list of tickers has changed.
        cached_tickers = set(cached_df.index.get_level_values('asset').unique())
        if set(tickers) != cached_tickers:
            print("Ticker list has changed. A new download is required.")
            should_download = True

        # Consider the cache up-to-date if the last entry is within 2 days of the requested end date.
        # This accounts for yfinance's exclusive end date and weekends.
        if not should_download and ((end_date_dt - last_cached_date).days <= 2 or (start_date_dt - first_cached_date).days <= 2):
            print(f"Cache is considered up to date (last date: {last_cached_date.date()}).")
            final_df = cached_df
        elif not should_download:
            print(f"Cache is outdated (ends on {last_cached_date.date()}, but {end_date_dt.date()} was requested).")
            print("A new download is required.")
            should_download = True

    if should_download:
        print("Downloading full history to ensure data integrity...")
        raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)
        
        if raw_data.empty:
            print("Failed to download any data.")
            # If a download fails, but we have old cached data, it's better to use that than nothing.
            if 'cached_df' in locals() and cached_df is not None:
                print("Using previously cached data due to download failure.")
                final_df = cached_df
            else:
                return pd.DataFrame()
        else:
            # Process Raw Data into Clean, Long Format
            df_long = raw_data.stack(future_stack=True)
            df_long.index.names = ['date', 'asset']
            
            df_long.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            }, inplace=True)
            if 'volume' in df_long.columns:
                df_long['volume'] = df_long['volume'].astype(int)
            
            final_df = df_long
            
            print(f"Saving new, complete data to '{cache_path}'...")
            final_df.to_parquet(cache_path)

    if final_df is None or final_df.empty:
        print("No data available to process.")
        return pd.DataFrame()
        
    print("\nProcessing data for alpha calculations...")
    
    df_to_use = final_df[(final_df.index.get_level_values('date') >= start_date_dt) & 
                         (final_df.index.get_level_values('date') <= end_date_dt)].copy()

    # ... (the rest of the function for adding columns etc. is unchanged)
    print("Adding calculated columns (vwap, returns)...")
    df_to_use['vwap'] = (df_to_use['close'] + df_to_use['open'] + df_to_use['high'] + df_to_use['low']) / 4
    df_to_use['returns'] = df_to_use.groupby(level='asset')['close'].pct_change()

    print("Fetching sector and market cap info...")
    asset_info = {}
    present_tickers = df_to_use.index.get_level_values('asset').unique()
    for ticker in present_tickers:
        try:
            info = yf.Ticker(ticker).info
            asset_info[ticker] = { 'sector': info.get('sector', 'Unknown'), 'cap': info.get('marketCap', 0) }
        except Exception:
            asset_info[ticker] = {'sector': 'Unknown', 'cap': 0}
            
    df_to_use['sector'] = df_to_use.index.get_level_values('asset').map(lambda x: asset_info.get(x, {}).get('sector'))
    df_to_use['cap'] = df_to_use.index.get_level_values('asset').map(lambda x: asset_info.get(x, {}).get('cap'))
            
    df_to_use.dropna(inplace=True)
    print("\nData preparation complete.")
    
    return df_to_use