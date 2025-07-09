import os
import pandas as pd
import yfinance as yf

try:
    import ccxt
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


def get_crypto_data(tickers, start_date, end_date, interval='1h', cache_path=None):
    """
    Downloads and processes crypto data from Binance with the same interface as get_stock_data.
    Supports hourly and minute-level data for cryptocurrencies only.
    
    Args:
        tickers: List of crypto symbols (e.g., ['BTC-USD', 'ETH-USD'])
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
        cache_path: Optional cache file path (auto-generated if None)
    
    Returns:
        pd.DataFrame: Same format as get_stock_data with MultiIndex (date, asset)
                      Columns: 'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns'
    """
    if not BINANCE_AVAILABLE:
        raise ImportError("ccxt library required for crypto data. Install with: pip install ccxt")
    
    # Auto-generate cache path if not provided
    if cache_path is None:
        tickers_str = "_".join([t.replace('-USD', '') for t in tickers])
        cache_path = f"crypto_data_{tickers_str}_{interval}.parquet"
    
    start_date_dt = pd.to_datetime(start_date, utc=True)
    end_date_dt = pd.to_datetime(end_date, utc=True)
    
    final_df = None
    should_download = False
    
    # Check cache
    if not os.path.exists(cache_path):
        print("No cache file found. A new download is required.")
        should_download = True
    else:
        print(f"Loading cached crypto data from '{cache_path}' to check its date range...")
        try:
            cached_df = pd.read_parquet(cache_path)
            last_cached_date = cached_df.index.get_level_values('date').max()
            first_cached_date = cached_df.index.get_level_values('date').min()
            
            # Check if the list of tickers has changed
            cached_tickers = set(cached_df.index.get_level_values('asset').unique())
            if set(tickers) != cached_tickers:
                print("Ticker list has changed. A new download is required.")
                should_download = True
            
            # For intraday data, consider cache fresh if within 1 hour
            time_tolerance = 1 if interval in ['1m', '5m', '15m', '1h'] else 2
            if not should_download and ((end_date_dt - last_cached_date).total_seconds() / 3600 <= time_tolerance):
                print(f"Cache is considered up to date (last date: {last_cached_date}).")
                final_df = cached_df
            elif not should_download:
                print(f"Cache is outdated (ends on {last_cached_date}, but {end_date_dt} was requested).")
                print("A new download is required.")
                should_download = True
        except Exception as e:
            print(f"Cache file corrupted: {e}. Re-downloading...")
            should_download = True
    
    if should_download:
        print(f"Downloading crypto data for {interval} interval...")
        
        # Initialize Binance client
        binance = ccxt.binance()
        
        all_data = []
        
        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            
            # Convert ticker format: BTC-USD -> BTC/USDT
            symbol_ccxt = ticker.replace('-USD', '/USDT')
            
            try:
                # Get timestamps
                since = binance.parse8601(f"{start_date}T00:00:00Z")
                end_ts = binance.parse8601(f"{end_date}T23:59:59Z")
                
                limit = 1000
                ohlcv = []
                
                # Fetch data in batches
                while since < end_ts:
                    batch = binance.fetch_ohlcv(symbol_ccxt, timeframe=interval, since=since, limit=limit)
                    if not batch:
                        break
                    
                    ohlcv.extend(batch)
                    since = batch[-1][0] + 1
                    
                    if len(batch) < limit:
                        break
                
                if not ohlcv:
                    print(f"⚠️ No data fetched for {ticker}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["asset"] = ticker
                df = df.set_index(["date", "asset"])[["open", "high", "low", "close", "volume"]]
                
                # Ensure numeric types
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"❌ Failed to fetch {ticker}: {e}")
                continue
        
        if not all_data:
            print("❌ No data was successfully downloaded")
            return pd.DataFrame()
        
        # Combine all assets
        final_df = pd.concat(all_data)
        final_df = final_df.sort_index()
        
        # Ensure volume is integer
        final_df['volume'] = final_df['volume'].fillna(0).astype(int)
        
        print(f"Saving crypto data to '{cache_path}'...")
        final_df.to_parquet(cache_path)
    
    if final_df is None or final_df.empty:
        print("No crypto data available to process.")
        return pd.DataFrame()
    
    print("\nProcessing crypto data for alpha calculations...")
    
    # Filter by date range
    df_to_use = final_df[(final_df.index.get_level_values('date') >= start_date_dt) & 
                         (final_df.index.get_level_values('date') <= end_date_dt)].copy()
    
    # Add calculated columns (same as stock data)
    print("Adding calculated columns (vwap, returns)...")
    df_to_use['vwap'] = (df_to_use['close'] + df_to_use['open'] + df_to_use['high'] + df_to_use['low']) / 4
    df_to_use['returns'] = df_to_use.groupby(level='asset')['close'].pct_change()
    
    # Add crypto-specific metadata (simplified - no need for yfinance sector lookup)
    print("Adding crypto metadata...")
    df_to_use['sector'] = 'Cryptocurrency'
    df_to_use['cap'] = 0  # Market cap not easily available via Binance API
    
    df_to_use.dropna(inplace=True)
    print(f"Crypto data preparation complete. Shape: {df_to_use.shape}")
    
    return df_to_use


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
                df_long['volume'] = df_long['volume'].fillna(0).astype(int)
            
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