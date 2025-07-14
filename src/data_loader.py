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
    Downloads and processes crypto data from Binance with intelligent caching.
    Automatically detects when requested date ranges extend beyond cached data.
    
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
    
    print(f"📅 Requested date range: {start_date} to {end_date}")
    
    final_df = None
    should_download = False
    
    # Check cache with intelligent range validation
    if not os.path.exists(cache_path):
        print("📂 No cache file found. A new download is required.")
        should_download = True
    else:
        print(f"📂 Loading cached crypto data from '{cache_path}' to validate coverage...")
        try:
            cached_df = pd.read_parquet(cache_path)
            
            if cached_df.empty:
                print("🔄 Cache file is empty. A new download is required.")
                should_download = True
            else:
                cache_start = cached_df.index.get_level_values('date').min()
                cache_end = cached_df.index.get_level_values('date').max()
                
                print(f"📊 Cached date range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
                
                # Check if the list of tickers has changed
                cached_tickers = set(cached_df.index.get_level_values('asset').unique())
                requested_tickers = set(tickers)
                if requested_tickers != cached_tickers:
                    print(f"🔄 Ticker list changed. Cached: {cached_tickers}, Requested: {requested_tickers}")
                    should_download = True
                
                # **INTELLIGENT RANGE COVERAGE CHECK**
                elif start_date_dt < cache_start or end_date_dt > cache_end:
                    # Check if the extension is significant (more than 7 days)
                    start_gap = max(0, (cache_start - start_date_dt).days)
                    end_gap = max(0, (end_date_dt - cache_end).days)
                    
                    if start_gap > 7 or end_gap > 7:
                        print(f"🔄 Requested range extends significantly beyond cached data:")
                        if start_date_dt < cache_start:
                            print(f"   • Start date {start_date} is {start_gap} days before cached start {cache_start.strftime('%Y-%m-%d')}")
                        if end_date_dt > cache_end:
                            print(f"   • End date {end_date} is {end_gap} days after cached end {cache_end.strftime('%Y-%m-%d')}")
                        print("   → A new download is required.")
                        should_download = True
                    else:
                        print(f"✅ Cache covers most of requested range (gaps: start={start_gap}d, end={end_gap}d). Using cached data.")
                        final_df = cached_df
                
                else:
                    print(f"✅ Cache fully covers requested range. Using cached data.")
                    final_df = cached_df
                
        except Exception as e:
            print(f"🔄 Cache file corrupted: {e}. Re-downloading...")
            should_download = True
    
    if should_download:
        print(f"🌐 Downloading crypto data for {interval} interval...")
        
        # Initialize Binance client
        binance = ccxt.binance()
        
        all_data = []
        
        for ticker in tickers:
            print(f"  📈 Fetching {ticker}...")
            
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
        
        print(f"💾 Saving crypto data to '{cache_path}'...")
        final_df.to_parquet(cache_path)
        
        actual_start = final_df.index.get_level_values('date').min()
        actual_end = final_df.index.get_level_values('date').max()
        print(f"✅ Downloaded and cached: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
    
    if final_df is None or final_df.empty:
        print("❌ No crypto data available to process.")
        return pd.DataFrame()
    
    print("\n🔧 Processing crypto data for alpha calculations...")
    
    # Filter by date range (ensure we only return requested range)
    df_to_use = final_df[(final_df.index.get_level_values('date') >= start_date_dt) & 
                         (final_df.index.get_level_values('date') <= end_date_dt)].copy()
    
    if df_to_use.empty:
        print(f"⚠️ No data available for requested range {start_date} to {end_date}")
        return pd.DataFrame()
    
    # Add calculated columns (same as stock data)
    print("📊 Adding calculated columns (vwap, returns)...")
    df_to_use['vwap'] = (df_to_use['close'] + df_to_use['open'] + df_to_use['high'] + df_to_use['low']) / 4
    df_to_use['returns'] = df_to_use.groupby(level='asset')['close'].pct_change()
    
    # Add crypto-specific metadata (simplified - no need for yfinance sector lookup)
    print("🏷️ Adding crypto metadata...")
    df_to_use['sector'] = 'Cryptocurrency'
    df_to_use['cap'] = 0  # Market cap not easily available via Binance API
    
    df_to_use.dropna(inplace=True)
    
    final_start = df_to_use.index.get_level_values('date').min()
    final_end = df_to_use.index.get_level_values('date').max()
    print(f"✅ Crypto data preparation complete.")
    print(f"📅 Final data range: {final_start.strftime('%Y-%m-%d')} to {final_end.strftime('%Y-%m-%d')}")
    print(f"📊 Shape: {df_to_use.shape}")
    
    return df_to_use


def get_stock_data(tickers, start_date, end_date, cache_path='stock_data.parquet'):
    """
    Downloads and processes stock data with intelligent caching.
    Automatically detects when requested date ranges extend beyond cached data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the processed stock data with columns for 
                      'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 
                      'sector', and 'cap'.
    """
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    print(f"📅 Requested date range: {start_date} to {end_date}")
    
    final_df = None
    should_download = False 

    if not os.path.exists(cache_path):
        print("📂 No cache file found. A new download is required.")
        should_download = True
    else:
        print(f"📂 Loading cached data from '{cache_path}' to validate coverage...")
        try:
            cached_df = pd.read_parquet(cache_path)
            
            if cached_df.empty:
                print("🔄 Cache file is empty. A new download is required.")
                should_download = True
            else:
                cache_start = cached_df.index.get_level_values('date').min()
                cache_end = cached_df.index.get_level_values('date').max()
                
                print(f"📊 Cached date range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
                
                # Check if the list of tickers has changed
                cached_tickers = set(cached_df.index.get_level_values('asset').unique())
                requested_tickers = set(tickers)
                if requested_tickers != cached_tickers:
                    print(f"🔄 Ticker list changed. Cached: {cached_tickers}, Requested: {requested_tickers}")
                    should_download = True
                
                # **INTELLIGENT RANGE COVERAGE CHECK**
                elif start_date_dt < cache_start or end_date_dt > cache_end:
                    # Check if the extension is significant (more than 7 days)
                    start_gap = max(0, (cache_start - start_date_dt).days)
                    end_gap = max(0, (end_date_dt - cache_end).days)
                    
                    if start_gap > 7 or end_gap > 7:
                        print(f"🔄 Requested range extends significantly beyond cached data:")
                        if start_date_dt < cache_start:
                            print(f"   • Start date {start_date} is {start_gap} days before cached start {cache_start.strftime('%Y-%m-%d')}")
                        if end_date_dt > cache_end:
                            print(f"   • End date {end_date} is {end_gap} days after cached end {cache_end.strftime('%Y-%m-%d')}")
                        print("   → A new download is required.")
                        should_download = True
                    else:
                        print(f"✅ Cache covers most of requested range (gaps: start={start_gap}d, end={end_gap}d). Using cached data.")
                        final_df = cached_df
                
                else:
                    print(f"✅ Cache fully covers requested range. Using cached data.")
                    final_df = cached_df
                    
        except Exception as e:
            print(f"🔄 Cache file corrupted: {e}. Re-downloading...")
            should_download = True

    if should_download:
        print("🌐 Downloading stock data...")
        raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)
        
        if raw_data.empty:
            print("❌ Failed to download any data.")
            # If a download fails, but we have old cached data, it's better to use that than nothing.
            if 'cached_df' in locals() and cached_df is not None:
                print("📂 Using previously cached data due to download failure.")
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
            
            print(f"💾 Saving new data to '{cache_path}'...")
            final_df.to_parquet(cache_path)
            
            actual_start = final_df.index.get_level_values('date').min()
            actual_end = final_df.index.get_level_values('date').max()
            print(f"✅ Downloaded and cached: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")

    if final_df is None or final_df.empty:
        print("❌ No data available to process.")
        return pd.DataFrame()
        
    print("\n🔧 Processing data for alpha calculations...")
    
    # Filter by date range (ensure we only return requested range)
    df_to_use = final_df[(final_df.index.get_level_values('date') >= start_date_dt) & 
                         (final_df.index.get_level_values('date') <= end_date_dt)].copy()

    if df_to_use.empty:
        print(f"⚠️ No data available for requested range {start_date} to {end_date}")
        return pd.DataFrame()

    # Add calculated columns (vwap, returns)
    print("📊 Adding calculated columns (vwap, returns)...")
    df_to_use['vwap'] = (df_to_use['close'] + df_to_use['open'] + df_to_use['high'] + df_to_use['low']) / 4
    df_to_use['returns'] = df_to_use.groupby(level='asset')['close'].pct_change()

    print("🏢 Fetching sector and market cap info...")
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
    
    final_start = df_to_use.index.get_level_values('date').min()
    final_end = df_to_use.index.get_level_values('date').max()
    print(f"✅ Stock data preparation complete.")
    print(f"📅 Final data range: {final_start.strftime('%Y-%m-%d')} to {final_end.strftime('%Y-%m-%d')}")
    print(f"📊 Shape: {df_to_use.shape}")
    
    return df_to_use