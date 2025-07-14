"""
ML Data Loading
===============

Enhanced data loading for ML forecasting with intelligent caching.
Supports both direct loading and feature engineering pipelines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Import exchange libraries
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from .config import MLConfig


def load_and_validate_data(config: MLConfig) -> pd.DataFrame:
    """
    Main entry point for loading and validating price data.
    
    Args:
        config: ML configuration object
        
    Returns:
        DataFrame with timestamp index and processed price data
        
    Raises:
        RuntimeError: If data loading fails
    """
    print(f"📊 Loading data for {config.symbol} ({config.start} to {config.end})")
    
    # Load raw price data
    df = load_price_history(config)
    
    # Validate and clean
    df = validate_data(df, config)
    
    # Apply basic preprocessing
    df = preprocess_data(df, config)
    
    return df


def load_price_history(config: MLConfig) -> pd.DataFrame:
    """
    Load price history with intelligent caching that detects date range changes.
    
    Args:
        config: ML configuration object
        
    Returns:
        DataFrame with timestamp index and 'close' column
        
    Raises:
        RuntimeError: If no data can be fetched
    """
    # Create cache file path
    cache_file = (
        config.data_cache_dir / 
        f"prices_{config.symbol.replace('/', '').replace('-', '')}_{config.interval}.parquet"
    )
    
    start_date_dt = pd.to_datetime(config.start, utc=True)
    end_date_dt = pd.to_datetime(config.end, utc=True)
    
    print(f"📅 Requested date range: {config.start} to {config.end}")
    
    # **INTELLIGENT CACHE VALIDATION**
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            
            if not cached_df.empty:
                cache_start = cached_df.index.min()
                cache_end = cached_df.index.max()
                
                print(f"📊 Cached date range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
                
                # **KEY FIX: Check if requested range is FULLY COVERED by cache**
                if start_date_dt >= cache_start and end_date_dt <= cache_end:
                    print(f"✅ Cache fully covers requested range. Using cached data.")
                    # Filter to requested range
                    mask = (cached_df.index >= start_date_dt) & (cached_df.index <= end_date_dt)
                    return cached_df[mask]
                else:
                    print(f"🔄 Requested range extends beyond cached data:")
                    if start_date_dt < cache_start:
                        print(f"   • Start date {config.start} is before cached start {cache_start.strftime('%Y-%m-%d')}")
                    if end_date_dt > cache_end:
                        print(f"   • End date {config.end} is after cached end {cache_end.strftime('%Y-%m-%d')}")
                    print("   → Fetching fresh data...")
            else:
                print(f"🔄 Cache file is empty. Fetching fresh data...")
                
        except Exception as e:
            print(f"🔄 Error loading cache: {e}. Fetching fresh data...")
    else:
        print(f"📂 No cache file found. Fetching fresh data...")
    
    # Fetch from exchange
    print(f"🌐 Fetching {config.symbol} data from Binance...")
    df = _fetch_from_binance(config)
    
    # Save to cache
    try:
        df.to_parquet(cache_file)
        actual_start = df.index.min()
        actual_end = df.index.max()
        print(f"💾 Saved {len(df)} rows to cache: {cache_file.name}")
        print(f"✅ Downloaded and cached: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save to cache: {e}")
    
    return df


def _fetch_from_binance(config: MLConfig) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance exchange.
    
    Args:
        config: ML configuration object
        
    Returns:
        DataFrame with timestamp index and OHLCV columns
        
    Raises:
        RuntimeError: If no data can be fetched
    """
    if not CCXT_AVAILABLE:
        raise RuntimeError("ccxt library required for crypto data. Install with: pip install ccxt")
    
    try:
        binance = ccxt.binance({'timeout': 30000})  # 30 second timeout
        
        # Convert symbol format for CCXT
        symbol_ccxt = config.symbol.replace('-USD', '/USDT')
        
        # Parse dates
        since = binance.parse8601(f"{config.start}T00:00:00Z")
        end_ts = binance.parse8601(f"{config.end}T00:00:00Z")
        
        timeframe = config.interval
        limit = 1000  # Binance limit per request
        all_ohlcv = []
        
        print(f"📊 Fetching {symbol_ccxt} {timeframe} data from {config.start} to {config.end}")
        
        # Fetch data in batches
        current_since = since
        batch_count = 0
        
        while current_since < end_ts:
            try:
                batch = binance.fetch_ohlcv(
                    symbol_ccxt, 
                    timeframe=timeframe, 
                    since=current_since, 
                    limit=limit
                )
                
                if not batch:
                    print("⚠️  No more data available")
                    break
                
                all_ohlcv.extend(batch)
                batch_count += 1
                
                # Update since to last timestamp + 1ms
                current_since = batch[-1][0] + 1
                
                # Progress indicator
                if batch_count % 10 == 0:
                    current_date = pd.to_datetime(batch[-1][0], unit='ms')
                    print(f"    Fetched {len(all_ohlcv)} bars (up to {current_date.strftime('%Y-%m-%d')})")
                
                # Break if we got less than the limit (indicates end of data)
                if len(batch) < limit:
                    break
                    
            except Exception as e:
                print(f"⚠️  Error fetching batch: {e}")
                # Try to continue with next batch
                current_since += 1000 * 60 * 1000  # Skip ahead 1000 minutes
                continue
        
        if not all_ohlcv:
            raise RuntimeError(f"No data fetched for {symbol_ccxt}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated()].sort_index()
        
        # Filter to requested date range
        start_date = pd.to_datetime(config.start, utc=True)
        end_date = pd.to_datetime(config.end, utc=True)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        print(f"✅ Successfully fetched {len(df)} bars")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Return only close price for compatibility with existing code
        return df[["close"]]
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from Binance: {e}")


def validate_data(df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
    """
    Validate and clean price data.
    
    Args:
        df: Raw price DataFrame
        config: ML configuration object
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        ValueError: If data is insufficient or invalid
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    # Check for minimum data points
    min_required = max(100, config.vol_window_hours * 2)  # At least 2x volatility window
    if len(df) < min_required:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_required}")
    
    # Remove rows with invalid prices
    initial_len = len(df)
    df = df[df['close'] > 0]  # Remove zero/negative prices
    df = df.dropna(subset=['close'])  # Remove NaN prices
    
    if len(df) == 0:
        raise ValueError("No valid price data after cleaning")
    
    if len(df) < initial_len * 0.95:  # Lost more than 5% of data
        print(f"⚠️  Warning: Removed {initial_len - len(df)} invalid price rows ({(initial_len - len(df))/initial_len*100:.1f}%)")
    
    # Check for data gaps
    time_diff = df.index.to_series().diff()
    expected_freq = pd.Timedelta(hours=bar_size_hours(config.interval))
    large_gaps = time_diff > expected_freq * 2  # Gaps larger than 2x expected frequency
    
    if large_gaps.sum() > 0:
        print(f"⚠️  Warning: Found {large_gaps.sum()} large time gaps in data")
        gap_locations = df.index[large_gaps]
        for gap_loc in gap_locations[:5]:  # Show first 5 gaps
            gap_size = time_diff[time_diff.index == gap_loc].iloc[0]
            print(f"    Gap at {gap_loc}: {gap_size}")
    
    return df


def preprocess_data(df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
    """
    Apply basic preprocessing to price data.
    
    Args:
        df: Price DataFrame with 'close' column
        config: ML configuration object
        
    Returns:
        DataFrame with basic preprocessing applied
    """
    df = df.copy()
    
    # Basic return calculation
    df['return'] = df['close'].pct_change()
    
    # Remove extreme outliers (returns > 50% or < -50%)
    extreme_mask = (df['return'].abs() > 0.5)
    if extreme_mask.sum() > 0:
        print(f"⚠️  Warning: Removing {extreme_mask.sum()} extreme return outliers")
        df.loc[extreme_mask, 'return'] = np.nan
        df['return'] = df['return'].ffill()  # Forward fill
    
    # Ensure no infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    if len(df) == 0:
        raise ValueError("No data remaining after preprocessing")
    
    return df


def bar_size_hours(interval: str) -> float:
    """
    Convert interval string to hours.
    
    Args:
        interval: Time interval string (e.g., '1h', '4h', '1d')
        
    Returns:
        Number of hours per bar
    """
    interval_mapping = {
        '1m': 1/60,
        '5m': 5/60,
        '15m': 15/60,
        '1h': 1,
        '4h': 4,
        '1d': 24,
        '1w': 168
    }
    
    return interval_mapping.get(interval, 1)  # Default to 1 hour


def ensure_minimum_history(df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
    """
    Ensure we have enough historical data for feature calculation.
    
    Args:
        df: Price DataFrame
        config: ML configuration
        
    Returns:
        DataFrame with sufficient history
        
    Raises:
        ValueError: If insufficient data even after extending history
    """
    min_required = config.vol_window_hours + max(config.sma_windows) + max(config.momentum_windows)
    
    if len(df) < min_required:
        # Calculate how much more history we need
        shortage = min_required - len(df)
        extended_start = df.index.min() - pd.Timedelta(hours=shortage * bar_size_hours(config.interval))
        
        print(f"⚠️  Need {shortage} more bars for feature calculation")
        print(f"📅 Extending start date to {extended_start.strftime('%Y-%m-%d')}")
        
        # Create extended config
        extended_config = MLConfig(**config.to_dict())
        extended_config.start = extended_start.strftime('%Y-%m-%d')
        
        # Reload with extended range
        return load_and_validate_data(extended_config)
    
    return df 