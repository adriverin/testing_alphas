"""
Data Loading and Preprocessing
==============================

Centralized data loading functionality for ML forecasting.
Extracted and improved from original ml_forecast_prob_dist.py.
"""

import pandas as pd
import numpy as np
import ccxt
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from .config import MLConfig


def bar_size_hours(interval: str) -> float:
    """
    Convert interval string to hours.
    
    Args:
        interval: Time interval string (e.g., '1m', '5m', '1h', '4h', '1d')
        
    Returns:
        Number of hours in the interval
        
    Examples:
        >>> bar_size_hours('1m')
        0.016666666666666666
        >>> bar_size_hours('1h') 
        1.0
        >>> bar_size_hours('1d')
        24.0
    """
    if interval.endswith('m'):
        return int(interval[:-1]) / 60
    elif interval.endswith('h'):
        return int(interval[:-1])
    elif interval.endswith('d'):
        return int(interval[:-1]) * 24
    else:
        raise ValueError(f"Unknown interval format: {interval}")


def load_price_history(config: MLConfig) -> pd.DataFrame:
    """
    Load price history from cache or fetch from exchange.
    
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
    
    # Try loading from cache first
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                print(f"📁 Loaded {len(df)} rows from cache: {cache_file.name}")
                
                # Validate date range
                if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                    cache_start = df.index.min()
                    cache_end = df.index.max()
                    requested_start = pd.to_datetime(config.start, utc=True)
                    requested_end = pd.to_datetime(config.end, utc=True)
                    
                    # Check if cache covers requested range
                    if cache_start <= requested_start and cache_end >= requested_end:
                        # Filter to requested range
                        mask = (df.index >= requested_start) & (df.index <= requested_end)
                        return df[mask]
                    else:
                        print(f"⚠️  Cache range ({cache_start} to {cache_end}) doesn't cover requested range ({requested_start} to {requested_end})")
                        print("🔄 Fetching fresh data...")
                else:
                    return df
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            print("🔄 Fetching fresh data...")
    
    # Fetch from exchange
    print(f"🌐 Fetching {config.symbol} data from Binance...")
    df = _fetch_from_binance(config)
    
    # Save to cache
    try:
        df.to_parquet(cache_file)
        print(f"💾 Saved {len(df)} rows to cache: {cache_file.name}")
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


# Utility functions for backward compatibility
def load_and_validate_data(config: MLConfig) -> pd.DataFrame:
    """
    Complete data loading and validation pipeline.
    
    Args:
        config: ML configuration object
        
    Returns:
        Clean, validated DataFrame ready for feature engineering
    """
    # Load data
    df = load_price_history(config)
    
    # Validate
    df = validate_data(df, config)
    
    # Preprocess
    df = preprocess_data(df, config)
    
    print(f"✅ Data loading complete: {len(df)} clean rows")
    return df 