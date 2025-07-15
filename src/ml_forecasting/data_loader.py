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
    Load price history with intelligent incremental caching using unified OHLCV cache.
    
    Args:
        config: ML configuration object
        
    Returns:
        DataFrame with timestamp index and selected price column
        
    Raises:
        RuntimeError: If no data can be fetched
    """
    # Create cache file path for OHLCV data (unified with main.py)
    cache_file = (
        config.data_cache_dir / 
        f"ohlcv_{config.symbol.replace('/', '').replace('-', '')}_{config.interval}.parquet"
    )
    
    start_date_dt = pd.to_datetime(config.start, utc=True)
    end_date_dt = pd.to_datetime(config.end, utc=True)
    
    print(f"📅 Requested date range: {config.start} to {config.end}")
    print(f"📊 Using price column: {config.price_column}")
    
    # **INTELLIGENT INCREMENTAL CACHE VALIDATION**
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            
            if not cached_df.empty:
                cache_start = cached_df.index.min()
                cache_end = cached_df.index.max()
                
                print(f"📊 Cached OHLCV range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
                
                # **NEW: Check if requested range is FULLY COVERED by cache**
                if start_date_dt >= cache_start and end_date_dt <= cache_end:
                    print(f"✅ OHLCV cache fully covers requested range. Using cached data.")
                    # Filter to requested range and return selected price column
                    mask = (cached_df.index >= start_date_dt) & (cached_df.index <= end_date_dt)
                    filtered_df = cached_df[mask]
                    return _extract_price_column(filtered_df, config.price_column)
                
                # **NEW: INCREMENTAL CACHING - Check for partial overlap**
                cache_overlap = not (end_date_dt < cache_start or start_date_dt > cache_end)
                
                if cache_overlap:
                    print(f"📊 Partial OHLCV cache coverage detected. Using incremental download strategy...")
                    
                    # Calculate missing data ranges
                    missing_ranges = []
                    
                    # Missing data before cache
                    if start_date_dt < cache_start:
                        gap_start = start_date_dt
                        gap_end = min(cache_start - pd.Timedelta(hours=1), end_date_dt)
                        missing_ranges.append((gap_start, gap_end, "before"))
                        print(f"   • Missing before cache: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
                    
                    # Missing data after cache  
                    if end_date_dt > cache_end:
                        gap_start = max(cache_end + pd.Timedelta(hours=1), start_date_dt)
                        gap_end = end_date_dt
                        missing_ranges.append((gap_start, gap_end, "after"))
                        print(f"   • Missing after cache: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
                    
                    # Download missing data and merge with cache
                    if missing_ranges:
                        return _download_and_merge_missing_ohlcv_data(cached_df, missing_ranges, config, cache_file)
                    else:
                        # Full overlap, just filter cached data
                        mask = (cached_df.index >= start_date_dt) & (cached_df.index <= end_date_dt)
                        filtered_df = cached_df[mask]
                        return _extract_price_column(filtered_df, config.price_column)
                else:
                    print(f"🔄 No overlap with cached OHLCV data. Fetching fresh data...")
            else:
                print(f"🔄 OHLCV cache file is empty. Fetching fresh data...")
                
        except Exception as e:
            print(f"🔄 Error loading OHLCV cache: {e}. Fetching fresh data...")
    else:
        print(f"📂 No OHLCV cache file found. Fetching fresh data...")
    
    # Fetch from exchange (fallback for no cache or no overlap)
    print(f"🌐 Fetching {config.symbol} OHLCV data from Binance...")
    df_ohlcv = _fetch_from_binance_ohlcv(config)
    
    # Save full OHLCV to cache
    try:
        df_ohlcv.to_parquet(cache_file)
        actual_start = df_ohlcv.index.min()
        actual_end = df_ohlcv.index.max()
        print(f"💾 Saved {len(df_ohlcv)} OHLCV bars to cache: {cache_file.name}")
        print(f"✅ Downloaded and cached: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save to cache: {e}")
    
    # Return selected price column
    return _extract_price_column(df_ohlcv, config.price_column)


def _download_and_merge_missing_data(cached_df: pd.DataFrame, missing_ranges: list, 
                                   config: MLConfig, cache_file: Path) -> pd.DataFrame:
    """
    Download missing data ranges and merge with cached data.
    
    Args:
        cached_df: Existing cached DataFrame
        missing_ranges: List of (start, end, location) tuples for missing data
        config: ML configuration
        cache_file: Path to cache file for updating
        
    Returns:
        Combined DataFrame with all requested data
    """
    print(f"📥 Downloading {len(missing_ranges)} missing data range(s)...")
    
    all_dataframes = [cached_df]
    total_new_rows = 0
    
    for gap_start, gap_end, location in missing_ranges:
        try:
            print(f"   🌐 Fetching {location} gap: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            
            # Create temporary config for this range with only constructor arguments
            gap_config = MLConfig(
                symbol=config.symbol,
                start=gap_start.strftime('%Y-%m-%d'),
                end=gap_end.strftime('%Y-%m-%d'),
                interval=config.interval,
                price_column=config.price_column,  # Copy price column
                forecast_horizon_hours=config.forecast_horizon_hours,
                vol_window_hours=config.vol_window_hours,
                sma_windows=config.sma_windows,
                volatility_windows=config.volatility_windows,
                momentum_windows=config.momentum_windows,
                rsi_windows=config.rsi_windows,
                enable_regime_features=config.enable_regime_features,
                volatility_regime_window=config.volatility_regime_window,
                feature_stability_window=config.feature_stability_window,
                max_feature_drift=config.max_feature_drift,
                n_quantiles=config.n_quantiles,
                hidden_sizes=config.hidden_sizes,
                dropout_rate=config.dropout_rate,
                training_mode=config.training_mode,
                n_epochs=config.n_epochs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                batch_size=config.batch_size,
                test_fraction=config.test_fraction,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                min_train_samples=config.min_train_samples,
                validation_months=config.validation_months,
                walk_forward_step=config.walk_forward_step,
                n_ensemble_models=config.n_ensemble_models,
                early_stopping_patience=config.early_stopping_patience,
                min_improvement=config.min_improvement,
                threshold=config.threshold,
                signal_percentiles=config.signal_percentiles,
                cache_dir=config.cache_dir,
                device=config.device,
                verbose=config.verbose,
                plot_reliability=config.plot_reliability,
                random_seed=config.random_seed
            )
            
            # Download missing data
            gap_df = _fetch_from_binance(gap_config)
            
            if not gap_df.empty:
                all_dataframes.append(gap_df)
                total_new_rows += len(gap_df)
                print(f"   ✅ Downloaded {len(gap_df)} rows for {location} gap")
            else:
                print(f"   ⚠️  No data available for {location} gap")
                
        except Exception as e:
            print(f"   ❌ Failed to download {location} gap: {e}")
            continue
    
    # Merge all data
    print(f"🔗 Merging cached data with {total_new_rows} new rows...")
    combined_df = pd.concat(all_dataframes, axis=0)
    combined_df = combined_df[~combined_df.index.duplicated()].sort_index()
    
    # Update cache with combined data
    try:
        combined_df.to_parquet(cache_file)
        print(f"💾 Updated cache with combined data: {len(combined_df)} total rows")
        print(f"📅 New cache range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not update cache: {e}")
    
    # Return only the requested date range
    start_date_dt = pd.to_datetime(config.start, utc=True)
    end_date_dt = pd.to_datetime(config.end, utc=True)
    mask = (combined_df.index >= start_date_dt) & (combined_df.index <= end_date_dt)
    result_df = combined_df[mask]
    
    print(f"✅ Returning {len(result_df)} rows for requested range")
    return result_df


def _download_and_merge_missing_ohlcv_data(cached_df: pd.DataFrame, missing_ranges: list, 
                                   config: MLConfig, cache_file: Path) -> pd.DataFrame:
    """
    Download missing OHLCV data ranges and merge with cached data.
    
    Args:
        cached_df: Existing cached DataFrame
        missing_ranges: List of (start, end, location) tuples for missing data
        config: ML configuration
        cache_file: Path to cache file for updating
        
    Returns:
        Combined DataFrame with all requested data
    """
    print(f"📥 Downloading {len(missing_ranges)} missing OHLCV data range(s)...")
    
    all_dataframes = [cached_df]
    total_new_rows = 0
    
    for gap_start, gap_end, location in missing_ranges:
        try:
            print(f"   🌐 Fetching {location} gap: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            
            # Create temporary config for this range with only constructor arguments
            gap_config = MLConfig(
                symbol=config.symbol,
                start=gap_start.strftime('%Y-%m-%d'),
                end=gap_end.strftime('%Y-%m-%d'),
                interval=config.interval,
                price_column=config.price_column,  # Copy price column
                forecast_horizon_hours=config.forecast_horizon_hours,
                vol_window_hours=config.vol_window_hours,
                sma_windows=config.sma_windows,
                volatility_windows=config.volatility_windows,
                momentum_windows=config.momentum_windows,
                rsi_windows=config.rsi_windows,
                enable_regime_features=config.enable_regime_features,
                volatility_regime_window=config.volatility_regime_window,
                feature_stability_window=config.feature_stability_window,
                max_feature_drift=config.max_feature_drift,
                n_quantiles=config.n_quantiles,
                hidden_sizes=config.hidden_sizes,
                dropout_rate=config.dropout_rate,
                training_mode=config.training_mode,
                n_epochs=config.n_epochs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                batch_size=config.batch_size,
                test_fraction=config.test_fraction,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                min_train_samples=config.min_train_samples,
                validation_months=config.validation_months,
                walk_forward_step=config.walk_forward_step,
                n_ensemble_models=config.n_ensemble_models,
                early_stopping_patience=config.early_stopping_patience,
                min_improvement=config.min_improvement,
                threshold=config.threshold,
                signal_percentiles=config.signal_percentiles,
                cache_dir=config.cache_dir,
                device=config.device,
                verbose=config.verbose,
                plot_reliability=config.plot_reliability,
                random_seed=config.random_seed
            )
            
            # Download missing data
            gap_df = _fetch_from_binance_ohlcv(gap_config)
            
            if not gap_df.empty:
                all_dataframes.append(gap_df)
                total_new_rows += len(gap_df)
                print(f"   ✅ Downloaded {len(gap_df)} OHLCV bars for {location} gap")
            else:
                print(f"   ⚠️  No OHLCV data available for {location} gap")
                
        except Exception as e:
            print(f"   ❌ Failed to download {location} gap: {e}")
            continue
    
    # Merge all data
    print(f"🔗 Merging cached OHLCV data with {total_new_rows} new OHLCV bars...")
    combined_df = pd.concat(all_dataframes, axis=0)
    combined_df = combined_df[~combined_df.index.duplicated()].sort_index()
    
    # Update cache with combined data
    try:
        combined_df.to_parquet(cache_file)
        print(f"💾 Updated OHLCV cache with combined data: {len(combined_df)} total OHLCV bars")
        print(f"📅 New OHLCV cache range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not update OHLCV cache: {e}")
    
    # Return only the requested date range with selected price column
    start_date_dt = pd.to_datetime(config.start, utc=True)
    end_date_dt = pd.to_datetime(config.end, utc=True)
    mask = (combined_df.index >= start_date_dt) & (combined_df.index <= end_date_dt)
    result_df = combined_df[mask]
    
    print(f"✅ Returning {len(result_df)} OHLCV bars for requested range")
    return _extract_price_column(result_df, config.price_column)


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


def _fetch_from_binance_ohlcv(config: MLConfig) -> pd.DataFrame:
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
        
        print(f"✅ Successfully fetched {len(df)} OHLCV bars")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OHLCV data from Binance: {e}")


def _extract_price_column(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    Extract a specific price column from an OHLCV DataFrame.
    Supports calculated prices like VWAP.
    
    Args:
        df: DataFrame with OHLCV columns
        price_column: Name of the price column to extract:
                     - "open", "high", "low", "close": Direct OHLC columns
                     - "vwap": Calculate VWAP as (H+L+C)/3
                     - "typical": Calculate typical price as (H+L+C)/3
                     - "median": Calculate median price as (H+L)/2
        
    Returns:
        DataFrame with timestamp index and the selected price column
    """
    df_copy = df.copy()
    
    if price_column in ["vwap", "typical"]:
        # Calculate VWAP/typical price as (High + Low + Close) / 3
        if all(col in df_copy.columns for col in ['high', 'low', 'close']):
            df_copy[price_column] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        else:
            missing = [col for col in ['high', 'low', 'close'] if col not in df_copy.columns]
            raise ValueError(f"Cannot calculate {price_column}: missing columns {missing}")
            
    elif price_column == "median":
        # Calculate median price as (High + Low) / 2
        if all(col in df_copy.columns for col in ['high', 'low']):
            df_copy[price_column] = (df_copy['high'] + df_copy['low']) / 2
        else:
            missing = [col for col in ['high', 'low'] if col not in df_copy.columns]
            raise ValueError(f"Cannot calculate {price_column}: missing columns {missing}")
            
    elif price_column not in df_copy.columns:
        available_cols = list(df_copy.columns)
        raise ValueError(f"Price column '{price_column}' not found. Available: {available_cols}")
    
    # Return DataFrame with only the selected price column
    return df_copy[[price_column]]


def validate_data(df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
    """
    Validate and clean price data.
    
    Args:
        df: Raw price DataFrame with selected price column
        config: ML configuration object
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        ValueError: If data is insufficient or invalid
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    
    # Get the price column name (should be the only column after extraction)
    price_col = df.columns[0] if len(df.columns) == 1 else config.price_column
    
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    
    # Check for minimum data points
    min_required = max(100, config.vol_window_hours * 2)  # At least 2x volatility window
    if len(df) < min_required:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_required}")
    
    # Remove rows with invalid prices
    initial_len = len(df)
    df = df[df[price_col] > 0]  # Remove zero/negative prices
    df = df.dropna(subset=[price_col])  # Remove NaN prices
    
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
        df: Price DataFrame with selected price column
        config: ML configuration object
        
    Returns:
        DataFrame with basic preprocessing applied
    """
    df = df.copy()
    
    # Get the price column name (should be the only column after extraction)
    price_col = df.columns[0] if len(df.columns) == 1 else config.price_column
    
    # Basic return calculation
    df['return'] = df[price_col].pct_change()
    
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