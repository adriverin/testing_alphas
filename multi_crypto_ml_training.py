"""
Multi-Cryptocurrency ML Training Script (Updated)
=================================================

Updated to use the new centralized ml_forecasting module.

This script now leverages the enhanced ml_forecasting module which provides:
- Unified configuration management
- Better error handling and parallel training support
- Enhanced signal analysis and model evaluation
- Backward compatibility with the original interface

For the legacy interface, see MIGRATION_GUIDE.md
"""

# Import from the new centralized module
from src.ml_forecasting import MLConfig, train_multi_crypto_models as train_models_new
import pandas as pd
from pathlib import Path

def train_multi_crypto_models(assets: list, base_config):
    """
    Legacy wrapper function for backward compatibility.
    
    This function maintains the original interface while using the new
    centralized ml_forecasting module under the hood.
    """
    
    print("🔄 Using legacy interface with new centralized module")
    print("ℹ️  Consider migrating to the new interface - see MIGRATION_GUIDE.md")
    
    # Convert legacy config to new MLConfig if needed
    if hasattr(base_config, 'enable_regime_features'):
        # Convert ImprovedConfig to MLConfig
        new_config = MLConfig.for_improved_training(
            start=base_config.start,
            end=base_config.end,
            interval=getattr(base_config, 'interval', '4h'),
            forecast_horizon_hours=getattr(base_config, 'forecast_horizon_hours', 6),
            vol_window_hours=getattr(base_config, 'vol_window_hours', 60),
            n_quantiles=base_config.n_quantiles,
            hidden_sizes=base_config.hidden_sizes,
            n_epochs=base_config.n_epochs,
            lr=base_config.lr,
            enable_regime_features=base_config.enable_regime_features
        )
    else:
        # Assume it's already an MLConfig
        new_config = base_config
    
    # Use the new centralized function
    results = train_models_new(
        assets=assets,
        base_config=new_config,
        parallel=False  # Keep sequential for backward compatibility
    )
    
    # Extract signals_df for backward compatibility
    signals_df = results['signals_df']
    
    # Save to legacy location for compatibility
    output_dir = Path("artefacts/improved_ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_dir / "multi_crypto_signals.parquet")
    
    print(f"💾 Saved to legacy location: {output_dir / 'multi_crypto_signals.parquet'}")
    print(f"📊 New enhanced results also available in: artefacts/multi_asset/")
    
    return signals_df







def create_maximum_cache_for_assets(assets: list, interval: str = "1h", 
                                   start: str = "2017-01-01", end: str = "2025-12-31",
                                   create_info_file: bool = True):
    """
    Utility function to pre-cache maximum feasible OHLCV data ranges for multiple assets.
    
    This creates the largest possible OHLCV cache files so future requests will always
    use cached data instead of downloading. Also creates an asset information file.
    
    Args:
        assets: List of cryptocurrency symbols
        interval: Time interval for data
        start: Earliest possible start date (will use actual exchange start if later)
        end: Latest possible end date (will use actual exchange end if earlier)
        create_info_file: Whether to create a detailed asset information file
        
    Usage:
        # Run this once to create maximum OHLCV cache files
        create_maximum_cache_for_assets(
            assets=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD'],
            interval="1h",
            start="2015-01-01",    # Go back as far as possible (will auto-adjust)
            end="2026-12-31",      # Go forward to include future requests
            create_info_file=True  # Creates asset_info.json with details
        )
    """
    from src.ml_forecasting import MLConfig
    from src.ml_forecasting.data_loader import _fetch_from_binance_ohlcv
    import json
    from datetime import datetime
    from pathlib import Path
    import pandas as pd
    
    print(f"🗄️  Creating maximum OHLCV cache files for {len(assets)} assets")
    print(f"📅 Requested range: {start} to {end} ({interval} interval)")
    print(f"🔍 Smart detection: Will find actual available data ranges")
    print("=" * 70)
    
    asset_info = {}
    cache_stats = {
        'creation_date': datetime.now().isoformat(),
        'requested_range': {'start': start, 'end': end},
        'interval': interval,
        'cache_type': 'ohlcv',  # Indicate this is OHLCV cache
        'total_assets': len(assets),
        'successful_assets': 0,
        'failed_assets': 0,
        'assets': {}
    }
    
    cache_dir = Path("artefacts/data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, asset in enumerate(assets, 1):
        print(f"\n📦 Caching {asset} ({i}/{len(assets)})...")
        
        try:
            config = MLConfig(
                symbol=asset,
                start=start,
                end=end,
                interval=interval
            )
            
            # Create cache file path for OHLCV data (unified cache)
            cache_file = cache_dir / f"ohlcv_{asset.replace('/', '').replace('-', '')}_{interval}.parquet"
            
            # Load OHLCV data (this handles smart date range detection automatically)
            df_ohlcv = _fetch_from_binance_ohlcv(config)
            
            if not df_ohlcv.empty:
                # Save to unified OHLCV cache
                df_ohlcv.to_parquet(cache_file)
                
                actual_start = df_ohlcv.index.min()
                actual_end = df_ohlcv.index.max()
                actual_start_str = actual_start.strftime('%Y-%m-%d')
                actual_end_str = actual_end.strftime('%Y-%m-%d')
                
                # Calculate data quality metrics
                total_hours_expected = int((actual_end - actual_start).total_seconds() / 3600)
                actual_hours = len(df_ohlcv)
                coverage_pct = (actual_hours / total_hours_expected) * 100 if total_hours_expected > 0 else 0
                
                # Check for gaps
                time_diffs = df_ohlcv.index.to_series().diff()
                expected_freq = pd.Timedelta(hours=1) if interval == '1h' else pd.Timedelta(hours=4)
                large_gaps = (time_diffs > expected_freq * 2).sum()
                
                asset_info[asset] = {
                    'status': 'success',
                    'cache_type': 'ohlcv',
                    'available_from': actual_start_str,
                    'available_to': actual_end_str,
                    'total_bars': len(df_ohlcv),
                    'data_coverage_pct': round(coverage_pct, 2),
                    'large_gaps_detected': large_gaps,
                    'trading_days': int((actual_end - actual_start).days),
                    'ohlcv_columns': list(df_ohlcv.columns),
                    'cache_file': f"ohlcv_{asset.replace('/', '').replace('-', '')}_{interval}.parquet"
                }
                
                cache_stats['successful_assets'] += 1
                cache_stats['assets'][asset] = asset_info[asset]
                
                print(f"✅ {asset}: {len(df_ohlcv):,} OHLCV bars cached")
                print(f"   📅 Available: {actual_start_str} to {actual_end_str}")
                print(f"   📊 Coverage: {coverage_pct:.1f}% ({large_gaps} gaps detected)")
                print(f"   📈 Columns: {list(df_ohlcv.columns)}")
                
                # Check if we got different dates than requested
                if actual_start_str != start:
                    print(f"   ℹ️  Exchange data starts {actual_start_str} (later than requested {start})")
                if actual_end_str != end:
                    print(f"   ℹ️  Exchange data ends {actual_end_str} (earlier than requested {end})")
                    
            else:
                asset_info[asset] = {
                    'status': 'no_data',
                    'error': 'No OHLCV data available for this symbol/timeframe'
                }
                cache_stats['failed_assets'] += 1
                cache_stats['assets'][asset] = asset_info[asset]
                print(f"⚠️  {asset}: No OHLCV data available")
                
        except Exception as e:
            error_msg = str(e)
            asset_info[asset] = {
                'status': 'error',
                'error': error_msg
            }
            cache_stats['failed_assets'] += 1
            cache_stats['assets'][asset] = asset_info[asset]
            print(f"❌ {asset}: Failed - {error_msg}")
    
    # Create comprehensive asset information file
    if create_info_file:
        info_file = cache_dir / f"asset_cache_info_{interval}.json"
        
        with open(info_file, 'w') as f:
            json.dump(cache_stats, f, indent=2, default=str)
        
        # Also create a readable summary
        summary_file = cache_dir / f"asset_cache_summary_{interval}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Cryptocurrency Asset OHLCV Cache Summary\n")
            f.write(f"=========================================\n\n")
            f.write(f"Created: {cache_stats['creation_date']}\n")
            f.write(f"Cache Type: OHLCV (Open, High, Low, Close, Volume)\n")
            f.write(f"Interval: {interval}\n")
            f.write(f"Requested Range: {start} to {end}\n")
            f.write(f"Total Assets: {len(assets)}\n")
            f.write(f"Successful: {cache_stats['successful_assets']}\n")
            f.write(f"Failed: {cache_stats['failed_assets']}\n\n")
            
            f.write("Asset Details:\n")
            f.write("-" * 50 + "\n")
            
            for asset, info in asset_info.items():
                f.write(f"\n{asset}:\n")
                if info['status'] == 'success':
                    f.write(f"  Status: ✅ Success\n")
                    f.write(f"  Type: OHLCV Cache\n")
                    f.write(f"  Available: {info['available_from']} to {info['available_to']}\n")
                    f.write(f"  Bars: {info['total_bars']:,}\n")
                    f.write(f"  Coverage: {info['data_coverage_pct']}%\n")
                    f.write(f"  Trading Days: {info['trading_days']:,}\n")
                    f.write(f"  Gaps: {info['large_gaps_detected']}\n")
                    f.write(f"  Columns: {', '.join(info['ohlcv_columns'])}\n")
                    f.write(f"  Cache File: {info['cache_file']}\n")
                else:
                    f.write(f"  Status: ❌ {info['status']}\n")
                    f.write(f"  Error: {info.get('error', 'Unknown error')}\n")
        
        print(f"\n📋 Asset information files created:")
        print(f"   📄 Detailed: {info_file}")
        print(f"   📄 Summary: {summary_file}")
    
    print(f"\n🎉 Maximum OHLCV cache creation complete!")
    print(f"📊 Results: {cache_stats['successful_assets']} successful, {cache_stats['failed_assets']} failed")
    print(f"💡 Future training runs and main.py will use cached OHLCV data for faster performance")
    print(f"🏷️ ML training can select any price column: open, high, low, close, vwap, typical, median")
    
    return asset_info


def load_asset_cache_info(interval: str = "1h") -> dict:
    """
    Load and display asset OHLCV cache information.
    
    Args:
        interval: Time interval to check info for
        
    Returns:
        Dictionary with asset cache information
    """
    from pathlib import Path
    import json
    
    info_file = Path(f"artefacts/data/asset_cache_info_{interval}.json")
    
    if not info_file.exists():
        print(f"❌ No cache info file found: {info_file}")
        print(f"💡 Run create_maximum_cache_for_assets() first to generate cache info")
        return {}
    
    with open(info_file, 'r') as f:
        data = json.load(f)
    
    cache_type = data.get('cache_type', 'unknown')
    print(f"📋 Asset {cache_type.upper()} Cache Information ({interval} interval)")
    print("=" * 70)
    print(f"Created: {data['creation_date']}")
    print(f"Cache Type: {cache_type.upper()} (Open, High, Low, Close, Volume)")
    print(f"Requested Range: {data['requested_range']['start']} to {data['requested_range']['end']}")
    print(f"Success Rate: {data['successful_assets']}/{data['total_assets']} assets")
    print()
    
    successful_assets = []
    failed_assets = []
    
    for asset, info in data['assets'].items():
        if info['status'] == 'success':
            successful_assets.append((asset, info))
            print(f"✅ {asset}:")
            print(f"   📅 Available: {info['available_from']} to {info['available_to']}")
            print(f"   📊 Quality: {info['data_coverage_pct']}% coverage, {info['total_bars']:,} bars")
            if 'ohlcv_columns' in info:
                print(f"   📈 Columns: {', '.join(info['ohlcv_columns'])}")
            print(f"   🗃️  Cache: {info['cache_file']}")
            if int(info['large_gaps_detected']) > 0:
                print(f"   ⚠️  {info['large_gaps_detected']} gaps detected")
            print()
        else:
            failed_assets.append((asset, info))
    
    if failed_assets:
        print("❌ Failed Assets:")
        for asset, info in failed_assets:
            print(f"   {asset}: {info.get('error', 'Unknown error')}")
        print()
    
    # Summary of date ranges
    if successful_assets:
        print("📅 Available Date Ranges Summary:")
        for asset, info in successful_assets:
            print(f"   {asset}: {info['available_from']} → {info['available_to']}")
        print()
    
    # ML Training Usage Instructions
    if cache_type == 'ohlcv':
        print("🏷️  ML Training Price Column Options:")
        print("    • 'close': Close price (default)")
        print("    • 'open': Open price") 
        print("    • 'high': High price")
        print("    • 'low': Low price")
        print("    • 'vwap': Calculated VWAP (H+L+C)/3")
        print("    • 'typical': Same as VWAP (H+L+C)/3")
        print("    • 'median': Median price (H+L)/2")
        print()
        print("💡 Example: config = MLConfig(symbol='BTC-USD', price_column='vwap')")
    
    return data


# Uncomment to run maximum cache creation:
# if __name__ == "__main__":
#     # Option 1: Create maximum cache files first (run this once)
#     create_maximum_cache_for_assets(
#         assets=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD', 'PEPE-USD'],
#         interval="1h",
#         start="2017-01-01",
#         end="2025-12-31"
#     )
#     
#     # Option 2: Load and display existing cache info
#     load_asset_cache_info("1h")



# Usage
if __name__ == "__main__":
    print("🚀 Multi-Crypto ML Training Script")
    print("=" * 50)


    # crypto_assets_new = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    crypto_assets_new = ['DOGE-USD', 'PEPE-USD', 'SHIB-USD', 'FLOKI-USD']
    
    base_config_new = MLConfig.for_improved_training(
        start="2023-05-05", 
        end="2025-01-01",  
        interval="1h",
        price_column="typical",  # Options: "open", "high", "low", "close", "vwap", "typical", "median"
        forecast_horizon_hours=1,
        vol_window_hours=24,
        n_quantiles=5,
        hidden_sizes=(32, 16, 8),
        n_epochs=100,  
        lr=5e-5,
        enable_regime_features=True,
        verbose=False,
        sma_windows=(3, 5, 10, 20, 30),
        volatility_windows=(5, 10, 20, 30),
        momentum_windows=(7, 14, 21, 30, 40, 50),
        rsi_windows=(3, 7, 14, 21),
        signal_percentiles=(5, 95),
        train_ratio = 0.75,     # For improved mode
        val_ratio = 0.1,       # For improved mode
        test_ratio = 0.15      # For improved mode        
    )

    # FOLLOWING PREDICTS 36% OF QUINTILES??? STILL LEAD TO BAD RETURNS SOMEHOW
    # crypto_assets_new = ['PEPE-USD']
    
    # base_config_new = MLConfig.for_improved_training(
    #     start="2023-06-01", 
    #     end="2024-12-31",  
    #     interval="1h",
    #     forecast_horizon_hours=2,
    #     vol_window_hours=24,
    #     n_quantiles=5,
    #     hidden_sizes=(32, 16, 8),
    #     n_epochs=100,  # Reduced for demo
    #     lr=5e-5,
    #     enable_regime_features=True,
    #     verbose=False
    # )
    
    # Use new centralized interface directly
    results_new = train_models_new(
        assets=crypto_assets_new,
        base_config=base_config_new,
        parallel=False,  # Enable parallel training
        max_workers=1
    )
    
    print(f"✅ New interface complete:")
    print(f"   Signals shape: {results_new['signals_df'].shape}")
    print(f"   Successful assets: {results_new['summary']['successful_assets']}")
    print(f"   Training time: {results_new['summary']['total_training_time']:.1f}s")
    
    # print("\n🎉 Multi-crypto training demonstration complete!")
    # print("📋 Check MIGRATION_GUIDE.md for full migration instructions")

