"""
Multi-Asset Training
====================

Enhanced multi-cryptocurrency training functionality.
Improved version of original multi_crypto_ml_training.py.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .config import MLConfig
from .training import train_model


def train_multi_crypto_models(assets: List[str], base_config: MLConfig, 
                             parallel: bool = False, max_workers: int = 2) -> Dict:
    """
    Train separate ML models for multiple cryptocurrencies.
    
    Args:
        assets: List of cryptocurrency symbols
        base_config: Base configuration to use for all assets
        parallel: Whether to train models in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with training results for each asset
    """
    print(f"🚀 Training ML models for {len(assets)} cryptocurrencies")
    print(f"📊 Training mode: {base_config.training_mode}")
    print(f"⚙️  Parallel training: {parallel}")
    print("=" * 60)
    
    if parallel and len(assets) > 1:
        return _train_parallel(assets, base_config, max_workers)
    else:
        return _train_sequential(assets, base_config)


def _train_sequential(assets: List[str], base_config: MLConfig) -> Dict:
    """Train models sequentially for each asset."""
    all_results = {}
    all_signals = {}
    
    for i, asset in enumerate(assets, 1):
        print(f"\n📊 Training model for {asset} ({i}/{len(assets)})...")
        
        try:
            # Create asset-specific config
            asset_config = _create_asset_config(asset, base_config)
            
            # Train model
            start_time = time.time()
            results = train_model(asset_config)
            training_time = time.time() - start_time
            
            # Extract signals and metadata
            signals = results['signals']
            signals.name = asset
            
            all_results[asset] = {
                'results': results,
                'training_time': training_time,
                'signal_stats': signals.value_counts().to_dict(),
                'config': asset_config.to_dict()
            }
            all_signals[asset] = signals
            
            print(f"✅ {asset} complete in {training_time:.1f}s: {signals.value_counts().to_dict()}")
            
        except Exception as e:
            print(f"❌ Failed to train {asset}: {str(e)}")
            if base_config.verbose:
                print(traceback.format_exc())
            
            all_results[asset] = {
                'error': str(e),
                'training_time': 0,
                'signal_stats': {},
                'config': None
            }
            all_signals[asset] = pd.Series([], name=asset)
    
    # Combine and save results
    combined_results = _combine_and_save_results(all_signals, all_results, base_config)
    
    return combined_results


def _train_parallel(assets: List[str], base_config: MLConfig, max_workers: int) -> Dict:
    """Train models in parallel for each asset."""
    print(f"🔄 Using {max_workers} parallel workers")
    
    all_results = {}
    all_signals = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training jobs
        future_to_asset = {}
        for asset in assets:
            asset_config = _create_asset_config(asset, base_config)
            future = executor.submit(_train_single_asset, asset, asset_config)
            future_to_asset[future] = asset
        
        # Collect results as they complete
        for future in as_completed(future_to_asset):
            asset = future_to_asset[future]
            
            try:
                result = future.result()
                all_results[asset] = result
                
                if 'results' in result and 'signals' in result['results']:
                    signals = result['results']['signals']
                    signals.name = asset
                    all_signals[asset] = signals
                    
                    print(f"✅ {asset} complete in {result['training_time']:.1f}s: {result['signal_stats']}")
                else:
                    all_signals[asset] = pd.Series([], name=asset)
                    print(f"❌ {asset} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {asset} failed with exception: {str(e)}")
                all_results[asset] = {
                    'error': str(e),
                    'training_time': 0,
                    'signal_stats': {},
                    'config': None
                }
                all_signals[asset] = pd.Series([], name=asset)
    
    # Combine and save results
    combined_results = _combine_and_save_results(all_signals, all_results, base_config)
    
    return combined_results


def _train_single_asset(asset: str, config: MLConfig) -> Dict:
    """Train a single asset model (for parallel execution)."""
    try:
        start_time = time.time()
        results = train_model(config)
        training_time = time.time() - start_time
        
        signals = results['signals']
        
        return {
            'results': results,
            'training_time': training_time,
            'signal_stats': signals.value_counts().to_dict(),
            'config': config.to_dict()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'training_time': 0,
            'signal_stats': {},
            'config': None
        }


def _create_asset_config(asset: str, base_config: MLConfig) -> MLConfig:
    """Create asset-specific configuration."""
    # Start with base config
    asset_config = MLConfig(
        symbol=asset,
        start=base_config.start,
        end=base_config.end,
        interval=base_config.interval,
        forecast_horizon_hours=base_config.forecast_horizon_hours,
        vol_window_hours=base_config.vol_window_hours,
        
        # Feature engineering
        sma_windows=base_config.sma_windows,
        volatility_windows=base_config.volatility_windows,
        momentum_windows=base_config.momentum_windows,
        rsi_windows=base_config.rsi_windows,
        enable_regime_features=base_config.enable_regime_features,
        volatility_regime_window=base_config.volatility_regime_window,
        
        # Model architecture
        n_quantiles=base_config.n_quantiles,
        hidden_sizes=base_config.hidden_sizes,
        dropout_rate=base_config.dropout_rate,
        
        # Training parameters
        training_mode=base_config.training_mode,
        n_epochs=base_config.n_epochs,
        lr=base_config.lr,
        weight_decay=base_config.weight_decay,
        batch_size=base_config.batch_size,
        
        # Infrastructure
        cache_dir=base_config.cache_dir,
        device=base_config.device,
        verbose=base_config.verbose,
        random_seed=base_config.random_seed
    )
    
    return asset_config


def _combine_and_save_results(all_signals: Dict[str, pd.Series], 
                             all_results: Dict[str, Dict], 
                             base_config: MLConfig) -> Dict:
    """Combine results from all assets and save."""
    print(f"\n💾 Combining and saving results...")
    
    # Create combined signals DataFrame
    valid_signals = {asset: signals for asset, signals in all_signals.items() 
                    if not signals.empty}
    
    if valid_signals:
        signals_df = pd.DataFrame(valid_signals)
        
        # Save combined signals
        output_dir = base_config.cache_dir / "multi_asset"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signals_path = output_dir / f"multi_crypto_signals_{base_config.training_mode}.parquet"
        signals_df.to_parquet(signals_path)
        
        print(f"📊 Combined signals saved: {signals_path}")
        print(f"📊 Signal matrix shape: {signals_df.shape}")
    else:
        print("⚠️  No valid signals to combine")
        signals_df = pd.DataFrame()
    
    # Create summary statistics
    summary_stats = {}
    successful_assets = []
    failed_assets = []
    
    for asset, result in all_results.items():
        if 'error' in result:
            failed_assets.append(asset)
        else:
            successful_assets.append(asset)
            
            summary_stats[asset] = {
                'training_time': result['training_time'],
                'signal_distribution': result['signal_stats'],
                'total_signals': sum(result['signal_stats'].values()) if result['signal_stats'] else 0
            }
    
    # Overall summary
    total_training_time = sum(result['training_time'] for result in all_results.values())
    
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_assets': len(all_results),
        'successful_assets': len(successful_assets),
        'failed_assets': len(failed_assets),
        'success_rate': len(successful_assets) / len(all_results) if all_results else 0,
        'total_training_time': total_training_time,
        'average_training_time': total_training_time / len(successful_assets) if successful_assets else 0,
        'base_config': base_config.to_dict(),
        'asset_results': summary_stats,
        'successful_assets': successful_assets,
        'failed_assets': failed_assets
    }
    
    # Save summary
    if valid_signals:
        summary_path = output_dir / f"training_summary_{base_config.training_mode}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 Training summary saved: {summary_path}")
    
    # Print final statistics
    print(f"\n🏁 Multi-asset training complete!")
    print(f"   Total assets: {len(all_results)}")
    print(f"   Successful: {len(successful_assets)} ({len(successful_assets)/len(all_results)*100:.1f}%)")
    print(f"   Failed: {len(failed_assets)}")
    print(f"   Total time: {total_training_time:.1f}s")
    
    if successful_assets:
        print(f"   Average time per asset: {total_training_time/len(successful_assets):.1f}s")
        
        # Show signal statistics
        print(f"\n📊 Combined Signal Statistics:")
        if not signals_df.empty:
            total_signals_per_type = {}
            for signal_type in [-1, 0, 1]:
                count = (signals_df == signal_type).sum().sum()
                total_signals_per_type[signal_type] = count
            
            total_all_signals = sum(total_signals_per_type.values())
            if total_all_signals > 0:
                for signal_type, count in total_signals_per_type.items():
                    percentage = count / total_all_signals * 100
                    signal_name = {-1: "Short", 0: "Neutral", 1: "Long"}[signal_type]
                    print(f"   {signal_name}: {count:,} ({percentage:.1f}%)")
    
    if failed_assets:
        print(f"\n❌ Failed assets: {', '.join(failed_assets)}")
    
    return {
        'signals_df': signals_df,
        'summary': summary,
        'individual_results': all_results,
        'successful_assets': successful_assets,
        'failed_assets': failed_assets
    }


def analyze_multi_asset_signals(signals_df: pd.DataFrame, config: MLConfig) -> Dict:
    """
    Analyze signals across multiple assets for correlation and patterns.
    
    Args:
        signals_df: DataFrame with signals for multiple assets
        config: ML configuration
        
    Returns:
        Analysis results
    """
    if signals_df.empty:
        return {'error': 'No signals to analyze'}
    
    print("🔍 Analyzing multi-asset signal patterns...")
    
    analysis = {}
    
    # Basic statistics
    analysis['asset_count'] = len(signals_df.columns)
    analysis['time_period'] = {
        'start': str(signals_df.index.min()),
        'end': str(signals_df.index.max()),
        'total_periods': len(signals_df)
    }
    
    # Signal distribution by asset
    analysis['signal_distribution_by_asset'] = {}
    for asset in signals_df.columns:
        if not signals_df[asset].empty:
            dist = signals_df[asset].value_counts().to_dict()
            total = signals_df[asset].count()
            analysis['signal_distribution_by_asset'][asset] = {
                'distribution': dist,
                'total_signals': total,
                'percentages': {k: v/total*100 for k, v in dist.items()} if total > 0 else {}
            }
    
    # Cross-asset correlation
    try:
        correlation_matrix = signals_df.corr()
        analysis['signal_correlations'] = {
            'mean_correlation': float(correlation_matrix.mean().mean()),
            'max_correlation': float(correlation_matrix.max().max()),
            'min_correlation': float(correlation_matrix.min().min()),
            'correlation_matrix': correlation_matrix.to_dict()
        }
    except Exception as e:
        analysis['signal_correlations'] = {'error': str(e)}
    
    # Consensus signals (when multiple assets agree)
    if len(signals_df.columns) > 1:
        # Calculate consensus strength
        consensus_long = (signals_df == 1).sum(axis=1)
        consensus_short = (signals_df == -1).sum(axis=1) 
        consensus_neutral = (signals_df == 0).sum(axis=1)
        
        analysis['consensus_analysis'] = {
            'strong_long_consensus': int((consensus_long >= len(signals_df.columns) * 0.7).sum()),
            'strong_short_consensus': int((consensus_short >= len(signals_df.columns) * 0.7).sum()),
            'mixed_signals': int(((consensus_long > 0) & (consensus_short > 0)).sum()),
            'unanimous_long': int((consensus_long == len(signals_df.columns)).sum()),
            'unanimous_short': int((consensus_short == len(signals_df.columns)).sum())
        }
    
    # Time-based patterns
    if hasattr(signals_df.index, 'hour'):
        # Hourly patterns
        hourly_patterns = {}
        for hour in range(24):
            hour_data = signals_df[signals_df.index.hour == hour]
            if not hour_data.empty:
                hourly_patterns[hour] = {
                    'total_signals': int(hour_data.count().sum()),
                    'long_percentage': float((hour_data == 1).sum().sum() / hour_data.count().sum() * 100),
                    'short_percentage': float((hour_data == -1).sum().sum() / hour_data.count().sum() * 100)
                }
        analysis['hourly_patterns'] = hourly_patterns
    
    # Activity levels
    analysis['activity_analysis'] = {}
    for asset in signals_df.columns:
        if not signals_df[asset].empty:
            non_neutral = signals_df[asset] != 0
            analysis['activity_analysis'][asset] = {
                'activity_rate': float(non_neutral.mean()),
                'periods_with_signals': int(non_neutral.sum()),
                'consecutive_neutrals_max': int((~non_neutral).groupby((~non_neutral).cumsum()).cumcount().max() + 1) if not non_neutral.all() else len(signals_df)
            }
    
    print(f"✅ Multi-asset analysis complete for {len(signals_df.columns)} assets")
    return analysis


def create_asset_comparison_report(results: Dict, output_dir: Optional[Path] = None) -> str:
    """
    Create a comprehensive comparison report for multi-asset training.
    
    Args:
        results: Results from train_multi_crypto_models
        output_dir: Directory to save report (defaults to cache_dir)
        
    Returns:
        Path to generated report
    """
    if output_dir is None:
        output_dir = Path("artefacts/multi_asset")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML report
    html_content = _generate_html_report(results)
    
    report_path = output_dir / f"multi_asset_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html"
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📋 Asset comparison report saved: {report_path}")
    return str(report_path)


def _generate_html_report(results: Dict) -> str:
    """Generate HTML report content."""
    summary = results['summary']
    signals_df = results['signals_df']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Asset Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multi-Asset Training Report</h1>
            <p>Generated: {summary['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <ul>
                <li>Total Assets: {summary['total_assets']}</li>
                <li>Successful: <span class="success">{summary['successful_assets']}</span></li>
                <li>Failed: <span class="failure">{summary['failed_assets']}</span></li>
                <li>Success Rate: {summary['success_rate']*100:.1f}%</li>
                <li>Total Training Time: {summary['total_training_time']:.1f}s</li>
                <li>Average Time per Asset: {summary.get('average_training_time', 0):.1f}s</li>
            </ul>
        </div>
    """
    
    # Asset details table
    if summary['asset_results']:
        html += """
        <div class="section">
            <h2>Asset Training Details</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Training Time (s)</th>
                    <th>Total Signals</th>
                    <th>Long Signals</th>
                    <th>Short Signals</th>
                    <th>Neutral Signals</th>
                </tr>
        """
        
        for asset, stats in summary['asset_results'].items():
            dist = stats['signal_distribution']
            html += f"""
                <tr>
                    <td>{asset}</td>
                    <td>{stats['training_time']:.1f}</td>
                    <td>{stats['total_signals']}</td>
                    <td>{dist.get(1, 0)}</td>
                    <td>{dist.get(-1, 0)}</td>
                    <td>{dist.get(0, 0)}</td>
                </tr>
            """
        
        html += "</table></div>"
    
    # Failed assets
    if summary['failed_assets']:
        html += """
        <div class="section">
            <h2>Failed Assets</h2>
            <ul>
        """
        for asset in summary['failed_assets']:
            html += f"<li class='failure'>{asset}</li>"
        html += "</ul></div>"
    
    html += """
    </body>
    </html>
    """
    
    return html 