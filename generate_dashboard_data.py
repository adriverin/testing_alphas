#!/usr/bin/env python3
"""
Generate Trading Strategy Dashboard Data

This script runs backtests using the existing main.py infrastructure and exports
the results in JSON format for consumption by the HTML dashboard.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import math
warnings.filterwarnings('ignore')

# Import the backtesting infrastructure
from main import (
    get_crypto_data_unified, 
    detect_ml_price_column,
    tickers as default_tickers,
    start_date as default_start_date,
    end_date as default_end_date
)

from src.alpha101 import Alpha101
from src.backtests import run_rank_backtest
from src.reporting import analyze_performance


def clean_nan_values(obj):
    """Recursively clean NaN and infinite values from data structure"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # Replace NaN/inf with 0
        return obj
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0  # Replace NaN/inf with 0
        return float(obj)  # Convert numpy float to Python float
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert numpy int to Python int
    else:
        return obj


def run_backtest_for_dashboard(tickers=None, start_date=None, end_date=None, 
                              alpha_name='alpha999', interval='1d', 
                              stop_loss_pct=None, price_column=None):
    """
    Run a complete backtest and return data formatted for the dashboard.
    Modified to use cached signals like interval analysis, avoiding probability file requirements.
    
    Args:
        tickers: List of crypto symbols (e.g., ['BTC-USD', 'ETH-USD'])
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        alpha_name: Name of alpha to test (default: 'alpha999' for ML)
        interval: Data interval ('1d', '1h', etc.)
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0)
        price_column: Price column for returns calculation
        
    Returns:
        dict: Formatted data for dashboard consumption
    """
    # Use defaults if not provided
    if tickers is None:
        tickers = default_tickers
    if start_date is None:
        start_date = default_start_date
    if end_date is None:
        end_date = default_end_date
    
    print(f"🚀 Running backtest for dashboard: {alpha_name}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🎯 Assets: {tickers}")
    
    # Detect price column from ML models
    if price_column is None:
        crypto_tickers = [t for t in tickers if '-USD' in t]
        price_column = detect_ml_price_column(crypto_tickers)
    
    # Load data
    print("📊 Loading price data...")
    price_data = get_crypto_data_unified(tickers, start_date, end_date, interval, price_column)
    
    if price_data.empty:
        raise ValueError("No price data loaded")
    
    # Initialize alpha calculator
    print("🧮 Initializing alpha calculator...")
    alpha_calculator = Alpha101(price_data)
    
    # Generate alpha signals using the same approach as interval analysis
    print(f"📡 Generating {alpha_name} signals (using real-time prediction like interval analysis)...")
    
    try:
        # Calculate the full alpha series once (same as interval analysis)
        if alpha_name == 'alpha999':
            # Use regular alpha999 which will generate fresh predictions if no cached signals for current assets/period
            full_alpha_series = alpha_calculator.alpha999().dropna()
        elif alpha_name == 'alpha999_dynamic':
            # Use alpha999_dynamic with default percentiles
            full_alpha_series = alpha_calculator.alpha999_dynamic(percentiles=(5, 95)).dropna()
        else:
            # Use traditional alpha
            if not hasattr(alpha_calculator, alpha_name):
                raise ValueError(f"Alpha {alpha_name} not found")
            full_alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
        
        # Ensure the calculated alpha series is sorted
        if not full_alpha_series.index.is_monotonic_increasing:
            full_alpha_series = full_alpha_series.sort_index()
            
    except Exception as e:
        print(f"❌ FAILED to calculate alpha series for {alpha_name}: {e}")
        raise ValueError(f"Failed to generate {alpha_name} signals: {e}")

    if full_alpha_series.empty:
        print(f"⚠️  No signals generated for {alpha_name}")
        # Check if we have cached signals and what date range they cover
        from pathlib import Path
        artifacts_dir = Path("artefacts")
        signals_dir = artifacts_dir / "signals"
        multi_asset_dir = artifacts_dir / "multi_asset"
        
        print(f"📁 Checking for cached signal files:")
        possible_paths = [
            multi_asset_dir / "multi_crypto_signals_improved.parquet",
            signals_dir / f"{tickers[0]}_improved_signals.parquet",
            artifacts_dir / "improved_ml" / "improved_trading_signals.parquet"
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    cached_signals = pd.read_parquet(path)
                    print(f"   ✅ Found: {path}")
                    print(f"   📅 Date range: {cached_signals.index.min()} to {cached_signals.index.max()}")
                except:
                    print(f"   ❌ Error reading: {path}")
            else:
                print(f"   ❌ Missing: {path}")
        
        raise ValueError(f"No signals generated for {alpha_name}. Check cached signal files above.")
    
    print(f"✅ Loaded {len(full_alpha_series)} cached signals")
    
    # Filter signals to the requested date range (same as interval analysis)
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    
    print(f"📅 Filtering signals to requested range: {start_dt} to {end_dt}")
    
    # Filter both price data and alpha series to the same date range
    try:
        filtered_price_data = price_data.loc[pd.IndexSlice[start_dt:end_dt, :]]
        filtered_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_dt:end_dt, :]]
        
        print(f"📊 Filtered price data: {len(filtered_price_data)} rows")
        print(f"📊 Filtered signals: {len(filtered_alpha_series)} rows")
        
        if filtered_alpha_series.empty or filtered_price_data.empty:
            print(f"⚠️  No data in requested date range")
            print(f"   Price data range: {price_data.index.get_level_values('date').min()} to {price_data.index.get_level_values('date').max()}")
            print(f"   Signal range: {full_alpha_series.index.get_level_values('date').min()} to {full_alpha_series.index.get_level_values('date').max()}")
            raise ValueError(f"No data available for date range {start_date} to {end_date}")
            
    except KeyError as e:
        print(f"❌ Date filtering failed: {e}")
        print(f"Available price data dates: {price_data.index.get_level_values('date').min()} to {price_data.index.get_level_values('date').max()}")
        print(f"Available signal dates: {full_alpha_series.index.get_level_values('date').min()} to {full_alpha_series.index.get_level_values('date').max()}")
        raise ValueError(f"Date range {start_date} to {end_date} not available in data")
    
    # Use the filtered data for backtesting
    alpha_series = filtered_alpha_series
    price_data = filtered_price_data
    
    # Run backtest
    print("🔄 Running backtest...")
    strategy_returns, portfolio_info = run_rank_backtest(price_data, alpha_series, stop_loss_pct)
    
    if strategy_returns.empty:
        raise ValueError("Backtest produced no returns")
    
    # Calculate net returns (after transaction costs)
    daily_turnover = portfolio_info['turnover'].groupby('date').first()
    daily_cost = daily_turnover * (5 / 10000.0)  # 5 bps transaction costs
    strategy_returns_net = strategy_returns - daily_cost.reindex(strategy_returns.index).fillna(0)
    
    # Calculate benchmark returns
    benchmark_returns = price_data['returns'].groupby(level='date').mean()
    benchmark_returns_aligned = benchmark_returns.reindex(strategy_returns.index).fillna(0)
    
    # Get performance metrics from analyze_performance
    print("📈 Calculating performance metrics...")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    
    perf_metrics = analyze_performance(
        strategy_returns, 
        portfolio_info, 
        price_data, 
        fig=fig, 
        title=f"{alpha_name} Performance"
    )
    plt.close(fig)  # Close figure to save memory
    
    # Calculate additional metrics
    print("🧮 Computing additional metrics...")
    
    # Periods per year (for annualization)
    if interval == '1d':
        periods_per_year = 252
    elif interval == '1h':
        periods_per_year = 252 * 24
    elif interval == '15m':
        periods_per_year = 252 * 24 * 4
    else:
        periods_per_year = 252  # Default assumption
    
    # Additional calculations
    total_return_strategy = (1 + strategy_returns_net).prod() - 1
    total_return_benchmark = (1 + benchmark_returns_aligned).prod() - 1
    
    volatility_strategy = strategy_returns_net.std() * np.sqrt(periods_per_year)
    volatility_benchmark = benchmark_returns_aligned.std() * np.sqrt(periods_per_year)
    
    # Drawdown calculations
    cumulative_strategy = (1 + strategy_returns_net).cumprod()
    peak_strategy = cumulative_strategy.expanding(min_periods=1).max()
    drawdown_strategy = (cumulative_strategy / peak_strategy - 1) * 100
    
    cumulative_benchmark = (1 + benchmark_returns_aligned).cumprod()
    peak_benchmark = cumulative_benchmark.expanding(min_periods=1).max()
    drawdown_benchmark = (cumulative_benchmark / peak_benchmark - 1) * 100
    
    # Rolling metrics (252-day window for daily data)
    rolling_window = min(252, len(strategy_returns_net) // 4)  # Adaptive window
    
    def rolling_sharpe(returns, window):
        return returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(periods_per_year)
    
    def rolling_sortino(returns, window):
        downside = returns.where(returns < 0, 0)
        downside_std = downside.rolling(window).std()
        return returns.rolling(window).mean() / downside_std * np.sqrt(periods_per_year)
    
    def rolling_info_ratio(strategy_ret, benchmark_ret, window):
        excess_ret = strategy_ret - benchmark_ret
        tracking_error = excess_ret.rolling(window).std()
        return excess_ret.rolling(window).mean() / tracking_error * np.sqrt(periods_per_year)
    
    rolling_sharpe_strategy = rolling_sharpe(strategy_returns_net, rolling_window)
    rolling_sortino_strategy = rolling_sortino(strategy_returns_net, rolling_window)
    rolling_info_ratio_strategy = rolling_info_ratio(strategy_returns_net, benchmark_returns_aligned, rolling_window)
    
    # Monthly returns analysis
    monthly_returns = strategy_returns_net.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_benchmark = benchmark_returns_aligned.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Find worst drawdown periods
    def find_worst_drawdowns(drawdown_series, n=5):
        """Find the worst n drawdown periods"""
        drawdowns = []
        in_drawdown = False
        start_idx = None
        min_dd = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
                min_dd = dd
            elif dd < min_dd and in_drawdown:
                min_dd = dd
            elif dd >= 0 and in_drawdown:
                drawdowns.append({
                    'start': drawdown_series.index[start_idx],
                    'end': drawdown_series.index[i-1],
                    'magnitude': min_dd,
                    'duration': i - start_idx
                })
                in_drawdown = False
        
        return sorted(drawdowns, key=lambda x: x['magnitude'])[:n]
    
    worst_drawdowns = find_worst_drawdowns(drawdown_strategy)
    
    # Calculate trade statistics
    print("📊 Extracting trade information...")
    weights = portfolio_info['weights']
    num_positions = len(weights[weights != 0])
    avg_position_size = weights[weights != 0].abs().mean() if num_positions > 0 else 0
    
    # Win rate approximation (daily level)
    winning_days = len(strategy_returns_net[strategy_returns_net > 0])
    total_days = len(strategy_returns_net)
    daily_win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
    
    # Skewness and kurtosis
    from scipy import stats
    skewness = stats.skew(strategy_returns_net.dropna())
    kurtosis = stats.kurtosis(strategy_returns_net.dropna())
    
    # Beta calculation (simplified)
    covariance = np.cov(strategy_returns_net.dropna(), benchmark_returns_aligned.dropna())[0, 1]
    benchmark_variance = np.var(benchmark_returns_aligned.dropna())
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Alpha (Jensen's alpha)
    alpha_metric = (total_return_strategy - total_return_benchmark * beta) * periods_per_year
    
    # Prepare data structure for dashboard
    dashboard_data = {
        'metadata': {
            'strategy_name': alpha_name,
            'start_date': start_date,
            'end_date': end_date,
            'assets': tickers,
            'interval': interval,
            'price_column': price_column,
            'stop_loss_pct': stop_loss_pct,
            'total_observations': len(strategy_returns_net),
            'generated_at': datetime.now().isoformat()
        },
        
        'time_series': {
            'dates': [d.isoformat() for d in strategy_returns_net.index],
            'strategy_returns': strategy_returns_net.tolist(),
            'benchmark_returns': benchmark_returns_aligned.tolist(),
            'cumulative_strategy': cumulative_strategy.tolist(),
            'cumulative_benchmark': cumulative_benchmark.tolist(),
            'drawdown_strategy': drawdown_strategy.tolist(),
            'drawdown_benchmark': drawdown_benchmark.tolist(),
            'turnover': daily_turnover.reindex(strategy_returns_net.index).fillna(0).tolist(),
            'rolling_sharpe': rolling_sharpe_strategy.dropna().tolist(),
            'rolling_sortino': rolling_sortino_strategy.dropna().tolist(),
            'rolling_info_ratio': rolling_info_ratio_strategy.dropna().tolist(),
            'rolling_dates': [d.isoformat() for d in rolling_sharpe_strategy.dropna().index]
        },
        
        'monthly_data': {
            'dates': [d.isoformat() for d in monthly_returns.index],
            'strategy_returns': (monthly_returns * 100).tolist(),
            'benchmark_returns': (monthly_benchmark * 100).tolist()
        },
        
        'performance_metrics': {
            'strategy': {
                'total_return': float(total_return_strategy * 100),
                'annualized_return': float(perf_metrics.get('annual_return', 0) * 100),
                'volatility': float(volatility_strategy * 100),
                'sharpe_ratio': float(perf_metrics.get('sharpe', 0)),
                'sortino_ratio': float(rolling_sortino_strategy.dropna().iloc[-1] if len(rolling_sortino_strategy.dropna()) > 0 else 0),
                'information_ratio': float(perf_metrics.get('ir', 0)),
                'max_drawdown': float(perf_metrics.get('max_drawdown', 0) * 100),
                'calmar_ratio': float(perf_metrics.get('annual_return', 0) / abs(perf_metrics.get('max_drawdown', 0.01))),
                'win_rate': float(daily_win_rate),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'beta': float(beta),
                'alpha': float(alpha_metric * 100),
                'avg_turnover': float(daily_turnover.mean() * 100)
            },
            'benchmark': {
                'total_return': float(total_return_benchmark * 100),
                'annualized_return': float(((1 + total_return_benchmark) ** (periods_per_year / len(benchmark_returns_aligned)) - 1) * 100),
                'volatility': float(volatility_benchmark * 100),
                'sharpe_ratio': float(benchmark_returns_aligned.mean() / benchmark_returns_aligned.std() * np.sqrt(periods_per_year)),
                'max_drawdown': float(drawdown_benchmark.min())
            }
        },
        
        'risk_metrics': {
            'var_95': float(np.percentile(strategy_returns_net.dropna() * 100, 5)),
            'var_99': float(np.percentile(strategy_returns_net.dropna() * 100, 1)),
            'worst_day': float(strategy_returns_net.min() * 100),
            'best_day': float(strategy_returns_net.max() * 100),
            'avg_daily_return': float(strategy_returns_net.mean() * 100),
            'worst_drawdowns': [
                {
                    'start': dd['start'].isoformat(),
                    'end': dd['end'].isoformat(), 
                    'magnitude': float(dd['magnitude']),
                    'duration': int(dd['duration'])
                }
                for dd in worst_drawdowns
            ]
        },
        
        'trading_stats': {
            'total_positions': int(num_positions),
            'avg_position_size': float(avg_position_size),
            'avg_holding_period': 1,  # Simplified for daily data
            'transaction_costs_bps': 5.0
        }
    }
    
    # Add stop-loss information if available
    if hasattr(portfolio_info, 'attrs') and stop_loss_pct is not None:
        dashboard_data['stop_loss'] = {
            'enabled': True,
            'threshold_pct': stop_loss_pct,
            'triggers': portfolio_info.attrs.get('stop_loss_triggers', 0),
            'stopped_positions': portfolio_info.attrs.get('stopped_positions', [])
        }
    else:
        dashboard_data['stop_loss'] = {'enabled': False}
    
    print("✅ Dashboard data generation complete!")
    print(f"📊 Total return: {total_return_strategy:.2%}")
    print(f"⚡ Sharpe ratio: {perf_metrics.get('sharpe', 0):.2f}")
    print(f"📉 Max drawdown: {perf_metrics.get('max_drawdown', 0):.2%}")
    
    return dashboard_data


def save_dashboard_data(data, filename="dashboard_data.json"):
    """Save dashboard data to JSON file"""
    output_path = Path(filename)
    
    # Clean NaN/inf values before saving
    cleaned_data = clean_nan_values(data)
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"💾 Dashboard data saved to: {output_path.absolute()}")
    return output_path


def main():
    """Main function to generate dashboard data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading strategy dashboard data")
    parser.add_argument('--alpha', type=str, default='alpha999', 
                       help='Alpha strategy to backtest (default: alpha999)')
    parser.add_argument('--start-date', type=str, default=default_start_date,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=default_end_date,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Data interval (1d, 1h, etc.)')
    parser.add_argument('--stop-loss', type=float, default=None,
                       help='Stop-loss percentage (e.g., -5.0)')
    parser.add_argument('--output', type=str, default='dashboard_data.json',
                       help='Output JSON filename')
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                       help='List of tickers to use (default: from main.py)')
    
    args = parser.parse_args()
    
    try:
        # Generate dashboard data
        data = run_backtest_for_dashboard(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            alpha_name=args.alpha,
            interval=args.interval,
            stop_loss_pct=args.stop_loss
        )
        
        
        # Save to file
        save_dashboard_data(data, args.output)
        
        print(f"\n🎉 Success! Use this data file with the HTML dashboard:")
        print(f"   File: {args.output}")
        print(f"   Strategy: {args.alpha}")
        print(f"   Period: {args.start_date} to {args.end_date}")
        
    except Exception as e:
        print(f"❌ Error generating dashboard data: {e}")
        raise


if __name__ == '__main__':
    main() 