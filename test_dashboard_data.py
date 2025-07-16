#!/usr/bin/env python3
"""
Test script to validate dashboard data structure and generate test data
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math

def generate_test_data():
    """Generate test data for dashboard validation"""
    
    # Create sample data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    days = (end_date - start_date).days
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate realistic returns
    np.random.seed(42)  # For reproducible results
    strategy_returns = np.random.normal(0.0005, 0.015, days)  # Mean 0.05%, std 1.5%
    benchmark_returns = np.random.normal(0.0003, 0.012, days)  # Mean 0.03%, std 1.2%
    
    # Calculate cumulative returns
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_benchmark = np.cumprod(1 + benchmark_returns)
    
    # Calculate drawdown
    peak = np.maximum.accumulate(cumulative_strategy)
    drawdown = (cumulative_strategy / peak - 1) * 100
    
    # Calculate rolling metrics (252-day window)
    window = min(252, len(strategy_returns) // 4)
    rolling_sharpe = []
    rolling_sortino = []
    rolling_info_ratio = []
    rolling_dates = []
    
    for i in range(window - 1, len(strategy_returns)):
        window_returns = strategy_returns[i - window + 1:i + 1]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        
        # Sharpe ratio (annualized)
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0.0
        rolling_sharpe.append(float(sharpe) if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0)
        
        # Sortino ratio (simplified)
        downside_returns = window_returns[window_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        rolling_sortino.append(float(sortino) if not np.isnan(sortino) and not np.isinf(sortino) else 0.0)
        
        # Information ratio (vs benchmark)
        bench_window = benchmark_returns[i - window + 1:i + 1]
        excess_returns = window_returns - bench_window
        tracking_error = np.std(excess_returns)
        info_ratio = (np.mean(excess_returns) * np.sqrt(252)) / tracking_error if tracking_error > 0 else 0.0
        rolling_info_ratio.append(float(info_ratio) if not np.isnan(info_ratio) and not np.isinf(info_ratio) else 0.0)
        
        rolling_dates.append(dates[i])
    
    # Generate monthly data
    monthly_data = {}
    for i, date in enumerate(dates):
        year = date.year
        month = date.month
        key = f"{year}-{month:02d}"
        
        if key not in monthly_data:
            monthly_data[key] = {
                'strategy_returns': [],
                'benchmark_returns': [],
                'date': datetime(year, month, 1)
            }
        
        monthly_data[key]['strategy_returns'].append(strategy_returns[i])
        monthly_data[key]['benchmark_returns'].append(benchmark_returns[i])
    
    # Calculate monthly returns
    monthly_dates = []
    monthly_strategy_returns = []
    monthly_benchmark_returns = []
    
    for key, data in sorted(monthly_data.items()):
        monthly_dates.append(data['date'])
        
        # Calculate compound monthly return
        strat_monthly = np.prod(1 + np.array(data['strategy_returns'])) - 1
        bench_monthly = np.prod(1 + np.array(data['benchmark_returns'])) - 1
        
        monthly_strategy_returns.append(strat_monthly * 100)
        monthly_benchmark_returns.append(bench_monthly * 100)
    
    # Create the dashboard data structure
    dashboard_data = {
        "metadata": {
            "strategy_name": "Test Strategy Alpha999",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "assets": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "interval": "1d",
            "total_observations": days,
            "price_column": "close"
        },
        "time_series": {
            "dates": [d.isoformat() for d in dates],
            "strategy_returns": strategy_returns.tolist(),
            "benchmark_returns": benchmark_returns.tolist(),
            "cumulative_strategy": cumulative_strategy.tolist(),
            "cumulative_benchmark": cumulative_benchmark.tolist(),
            "drawdown_strategy": drawdown.tolist(),
            "turnover": np.random.uniform(0.01, 0.1, days).tolist(),
            "rolling_sharpe": rolling_sharpe,
            "rolling_sortino": rolling_sortino,
            "rolling_info_ratio": rolling_info_ratio,
            "rolling_dates": [d.isoformat() for d in rolling_dates]
        },
        "performance_metrics": {
            "strategy": {
                "total_return": (cumulative_strategy[-1] - 1) * 100,
                "annualized_return": ((cumulative_strategy[-1] ** (252/days)) - 1) * 100,
                "volatility": np.std(strategy_returns) * np.sqrt(252) * 100,
                "sharpe_ratio": np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0.0,
                "sortino_ratio": np.mean(strategy_returns) / np.std(strategy_returns[strategy_returns < 0]) * np.sqrt(252) if len(strategy_returns[strategy_returns < 0]) > 0 and np.std(strategy_returns[strategy_returns < 0]) > 0 else 0.0,
                "information_ratio": np.mean(strategy_returns - benchmark_returns) / np.std(strategy_returns - benchmark_returns) * np.sqrt(252) if np.std(strategy_returns - benchmark_returns) > 0 else 0.0,
                "max_drawdown": np.min(drawdown),
                "calmar_ratio": (((cumulative_strategy[-1] ** (252/days)) - 1) * 100) / abs(np.min(drawdown)) if abs(np.min(drawdown)) > 0 else 0.0,
                "win_rate": (strategy_returns > 0).sum() / len(strategy_returns) * 100,
                "skewness": float(pd.Series(strategy_returns).skew()),
                "kurtosis": float(pd.Series(strategy_returns).kurtosis()),
                "beta": np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0.0,
                "alpha": (np.mean(strategy_returns) - np.mean(benchmark_returns)) * 252 * 100,
                "avg_turnover": np.mean(np.random.uniform(0.01, 0.1, days)) * 100
            },
            "benchmark": {
                "total_return": (cumulative_benchmark[-1] - 1) * 100,
                "annualized_return": ((cumulative_benchmark[-1] ** (252/days)) - 1) * 100,
                "volatility": np.std(benchmark_returns) * np.sqrt(252) * 100,
                "sharpe_ratio": np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252) if np.std(benchmark_returns) > 0 else 0.0,
                "max_drawdown": np.min((np.cumprod(1 + benchmark_returns) / np.maximum.accumulate(np.cumprod(1 + benchmark_returns)) - 1) * 100)
            }
        },
        "monthly_data": {
            "dates": [d.isoformat() for d in monthly_dates],
            "strategy_returns": monthly_strategy_returns,
            "benchmark_returns": monthly_benchmark_returns
        },
        "risk_metrics": {
            "worst_drawdowns": [
                {
                    "start": "2020-03-15T00:00:00.000Z",
                    "end": "2020-04-01T00:00:00.000Z",
                    "magnitude": -15.2,
                    "duration": 17
                },
                {
                    "start": "2021-09-10T00:00:00.000Z",
                    "end": "2021-10-05T00:00:00.000Z",
                    "magnitude": -8.7,
                    "duration": 25
                }
            ]
        },
        "stop_loss": {
            "enabled": False
        }
    }
    
    return dashboard_data

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

def validate_data_structure(data):
    """Validate that the data structure matches what the dashboard expects"""
    
    print("🔍 Validating data structure...")
    
    errors = []
    
    # Check top-level structure
    required_keys = ['metadata', 'time_series', 'performance_metrics', 'monthly_data', 'risk_metrics', 'stop_loss']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")
    
    # Check metadata
    if 'metadata' in data:
        metadata_keys = ['strategy_name', 'start_date', 'end_date', 'assets', 'interval']
        for key in metadata_keys:
            if key not in data['metadata']:
                errors.append(f"Missing metadata key: {key}")
    
    # Check time_series
    if 'time_series' in data:
        ts_keys = ['dates', 'strategy_returns', 'benchmark_returns', 'cumulative_strategy', 'cumulative_benchmark']
        for key in ts_keys:
            if key not in data['time_series']:
                errors.append(f"Missing time_series key: {key}")
            elif not isinstance(data['time_series'][key], list):
                errors.append(f"time_series.{key} is not a list")
    
    # Check data consistency
    if 'time_series' in data and all(k in data['time_series'] for k in ['dates', 'strategy_returns']):
        dates_len = len(data['time_series']['dates'])
        returns_len = len(data['time_series']['strategy_returns'])
        if dates_len != returns_len:
            errors.append(f"Date/returns length mismatch: {dates_len} dates vs {returns_len} returns")
    
    if errors:
        print("❌ Validation errors found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Data structure validation passed!")
        return True

def main():
    """Main test function"""
    print("🧪 Testing dashboard data generation...")
    
    # Generate test data
    test_data = generate_test_data()
    
    # Validate structure
    is_valid = validate_data_structure(test_data)
    
    if is_valid:
        # Clean data to remove NaN/inf values before saving
        cleaned_data = clean_nan_values(test_data)
        
        # Save test data
        with open('dashboard_data.json', 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        print("✅ Test data generated and saved to dashboard_data.json")
        print(f"📊 Data summary:")
        print(f"   - Strategy: {test_data['metadata']['strategy_name']}")
        print(f"   - Period: {test_data['metadata']['start_date']} to {test_data['metadata']['end_date']}")
        print(f"   - Observations: {len(test_data['time_series']['dates'])}")
        print(f"   - Assets: {', '.join(test_data['metadata']['assets'])}")
        print(f"   - Total Return: {test_data['performance_metrics']['strategy']['total_return']:.2f}%")
        print(f"   - Sharpe Ratio: {test_data['performance_metrics']['strategy']['sharpe_ratio']:.3f}")
    else:
        print("❌ Test failed due to validation errors")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 