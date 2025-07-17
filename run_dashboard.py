#!/usr/bin/env python3

"""
Trading Strategy Dashboard Runner
=================================

Interactive script to generate and view trading strategy dashboards.
Now optimized to work with cached signals, avoiding probability file requirements.
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

# Import current configuration from main.py
try:
    from main import tickers as current_tickers, start_date as default_start_date, end_date as default_end_date
    MAIN_CONFIG_AVAILABLE = True
except ImportError:
    current_tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'LTC-USD']  # Fallback
    default_start_date = '2025-01-01'
    default_end_date = '2025-07-15'
    MAIN_CONFIG_AVAILABLE = False

def check_cached_signals():
    """Check what cached signal files are available and their date ranges."""
    print("📁 Checking available cached signal files...\n")
    
    if MAIN_CONFIG_AVAILABLE:
        print(f"🎯 Current main.py configuration: {current_tickers}")
        print(f"📅 Default date range: {default_start_date} to {default_end_date}\n")
    else:
        print(f"⚠️  Could not import main.py config, using fallback: {current_tickers}\n")
    
    artifacts_dir = Path("artefacts")
    signal_files = [
        artifacts_dir / "multi_asset" / "multi_crypto_signals_improved.parquet",
        artifacts_dir / "improved_ml" / "improved_trading_signals.parquet",
    ]
    
    # Add individual asset signals
    signals_dir = artifacts_dir / "signals"
    if signals_dir.exists():
        signal_files.extend(list(signals_dir.glob("*_improved_signals.parquet")))
    
    available_ranges = {}
    relevant_signals = {}
    other_signals = {}
    
    for signal_file in signal_files:
        if signal_file.exists():
            try:
                df = pd.read_parquet(signal_file)
                start_date = df.index.min()
                end_date = df.index.max()
                
                # Format the display
                file_display = signal_file.name
                if "multi_crypto_signals" in file_display:
                    assets = list(df.columns) if hasattr(df, 'columns') else ['Unknown']
                    asset_info = f" (Assets: {', '.join(assets)})"
                    # Check if this multi-asset file contains current tickers
                    relevant_assets = [a for a in assets if a in current_tickers]
                    is_relevant = len(relevant_assets) > 0
                else:
                    # Individual asset file - extract asset name
                    asset_name = file_display.split('_')[0]
                    # Handle both formats: BTC-USD and BTCUSD
                    if not asset_name.endswith('-USD'):
                        asset_name = asset_name + '-USD'
                    assets = [asset_name]
                    asset_info = f" ({asset_name})"
                    is_relevant = asset_name in current_tickers
                
                signal_info = {
                    'start': start_date,
                    'end': end_date,
                    'assets': asset_info,
                    'shape': df.shape,
                    'file_path': signal_file
                }
                
                available_ranges[file_display] = signal_info
                
                if is_relevant:
                    relevant_signals[file_display] = signal_info
                else:
                    other_signals[file_display] = signal_info
                
            except Exception as e:
                print(f"❌ Error reading {signal_file.name}: {e}")
    
    if not available_ranges:
        print("❌ No cached signal files found!")
        print("💡 Run: python multi_crypto_ml_training.py")
        return None
    
    # Display relevant signals first
    if relevant_signals:
        print("✅ Relevant cached signals (for current main.py tickers):")
        for file_display, info in relevant_signals.items():
            print(f"   📊 {file_display}{info['assets']}")
            print(f"      📅 Date range: {info['start'].strftime('%Y-%m-%d')} to {info['end'].strftime('%Y-%m-%d')}")
            print(f"      📊 Shape: {info['shape']}")
        print()
    
    # Display other signals
    if other_signals:
        print("📂 Other cached signals (from previous training runs):")
        for file_display, info in other_signals.items():
            print(f"   📄 {file_display}{info['assets']}")
            print(f"      📅 Date range: {info['start'].strftime('%Y-%m-%d')} to {info['end'].strftime('%Y-%m-%d')}")
            print(f"      📊 Shape: {info['shape']}")
        print()
        print("💡 These signals are from previous training runs and won't be used with current tickers.")
        print("   To use them, update the tickers list in main.py or specify --tickers in generate_dashboard_data.py\n")
    
    # Find the best date range for Alpha999
    multi_asset_file = "multi_crypto_signals_improved.parquet"
    if multi_asset_file in relevant_signals:
        best_range = relevant_signals[multi_asset_file]
        print(f"🎯 Recommended for Alpha999: {best_range['start'].strftime('%Y-%m-%d')} to {best_range['end'].strftime('%Y-%m-%d')}")
        return best_range
    elif multi_asset_file in available_ranges:
        best_range = available_ranges[multi_asset_file]
        print(f"🎯 Multi-asset signals available: {best_range['start'].strftime('%Y-%m-%d')} to {best_range['end'].strftime('%Y-%m-%d')}")
        print(f"⚠️  But may not match current tickers: {current_tickers}")
        return best_range
    
    return list(relevant_signals.values())[0] if relevant_signals else list(available_ranges.values())[0]

def get_strategy_choice():
    """Get strategy choice from user."""
    print("📊 Available Trading Strategies:")
    print("1. Alpha999 (ML-based) - Uses cached signals")
    print("2. Alpha003 (Traditional factor)")
    print("3. Alpha041 (Traditional factor)")
    print("4. Alpha042 (Traditional factor)")
    print("5. Custom alpha (enter name)")
    
    while True:
        choice = input("\nSelect strategy (1-5): ").strip()
        
        if choice == "1":
            return "alpha999"
        elif choice == "2":
            return "alpha003"
        elif choice == "3":
            return "alpha041"
        elif choice == "4":
            return "alpha042"
        elif choice == "5":
            custom = input("Enter alpha name (e.g., alpha999_dynamic): ").strip()
            return custom if custom else "alpha999"
        else:
            print("Please enter 1-5")


from datetime import timedelta, datetime

def get_date_range(recommended_range=None):
    """Get date range from user."""
    if recommended_range:
        # Calculate yesterday's date and the next day after the end of the recommended range
        yesterday = datetime.now() - timedelta(days=1)
        next_day = recommended_range['end'] + timedelta(days=1)

        print(f"\n📅 Recommended date range (based on cached signals):")
        print(f"   Start: {next_day.strftime('%Y-%m-%d')}")
        print(f"   End: {yesterday.strftime('%Y-%m-%d')}")
        print("1. Use recommended range")
        print("2. Use main.py defaults")
        print("3. Enter custom range")
        
        choice = input("Select option (1-3): ").strip()
        if choice == "1":
            return (
                next_day.strftime('%Y-%m-%d'),
                yesterday.strftime('%Y-%m-%d')
            )
        elif choice == "2":
            print(f"📅 Using main.py defaults: {default_start_date} to {default_end_date}")
            return default_start_date, default_end_date
    
    print("\n📅 Enter custom date range:")
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    
    return start_date, end_date



def main():
    """Main interactive dashboard runner."""
    print("=" * 60)
    print("🚀 Trading Strategy Dashboard Runner")
    print("=" * 60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading strategy dashboard data")
    parser.add_argument('--interval', type=str, default='1d',
                       help='Data interval (1d, 1h, etc.)')    
    
    args = parser.parse_args()
    
    # Check available cached signals
    recommended_range = check_cached_signals()
    
    # Get strategy choice
    strategy = get_strategy_choice()
    
    # Get date range
    start_date, end_date = get_date_range(recommended_range)
    
    # Additional options
    print(f"\n🎯 Configuration:")
    print(f"   Strategy: {strategy}")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Generate dashboard data
    print("\n📊 Generating dashboard data...")
    
    cmd = [
        sys.executable, "generate_dashboard_data.py",
        "--interval", args.interval,
        "--alpha", strategy,
        "--start-date", start_date,
        "--end-date", end_date,
        "--output", "dashboard_data.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dashboard data generated successfully!")
        
        # Show key metrics from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'Total return:' in line or 'Sharpe ratio:' in line or 'Max drawdown:' in line:
                print(f"   {line.strip()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating dashboard data:")
        print(e.stderr)
        return
    
    # Start dashboard server
    print("\n🌐 Starting dashboard server...")
    
    try:
        # Check if server is already running
        import requests
        response = requests.get("http://localhost:8000", timeout=2)
        print("✅ Dashboard server already running!")
    except:
        print("🚀 Starting new dashboard server...")
        subprocess.Popen([sys.executable, "serve_dashboard.py"])
        
        # Wait a moment for server to start
        import time
        time.sleep(2)
    
    print("\n🎉 Dashboard ready!")
    print("🌐 Open in your browser: http://localhost:8000/trading_strategy_dashboard.html")
    print("\n💡 Tips:")
    print("   - Alpha999 uses cached ML signals (no probability files needed)")
    print("   - Use date ranges within cached signal coverage for best performance")
    print("   - Traditional alphas (003, 041, 042) work with any date range")

if __name__ == "__main__":
    main() 