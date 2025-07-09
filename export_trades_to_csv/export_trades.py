#!/usr/bin/env python3
"""
Trade Export Script for Alpha Backtesting

This script extracts all individual trades from backtest results and exports them
to spreadsheet format with detailed statistics and analysis.

Usage:
    python export_trades.py [alpha_name] [--all-alphas] [--stop-loss PERCENT]

Examples:
    python export_trades.py alpha998
    python export_trades.py alpha003 --stop-loss -5.0
    python export_trades.py --all-alphas --stop-loss -3.0
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / ".."))

from src.data_loader import get_stock_data, get_crypto_data
from src.alpha101 import Alpha101
from src.trade_export import export_backtest_trades

def main():
    parser = argparse.ArgumentParser(description="Export backtest trades to spreadsheet")
    parser.add_argument("alpha_name", nargs="?", default="alpha998", 
                       help="Alpha to analyze (default: alpha998)")
    parser.add_argument("--all-alphas", action="store_true",
                       help="Export trades for all available alphas")
    parser.add_argument("--output-dir", default="export_trades_to_csv/trade_exports",
                       help="Output directory for trade files (default: export_trades_to_csv/trade_exports)")
    parser.add_argument("--format", choices=["csv", "excel"], default="excel",
                       help="Export format: csv or excel (default: excel)")
    parser.add_argument("--stop-loss", type=float, default=None,
                       help="Individual position stop-loss percentage (e.g., -5.0 for 5%% loss)")
    parser.add_argument("--crypto-mode", action="store_true",
                       help="Use Binance crypto data instead of yfinance stock data")
    parser.add_argument("--interval", type=str, default="1d",
                       help="Data interval for crypto mode: 1m, 5m, 15m, 1h, 4h, 1d (default: 1d)")
    
    args = parser.parse_args()
    
    print("🚀 Loading data and initializing alpha calculator...")
    
    # Load data
    try:
        # Use same config as main.py
        # tickers = ['BTC-USD', 'ETH-USD']
        # start_date = '2024-03-31'
        # end_date = '2025-06-30'
        tickers = ['BTC-USD', 'ETH-USD']
        start_date = '2024-03-31'
        end_date = '2025-06-30'

        # Choose data source based on mode
        if args.crypto_mode:
            print(f"🔥 Crypto mode enabled - using Binance data with {args.interval} interval")
            crypto_tickers = [t for t in tickers if '-USD' in t]
            if not crypto_tickers:
                print("❌ No crypto tickers found. Use format like 'BTC-USD', 'ETH-USD'")
                return 1
            price_data = get_crypto_data(crypto_tickers, start_date=start_date, end_date=end_date, interval=args.interval)
        else:
            price_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
        alpha_calculator = Alpha101(price_data)
        print(f"✅ Data loaded successfully: {len(price_data)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display stop-loss configuration
    if args.stop_loss is not None:
        print(f"🛡️ Individual position stop-loss enabled: {args.stop_loss}%")
    else:
        print("🔓 Stop-loss disabled")
    
    if args.all_alphas:
        print("📊 Exporting trades for all available alphas...")
        
        # Get all alpha methods
        alpha_methods = [method for method in dir(alpha_calculator) 
                        if method.startswith('alpha') and 
                        callable(getattr(alpha_calculator, method))]
        
        successful_exports = 0
        failed_exports = []
        
        for alpha_name in sorted(alpha_methods):
            try:
                print(f"\n🔍 Processing {alpha_name}...")
                export_path = export_backtest_trades(
                    alpha_calculator, 
                    price_data, 
                    alpha_name, 
                    args.output_dir,
                    args.format,
                    args.stop_loss
                )
                if export_path:
                    successful_exports += 1
                    print(f"✅ {alpha_name} exported successfully")
                else:
                    failed_exports.append(alpha_name)
                    print(f"⚠️ {alpha_name} had no trades to export")
            except Exception as e:
                failed_exports.append(alpha_name)
                print(f"❌ Failed to export {alpha_name}: {e}")
        
        print(f"\n📈 Export Summary:")
        print(f"   Successfully exported: {successful_exports} alphas")
        print(f"   Failed or empty: {len(failed_exports)} alphas")
        if failed_exports:
            print(f"   Failed alphas: {', '.join(failed_exports[:10])}")
    
    else:
        # Export single alpha
        print(f"📊 Exporting trades for {args.alpha_name}...")
        
        try:
            export_path = export_backtest_trades(
                alpha_calculator, 
                price_data, 
                args.alpha_name, 
                args.output_dir,
                args.format,
                args.stop_loss
            )
            
            if export_path:
                print(f"✅ Trade export completed successfully!")
                print(f"📁 Files saved to: {export_path.parent}")
                
                # List generated files
                files = list(export_path.parent.glob(f"{args.alpha_name}_trades_*"))
                print(f"📄 Generated files:")
                for file in files:
                    print(f"   - {file.name}")
            else:
                print(f"⚠️ No trades found for {args.alpha_name}")
                return 1
                
        except Exception as e:
            print(f"❌ Failed to export trades: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n🎉 Trade export completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 