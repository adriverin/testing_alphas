#!/usr/bin/env python3
"""
Trade Export Script for Alpha Backtesting

This script extracts all individual trades from backtest results and exports them
to spreadsheet format with detailed statistics and analysis.

Usage:
    python export_trades.py [alpha_name] [--all-alphas]

Examples:
    python export_trades.py alpha998
    python export_trades.py alpha003
    python export_trades.py --all-alphas
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / ".."))

from src.data_loader import get_stock_data
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
    
    args = parser.parse_args()
    
    print("🚀 Loading data and initializing alpha calculator...")
    
    # Load data
    try:
        # Use same config as main.py
        tickers = ['BTC-USD', 'ETH-USD']
        start_date = '2024-03-31'
        end_date = '2025-06-30'

        price_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
        alpha_calculator = Alpha101(price_data)
        print(f"✅ Data loaded successfully: {len(price_data)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
                    args.format
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
                args.format
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