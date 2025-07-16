#
# main.py - The Central Command Script for the Alpha Factory
#
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np # Added for percentile testing

def get_crypto_data_unified(tickers, start_date, end_date, interval='1d', price_column='close'):
    """
    Unified crypto data loading that uses the same OHLCV cache as ML system.
    
    Args:
        tickers: List of crypto symbols (e.g., ['BTC-USD', 'ETH-USD'])
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
        price_column: Price column to use for returns calculation ('open', 'high', 'low', 'close', 'vwap')
    
    Returns:
        pd.DataFrame: Same format as get_stock_data with MultiIndex (date, asset)
                      Columns: 'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns'
    """
    import pandas as pd
    from pathlib import Path
    from src.ml_forecasting import MLConfig
    from src.ml_forecasting.data_loader import _fetch_from_binance_ohlcv
    
    print(f"🔗 Using unified OHLCV caching system")
    print(f"📅 Loading {len(tickers)} assets: {', '.join(tickers)}")
    print(f"📊 Date range: {start_date} to {end_date} ({interval} interval)")
    print(f"💰 Returns calculated from '{price_column}' price column")
    
    all_data = []
    cache_dir = Path("artefacts/data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    start_date_dt = pd.to_datetime(start_date, utc=True)
    end_date_dt = pd.to_datetime(end_date, utc=True)
    
    for ticker in tickers:
        print(f"\n📈 Loading {ticker}...")
        
        try:
            # Create cache file path for OHLCV data (same as ML system)
            cache_file = cache_dir / f"ohlcv_{ticker.replace('/', '').replace('-', '')}_{interval}.parquet"
            
            # Check if we can use cached OHLCV data
            df = None
            should_download = False
            
            if cache_file.exists():
                try:
                    print(f"📂 Checking OHLCV cache for {ticker}...")
                    cached_df = pd.read_parquet(cache_file)
                    
                    if not cached_df.empty:
                        cache_start = cached_df.index.min()
                        cache_end = cached_df.index.max()
                        
                        print(f"📊 Cached OHLCV range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
                        
                        # Smart cache validation (same logic as ML system)
                        if start_date_dt >= cache_start and end_date_dt <= cache_end:
                            print(f"✅ OHLCV cache fully covers requested range for {ticker}")
                            # Filter to requested range  
                            mask = (cached_df.index >= start_date_dt) & (cached_df.index <= end_date_dt)
                            df = cached_df[mask].copy()
                        else:
                            # Check if the gap is significant (more than 7 days)
                            start_gap = max(0, (cache_start - start_date_dt).days) if start_date_dt < cache_start else 0
                            end_gap = max(0, (end_date_dt - cache_end).days) if end_date_dt > cache_end else 0
                            
                            if start_gap > 7 or end_gap > 7:
                                print(f"🔄 Requested range extends significantly beyond OHLCV cache for {ticker}")
                                if start_gap > 7:
                                    print(f"   • Start date {start_date} is {start_gap} days before cached start")
                                if end_gap > 7:
                                    print(f"   • End date {end_date} is {end_gap} days after cached end")
                                should_download = True
                            else:
                                print(f"✅ OHLCV cache covers most of requested range for {ticker} (small gaps)")
                                # Use cached data even with small gaps
                                mask = (cached_df.index >= max(start_date_dt, cache_start)) & (cached_df.index <= min(end_date_dt, cache_end))
                                df = cached_df[mask].copy()
                    else:
                        print(f"🔄 OHLCV cache file is empty for {ticker}")
                        should_download = True
                        
                except Exception as e:
                    print(f"🔄 Error reading OHLCV cache for {ticker}: {e}")
                    should_download = True
            else:
                print(f"📂 No OHLCV cache file found for {ticker}")
                should_download = True
            
            # Download if needed
            if should_download or df is None or df.empty:
                print(f"🌐 Downloading fresh OHLCV data for {ticker}...")
                
                try:
                    config = MLConfig(
                        symbol=ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        cache_dir="artefacts"
                    )
                    
                    df_ohlcv = _fetch_from_binance_ohlcv(config)
                    
                    if not df_ohlcv.empty:
                        # Save to OHLCV cache
                        df_ohlcv.to_parquet(cache_file)
                        print(f"💾 Cached OHLCV data for {ticker}: {len(df_ohlcv)} bars")
                        
                        # Filter to requested range
                        mask = (df_ohlcv.index >= start_date_dt) & (df_ohlcv.index <= end_date_dt)
                        df = df_ohlcv[mask].copy()
                    else:
                        print(f"⚠️  No OHLCV data available for {ticker}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Failed to download OHLCV data for {ticker}: {e}")
                    continue
            
            if df is None or df.empty:
                print(f"⚠️  No data available for {ticker}")
                continue
            
            # Convert to main.py format (MultiIndex with date, asset)
            df_formatted = df.copy()
            df_formatted['asset'] = ticker
            df_formatted = df_formatted.reset_index().set_index(['timestamp', 'asset'])
            df_formatted.index.names = ['date', 'asset']
            
            # Ensure we have all required columns
            if 'volume' not in df_formatted.columns:
                df_formatted['volume'] = 1000000  # Default volume
            
            all_data.append(df_formatted)
            print(f"✅ {ticker}: {len(df_formatted)} bars loaded")
            
        except Exception as e:
            print(f"❌ Failed to load {ticker}: {e}")
            continue
    
    if not all_data:
        print("❌ No crypto data was successfully loaded")
        return pd.DataFrame()
    
    # Combine all assets
    print(f"\n🔗 Combining data from {len(all_data)} assets...")
    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()
    
    # Ensure volume is numeric
    if 'volume' in final_df.columns:
        final_df['volume'] = pd.to_numeric(final_df['volume'], errors='coerce').fillna(0).astype(int)
    
    print("🔧 Processing crypto data for alpha calculations...")
    
    # Add calculated columns (same as original get_crypto_data)
    print("📊 Adding calculated columns (vwap, returns)...")
    final_df['vwap'] = (final_df['close'] + final_df['open'] + final_df['high'] + final_df['low']) / 4
    
    # Calculate returns from the specified price column
    if price_column == 'vwap':
        final_df['returns'] = final_df.groupby(level='asset')['vwap'].pct_change()
    else:
        # Validate price column exists
        if price_column not in final_df.columns:
            print(f"⚠️  Warning: Price column '{price_column}' not found, falling back to 'close'")
            price_column = 'close'
        final_df['returns'] = final_df.groupby(level='asset')[price_column].pct_change()
    
    # Add crypto-specific metadata
    print("🏷️ Adding crypto metadata...")
    final_df['sector'] = 'Cryptocurrency'
    final_df['cap'] = 0
    
    final_df = final_df.dropna()
    
    final_start = final_df.index.get_level_values('date').min()
    final_end = final_df.index.get_level_values('date').max()
    
    print(f"✅ Unified crypto data loading complete!")
    print(f"📅 Final data range: {final_start.strftime('%Y-%m-%d')} to {final_end.strftime('%Y-%m-%d')}")
    print(f"📊 Shape: {final_df.shape}")
    print(f"💰 Returns calculated from '{price_column}' price column")
    print(f"💾 OHLCV data cached in: artefacts/data/ohlcv_*_{interval}.parquet")
    print(f"💾 Same cache used by ML training system for efficiency")
    
    return final_df


# --- Import all the functions from your modules in the 'src' directory ---
from src.data_loader import get_stock_data, get_crypto_data
from src.alpha101 import Alpha101
from src.backtests import run_rank_backtest, run_rank_dollar_neutral_backtest, run_alpha_value_backtest, run_alpha_value_dollar_neutral_backtest
from src.combiner import combine_alphas
from src.reporting import generate_interval_report, generate_summary_html_report, generate_date_intervals, analyze_performance
from src.validation import run_factor_analysis, run_oos_validation_report, run_is_validation_report



def detect_ml_price_column(tickers):
    """
    Detect the price column used for ML model training by reading model metadata.
    
    Args:
        tickers: List of crypto symbols (e.g., ['BTC-USD', 'ETH-USD'])
    
    Returns:
        str: Price column name ('close', 'open', 'high', 'low', 'vwap') or 'close' as default
    """
    from pathlib import Path
    import torch
    import json
    
    print("🔍 Detecting ML model price column configuration...")
    
    artifacts_dir = Path("artefacts")
    models_dir = artifacts_dir / "models"
    
    # Check if we have any ML models
    if not models_dir.exists():
        print(f"📂 No models directory found at {models_dir}")
        return 'close'
    
    # Try to find model metadata for any of the provided tickers
    price_column = None
    
    for ticker in tickers:
        # Try different model naming patterns
        potential_paths = [
            models_dir / f"{ticker}_improved_model.pt",
            models_dir / f"{ticker}_simple_model.pt",
            models_dir / f"{ticker}_improved_metadata.json",
            models_dir / f"{ticker}_simple_metadata.json",
        ]
        
        for model_path in potential_paths:
            if model_path.exists():
                print(f"📁 Found model file: {model_path}")
                
                try:
                    if model_path.suffix == '.pt':
                        # Load PyTorch model metadata
                        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                        if 'config' in model_data:
                            config = model_data['config']
                            if 'price_column' in config:
                                price_column = config['price_column']
                                print(f"✅ Found price_column in {ticker} model: '{price_column}'")
                                break
                    
                    elif model_path.suffix == '.json':
                        # Load JSON metadata
                        with open(model_path, 'r') as f:
                            metadata = json.load(f)
                            if 'config' in metadata and 'price_column' in metadata['config']:
                                price_column = metadata['config']['price_column']
                                print(f"✅ Found price_column in {ticker} metadata: '{price_column}'")
                                break
                            
                except Exception as e:
                    print(f"⚠️  Error reading {model_path}: {e}")
                    continue
        
        # Break if we found a price column
        if price_column:
            break
    
    # Final result
    if price_column:
        print(f"🎯 ML models trained using '{price_column}' price column")
        print(f"🔄 Backtesting will use same '{price_column}' column for consistency")
        return price_column
    else:
        print(f"⚠️  No ML model metadata found, using default 'close' price column")
        print(f"💡 Train models first with: python multi_crypto_ml_training.py")
        return 'close'
    




# ---------------------------------------------------------------------
# --- Central Configuration ---
# ---------------------------------------------------------------------
# tickers = [
#     'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
#     'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT'
# ]
# sp100_tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD', 'SHIB-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD', 'AVAX-USD', 'PEPE24478-USD']
# tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD']

# sp100_tickers = [
# "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
# "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","C",
# "CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS",
# "CVX","DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD",
# "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU",
# "JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
# "MDT","MET","META","MMM","MO","MRK","MS","MSFT","NFLX",
# "NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PLTR","PM",
# "QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS",
# "TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"
# ]

# tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
# tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD', 'SHIB-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD', 'AVAX-USD']
# tickers = ['BTC-USD']
tickers = ['DOGE-USD', 'PEPE-USD', 'SHIB-USD', 'FLOKI-USD']

start_date = '2025-01-01'  
end_date = '2025-07-14'    

backtest_func = run_rank_backtest
# backtest_func = run_rank_dollar_neutral_backtest
# backtest_func = run_alpha_value_dollar_neutral_backtest
# backtest_func = run_alpha_value_backtest


# --- Define the intervals you want to test ---
number_of_intervals = 1

# --- Define the first and last alpha to test ---
first_alpha = 999  # Use alpha999 for ML-based signals (recommended)
last_alpha = 999  # Same as first_alpha for single alpha mode





def main(tickers=tickers, start_date=start_date, end_date=end_date, number_of_intervals=number_of_intervals, first_alpha=first_alpha, last_alpha=last_alpha, backtest_func=backtest_func):
    """
    Main function to orchestrate the alpha research workflow.
    """
    # --- Setup Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Alpha Research and Backtesting Factory.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    parser.add_argument(
        'analysis_type', 
        choices=['interval', 'summary', 'oos', 'factor', 'combine'], 
        help="""The type of analysis to run:
    - interval:  Generate a detailed PDF report for each alpha, showing performance in different time intervals.
    - summary:   Generate a single, interactive HTML heatmap of all alphas' performance (Information Ratio) across intervals.
    - oos:       Run a formal In-Sample discovery and Out-of-Sample validation workflow.
    - factor:    Run a Fama-French 3-factor regression analysis on the combined 'mega-alpha'.
    - combine:   Run a full backtest on the combined 'mega-alpha' and show the performance plot.
    """
    )
    
    parser.add_argument(
        '--stop-loss',
        '-sl', 
        type=float, 
        default=None,
        help='Individual position stop-loss percentage (e.g., -5.0 for 5%% loss). Applies to interval, summary, oos, and combine analyses.'
    )
    parser.add_argument(
        '--stock-mode',
        action='store_true',
        help='Use yfinance stock data instead of default Binance crypto data'
    )
    parser.add_argument(
        '--interval',
        '-i',
        type=str,
        default='1d',
        help='Data interval for crypto mode: 1m, 5m, 15m, 1h, 4h, 1d (default: 1d)'
    )
    parser.add_argument(
        '--ml-percentiles',
        type=str,
        default='5,95',
        help='ML signal percentiles as "bottom,top" (e.g., "1,99" for conservative, "20,80" for aggressive)'
    )
    parser.add_argument(
        '--test-percentiles',
        action='store_true',
        help='Test multiple percentile configurations automatically'
    )
    parser.add_argument(
        '--price-column',
        type=str,
        default=None,
        help='Override price column for returns calculation (open, high, low, close, vwap). If not specified, detects from ML model metadata.'
    )
    
    args = parser.parse_args()


    intervals_to_test = generate_date_intervals(start_date, end_date, number_of_intervals)
    # print(f"Intervals to Test: {intervals_to_test}")

    # --- Load Data Once ---
    print("--- Loading Full Dataset ---")
    
    # Determine price column to use for returns calculation
    if args.price_column:
        # User override
        final_price_column = args.price_column
        print(f"🎯 Using user-specified price column: '{final_price_column}'")
    else:
        # Auto-detect from ML models
        relevant_tickers = tickers if args.stock_mode else [t for t in tickers if '-USD' in t]
        final_price_column = detect_ml_price_column(relevant_tickers)
    
    # Choose data source based on mode
    if args.stock_mode:
        print("📈 Stock mode - using yfinance data")
        price_data = get_stock_data(tickers, start_date=start_date, end_date=end_date, price_column=final_price_column)        
    else:
        print(f"🔥 Crypto mode (default) - using Binance data with {args.interval} interval")
        # Filter to crypto tickers only
        crypto_tickers = [t for t in tickers if '-USD' in t]
        if not crypto_tickers:
            print("❌ No crypto tickers found. Use format like 'BTC-USD', 'ETH-USD'")
            return
        
        price_data = get_crypto_data_unified(crypto_tickers, start_date, end_date, args.interval, final_price_column)        
    
    if price_data.empty:
        print("Could not load data. Exiting.")
        return

    print("\n--- Initializing Alpha Calculator ---")
    alpha_calculator = Alpha101(price_data)

    # --- Execute Chosen Analysis ---
    
    if args.analysis_type == 'interval':
        print("\n--- Running Per-Alpha Interval PDF Report ---")
        if args.stop_loss is not None:
            print(f"🛡️ Individual position stop-loss enabled: {args.stop_loss}%")
        generate_interval_report(alpha_calculator, price_data, intervals_to_test, first_alpha=first_alpha, last_alpha=last_alpha, stop_loss_pct=args.stop_loss, backtest_func=backtest_func)

    elif args.analysis_type == 'summary':
        print("\n--- Running Summary HTML Report ---")
        if args.stop_loss is not None:
            print(f"🛡️ Individual position stop-loss enabled: {args.stop_loss}%")
        generate_summary_html_report(alpha_calculator, price_data, intervals_to_test, first_alpha=first_alpha, last_alpha=last_alpha, stop_loss_pct=args.stop_loss)

    elif args.analysis_type == 'oos':
        print("\n--- Running In-Sample / Out-of-Sample Validation ---")
        if args.stop_loss is not None:
            print(f"🛡️ Individual position stop-loss enabled: {args.stop_loss}%")

        # Define your split date here
        core_alphas = ['alpha003', 'alpha041', 'alpha042', 'alpha054', 'alpha083', 'alpha101']
        in_sample_end_date = '2020-12-31'
        intervals_to_test = generate_date_intervals(start_date, in_sample_end_date, number_of_intervals)

        run_oos_validation_report(alpha_calculator, price_data, core_alphas, intervals_to_test, stop_loss_pct=args.stop_loss)
        run_is_validation_report(alpha_calculator, price_data, core_alphas, in_sample_end_date, end_date, stop_loss_pct=args.stop_loss)


    elif args.analysis_type == 'combine' or args.analysis_type == 'factor':
        # Both 'combine' and 'factor' analyses need the combined alpha returns
        print("\n--- Generating and Backtesting Combined Alpha ---")
        
        # Display stop-loss configuration
        if args.stop_loss is not None:
            print(f"🛡️ Individual position stop-loss enabled: {args.stop_loss}%")
        
        # This is your basket of "champion" alphas, selected from your research
        # if first_alpha == last_alpha:
        #     # Single alpha mode
        #     core_alphas = [f'alpha{first_alpha:03d}']
        # else:
        #     # Multiple alphas mode - use range
        #     core_alphas = [f'alpha{i:03d}' for i in range(first_alpha, last_alpha + 1)]
        
        # Parse ML percentiles
        try:
            percentile_str = args.ml_percentiles.split(',')
            ml_percentiles = (int(percentile_str[0]), int(percentile_str[1]))
            print(f"🎯 Using ML percentiles: {ml_percentiles}")
        except:
            ml_percentiles = (5, 95)
            print(f"⚠️  Invalid percentiles format, using default: {ml_percentiles}")
        
        # Test multiple percentiles if requested
        if args.test_percentiles:
            print("\n🧪 Testing multiple percentile configurations...")
            percentile_configs = [
                (1, 99),   # Very conservative
                (3, 97),   # Conservative  
                (5, 95),   # Default
                (10, 90),  # Moderate
                (20, 80),  # Aggressive
            ]
            
            for test_percentiles in percentile_configs:
                print(f"\n📊 Testing percentiles {test_percentiles}...")
                
                # Generate signals with current percentiles
                mega_alpha_signal = alpha_calculator.alpha999_dynamic(percentiles=test_percentiles)
                
                if mega_alpha_signal.empty or (mega_alpha_signal == 0).all():
                    print(f"❌ No signals generated for percentiles {test_percentiles}")
                    continue
                
                # Run backtest
                strategy_returns_gross, portfolio_info = run_rank_backtest(price_data, mega_alpha_signal, args.stop_loss)
                daily_turnover = portfolio_info['turnover'].groupby('date').first()
                daily_cost = daily_turnover * (5 / 10000.0) # 5 bps
                strategy_returns_net = strategy_returns_gross - daily_cost.reindex(strategy_returns_gross.index).fillna(0)
                
                # Quick performance summary
                total_return = (1 + strategy_returns_net).prod() - 1
                volatility = strategy_returns_net.std() * np.sqrt(252)
                sharpe = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252) if strategy_returns_net.std() > 0 else 0
                
                print(f"   📈 Total Return: {total_return:.2%}")
                print(f"   📊 Volatility: {volatility:.2%}")
                print(f"   ⚡ Sharpe Ratio: {sharpe:.2f}")
                
            print(f"\n🏁 Percentile testing complete. Choose the best configuration and run with --ml-percentiles")
            return
        
        # Use dynamic alpha999 with specified percentiles
        core_alphas = ['alpha999_dynamic']
        
        # Generate signals with dynamic percentiles
        mega_alpha_signal = alpha_calculator.alpha999_dynamic(percentiles=ml_percentiles)
        mega_alpha_signal.name = f'alpha999_dynamic_{ml_percentiles[0]}_{ml_percentiles[1]}'
        
        if mega_alpha_signal.empty:
            print("Alpha resulted in no signals. Halting.")
            return
            
        strategy_returns_gross, portfolio_info = run_rank_backtest(price_data, mega_alpha_signal, args.stop_loss)
        daily_turnover = portfolio_info['turnover'].groupby('date').first()
        daily_cost = daily_turnover * (5 / 10000.0) # 5 bps
        strategy_returns_net = strategy_returns_gross - daily_cost.reindex(strategy_returns_gross.index).fillna(0)

        if args.analysis_type == 'combine':
            fig = plt.figure(figsize=(12, 8))
            
            # Modify title to include stop-loss info
            title = f"Performance of Combined Alphas ({len(core_alphas)} signals)"
            if args.stop_loss is not None:
                title += f" with {args.stop_loss}% Stop-Loss"
            
            analyze_performance(
                strategy_returns_gross, 
                portfolio_info, 
                price_data, 
                fig=fig, 
                title=title
            )
            report_dir = "final_strategy_reports"
            if not os.path.exists(report_dir): os.makedirs(report_dir)
            plot_path = os.path.join(report_dir, "combined_alpha_performance.pdf")
            fig.savefig(plot_path)
            
            # Print stop-loss summary if applicable
            if hasattr(portfolio_info, 'attrs') and args.stop_loss is not None:
                stop_loss_triggers = portfolio_info.attrs.get('stop_loss_triggers', 0)
                print(f"\n🛡️ Stop-Loss Summary: {stop_loss_triggers} positions stopped out")
            
            print(f"\n--- Final Combined Strategy Report saved to '{plot_path}' ---")
            plt.show()

        elif args.analysis_type == 'factor':
            run_factor_analysis(strategy_returns_net, start_date, end_date)


if __name__ == '__main__':
    main()
    # for i in range(200, 210):
    #     os.system(f"open reports/interval_reports/alpha{i}_interval_report.pdf")

