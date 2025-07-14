#
# main.py - The Central Command Script for the Alpha Factory
#
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np # Added for percentile testing

# --- Import all the functions from your modules in the 'src' directory ---
from src.data_loader import get_stock_data, get_crypto_data
from src.alpha101 import Alpha101
from src.backtests import run_rank_backtest
from src.combiner import combine_alphas
from src.reporting import generate_interval_report, generate_summary_html_report, generate_date_intervals, analyze_performance
from src.validation import run_factor_analysis, run_oos_validation_report, run_is_validation_report



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

# tickers = ['BTC-USD']
# tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD', 'SHIB-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD', 'AVAX-USD']
tickers = ['BTC-USD']

start_date = '2025-01-01'  
end_date = '2025-07-10'    


# --- Define the intervals you want to test ---
number_of_intervals = 1

# --- Define the first and last alpha to test ---
first_alpha = 999  # Use alpha999 for ML-based signals (recommended)
last_alpha = 999   # Same as first_alpha for single alpha mode





def main(tickers=tickers, start_date=start_date, end_date=end_date, number_of_intervals=number_of_intervals, first_alpha=first_alpha, last_alpha=last_alpha):
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
        type=float, 
        default=None,
        help='Individual position stop-loss percentage (e.g., -5.0 for 5%% loss). Applies to interval, summary, oos, and combine analyses.'
    )
    parser.add_argument(
        '--crypto-mode',
        action='store_true',
        help='Use Binance crypto data instead of yfinance stock data'
    )
    parser.add_argument(
        '--interval',
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
    
    args = parser.parse_args()


    intervals_to_test = generate_date_intervals(start_date, end_date, number_of_intervals)
    # print(f"Intervals to Test: {intervals_to_test}")

    # --- Load Data Once ---
    print("--- Loading Full Dataset ---")
    
    # Choose data source based on mode
    if args.crypto_mode:
        print(f"🔥 Crypto mode enabled - using Binance data with {args.interval} interval")
        # Filter to crypto tickers only
        crypto_tickers = [t for t in tickers if '-USD' in t]
        if not crypto_tickers:
            print("❌ No crypto tickers found. Use format like 'BTC-USD', 'ETH-USD'")
            return
        price_data = get_crypto_data(crypto_tickers, start_date=start_date, end_date=end_date, interval=args.interval)
    else:
        print("📈 Stock mode - using yfinance data")
        price_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
    
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
        generate_interval_report(alpha_calculator, price_data, intervals_to_test, first_alpha=first_alpha, last_alpha=last_alpha, stop_loss_pct=args.stop_loss)

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