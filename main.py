#
# main.py - The Central Command Script for the Alpha Factory
#
import argparse
import os
import matplotlib.pyplot as plt

# --- Import all the functions from your modules in the 'src' directory ---
from src.data_loader import get_stock_data
from src.alpha101 import Alpha101
from src.backtests import run_rank_backtest
from src.combiner import combine_alphas
from src.reporting import generate_interval_report, generate_summary_html_report, generate_date_intervals, analyze_performance
from src.validation import run_factor_analysis, run_oos_validation_report, run_is_validation_report




# ---------------------------------------------------------------------
# --- Central Configuration ---
# ---------------------------------------------------------------------
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
    'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT'
]
start_date = '2015-01-01'
end_date = '2025-01-01' 

# --- Define the intervals you want to test ---
number_of_intervals = 5

# --- Define the first and last alpha to test ---
first_alpha = 1
last_alpha = 105





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
    
    args = parser.parse_args()


    intervals_to_test = generate_date_intervals(start_date, end_date, number_of_intervals)


    # --- Load Data Once ---
    print("--- Loading Full Dataset ---")
    price_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
    
    if price_data.empty:
        print("Could not load data. Exiting.")
        return

    print("\n--- Initializing Alpha Calculator ---")
    alpha_calculator = Alpha101(price_data)

    # --- Execute Chosen Analysis ---
    
    if args.analysis_type == 'interval':
        print("\n--- Running Per-Alpha Interval PDF Report ---")
        generate_interval_report(alpha_calculator, price_data, intervals_to_test, first_alpha=first_alpha, last_alpha=last_alpha)

    elif args.analysis_type == 'summary':
        print("\n--- Running Summary HTML Report ---")
        generate_summary_html_report(alpha_calculator, price_data, intervals_to_test, first_alpha=first_alpha, last_alpha=last_alpha)

    elif args.analysis_type == 'oos':
        print("\n--- Running In-Sample / Out-of-Sample Validation ---")

        # Define your split date here
        core_alphas = ['alpha003', 'alpha041', 'alpha042', 'alpha054', 'alpha083', 'alpha101']
        in_sample_end_date = '2020-12-31'
        intervals_to_test = generate_date_intervals(start_date, in_sample_end_date, number_of_intervals)

        run_oos_validation_report(alpha_calculator, price_data, core_alphas, intervals_to_test)
        run_is_validation_report(alpha_calculator, price_data, core_alphas, in_sample_end_date, end_date)


    elif args.analysis_type == 'combine' or args.analysis_type == 'factor':
        # Both 'combine' and 'factor' analyses need the combined alpha returns
        print("\n--- Generating and Backtesting Combined Alpha ---")
        
        # This is your basket of "champion" alphas, selected from your research
        core_alphas = ['alpha003', 'alpha041', 'alpha042', 'alpha054', 'alpha083', 'alpha101']
        
        mega_alpha_signal = combine_alphas(alpha_calculator, core_alphas)
        
        if mega_alpha_signal.empty:
            print("Combined alpha resulted in no signals. Halting.")
            return
            
        strategy_returns_gross, portfolio_info = run_rank_backtest(price_data, mega_alpha_signal)
        daily_turnover = portfolio_info['turnover'].groupby('date').first()
        daily_cost = daily_turnover * (5 / 10000.0) # 5 bps
        strategy_returns_net = strategy_returns_gross - daily_cost.reindex(strategy_returns_gross.index).fillna(0)

        if args.analysis_type == 'combine':
            fig = plt.figure(figsize=(12, 8))
            analyze_performance(
                strategy_returns_gross, # Pass gross returns to see both net and gross curves
                portfolio_info, 
                price_data, 
                fig=fig, 
                title=f"Performance of Combined Alphas ({len(core_alphas)} signals)"
            )
            report_dir = "final_strategy_reports"
            if not os.path.exists(report_dir): os.makedirs(report_dir)
            plot_path = os.path.join(report_dir, "combined_alpha_performance.pdf")
            fig.savefig(plot_path)
            print(f"\n--- Final Combined Strategy Report saved to '{plot_path}' ---")
            plt.show()

        elif args.analysis_type == 'factor':
            run_factor_analysis(strategy_returns_net, start_date, end_date)


if __name__ == '__main__':
    main()