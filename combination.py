import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Import all the functions from your existing modules ---
from src.data_loader import get_stock_data
from src.alpha101 import Alpha101
from src.backtests import run_rank_backtest
from src.analysis import analyze_performance
from src.combiner import combine_alphas # <-- Import our new function

# ---------------------------------------------------------------------
# --- MAIN CONFIGURATION ---
# ---------------------------------------------------------------------
if __name__ == '__main__':
    
    # --- 1. Define dataset parameters ---
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
        'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT'
    ]
    start_date = '2020-01-01'
    end_date = '2025-05-31'
    
    # --- 2. Define your basket of robust "Core Alphas" ---
    # This list should be populated based on the results of your
    # In-Sample / Out-of-Sample validation (Priority 1).
    # These are just examples.
    core_alphas = [
        'alpha024',
        'alpha042',
        'alpha052',
        'alpha104'
    ]
    
    print("--- Loading Full Dataset ---")
    price_data = get_stock_data(sp100_tickers, start_date=start_date, end_date=end_date)

    if price_data.empty:
        print("Could not load data. Exiting.")
    else:
        # --- 3. Initialize Calculator and Combine Alphas ---
        print("\nInitializing Alpha Calculator...")
        alpha_calculator = Alpha101(price_data)
        
        # Generate the single "mega-alpha" signal
        mega_alpha_signal = combine_alphas(alpha_calculator, core_alphas, method='mean')
        
        if mega_alpha_signal.empty:
            print("Combination resulted in no valid signals. Exiting.")
        else:
            # --- 4. Backtest and Analyze the Combined Strategy ---
            print("\n--- Backtesting Combined Alpha Signal ---")
            
            # Run the backtest on the combined signal
            strategy_returns, portfolio_info = run_rank_backtest(price_data, mega_alpha_signal)
            
            # Generate the final performance plot
            fig = plt.figure(figsize=(12, 8))
            analyze_performance(
                strategy_returns, 
                portfolio_info, 
                price_data, 
                fig=fig, 
                title=f"Performance of Combined Alphas ({len(core_alphas)} signals)"
            )
            
            # Save the final plot
            report_dir = "reports/final_strategy_reports"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            plot_path = os.path.join(report_dir, "combined_alpha_performance.pdf")
            fig.savefig(plot_path)
            print(f"\n--- Final Combined Strategy Report saved to '{plot_path}' ---")
            
            # Optionally, show the plot immediately
            plt.show()