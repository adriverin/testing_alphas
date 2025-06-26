#
# validation.py - In-Sample / Out-of-Sample Workflow
#

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf

from src.get_stock_data import get_stock_data
from src.alpha101 import Alpha101
from src.analysis import generate_full_report, generate_interval_report, generate_summary_html_report, analyze_performance
from src.backtests import run_rank_backtest
from src.utils import generate_date_intervals



def run_oos_validation_report(alpha_calc, full_price_data, alpha_list, intervals, report_dir="oos_validation"):
    """
    Runs and plots OOS validation for a given list of alphas.
    """
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    for alpha_name in alpha_list:
        print(f"\nValidating {alpha_name}...")
        pdf_path = os.path.join(report_dir, f"{alpha_name}_OOS_report.pdf")
        
        with backend_pdf.PdfPages(pdf_path) as pdf:
            try:
                # Calculate on full history to have signals ready for OOS start
                full_alpha_series = getattr(alpha_calc, alpha_name)().dropna()
                
                # Ensure the full series is sorted before we start slicing it
                if not full_alpha_series.index.is_monotonic_increasing:
                    full_alpha_series = full_alpha_series.sort_index()

                for start_str, end_str in intervals:
                    # Filter the calculated signals and data to the OOS interval
                    oos_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_str:end_str, :]]
                    
                    # Also filter the price data for this specific interval
                    oos_price_data_interval = full_price_data.loc[pd.IndexSlice[start_str:end_str, :]]
                    
                    # --- FIX 2: Sort the sliced data right before using it ---
                    if not oos_price_data_interval.index.is_monotonic_increasing:
                        oos_price_data_interval = oos_price_data_interval.sort_index()
                    # --- END OF FIX ---

                    if oos_alpha_series.empty or oos_price_data_interval.empty:
                        print("  -> No valid signals in OOS period.")
                        continue
                    
                    strategy_returns, portfolio_info = run_rank_backtest(oos_price_data_interval, oos_alpha_series)
                    
                    if strategy_returns.empty: continue
                        
                    fig = plt.figure(figsize=(11.69, 8.27))
                    # This function call can now "see" analyze_performance
                    analyze_performance(
                        strategy_returns, 
                        portfolio_info, 
                        oos_price_data_interval, 
                        fig=fig, 
                        title=f"{alpha_name} Out-of-Sample Performance"
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
                print(f"  -> OOS Report generated: {pdf_path}")
            except Exception as e:
                print(f"  -> FAILED to validate {alpha_name}: {e}")
                # This ensures the pdf is still created, even if empty, to avoid another error
                # The MatplotlibDeprecationWarning is harmless and can be ignored.
                plt.close('all')


# ---------------------------------------------------------------------
# --- MAIN CONFIGURATION ---
# ---------------------------------------------------------------------

if __name__ == '__main__':
    
    # --- 1. Define the complete dataset parameters ---
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
        'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT'
    ]
    full_start_date = '2011-01-01'
    full_end_date = '2025-05-31'
    
    # --- 2. Define the In-Sample / Out-of-Sample Split ---
    # We will use data up to the end of 2020 for discovery (In-Sample)
    # and data from 2021 onwards for validation (Out-of-Sample).
    in_sample_end_date = '2020-12-31'
    out_of_sample_start_date = '2021-01-01'
    
    print("--- Loading Full Dataset ---")
    full_price_data = get_stock_data(sp100_tickers, start_date=full_start_date, end_date=full_end_date)

    if full_price_data.empty:
        print("Could not load data. Exiting.")
    else:
        # ---------------------------------------------------------------------
        # --- PHASE 1: DISCOVERY (using In-Sample data ONLY) ---
        # ---------------------------------------------------------------------
        print("\n" + "="*50)
        print("PHASE 1: DISCOVERY - Analyzing performance on In-Sample data")
        print(f"In-Sample Period: {full_start_date} to {in_sample_end_date}")
        print("="*50)

        # Filter the data for the In-Sample period
        in_sample_data = full_price_data.loc[pd.IndexSlice[:in_sample_end_date, :]]
        
        # Initialize a calculator with ONLY the in-sample data
        is_alpha_calculator = Alpha101(in_sample_data)
        
        # Generate the summary report to find the best alphas
        is_intervals = generate_date_intervals(full_start_date, in_sample_end_date, n=4)
        generate_summary_html_report(
            is_alpha_calculator, 
            in_sample_data, 
            is_intervals,
            report_dir="in_sample_analysis"
        )
        print("\nSUCCESS: In-Sample HTML summary report has been generated in the 'in_sample_analysis' folder.")
        print("Please review the report and choose your champion alphas for the next phase.")
        
        # ---------------------------------------------------------------------
        # --- PHASE 2: VALIDATION (using Out-of-Sample data ONLY) ---
        # ---------------------------------------------------------------------
        
        # This is the list you will populate MANUALLY after reviewing the In-Sample report.
        # I've pre-filled it with a few examples that often perform well.
        champion_alphas_to_validate = [
            # 'alpha005', 
            # 'alpha019', 
            # 'alpha039',
            # 'alpha042',
            # 'alpha046',
            # 'alpha047',
            # 'alpha052',
            # 'alpha102',
            # 'alpha103',
            # 'alpha104',
            'alpha105'
        ]
        
        print("\n" + "="*50)
        print("PHASE 2: VALIDATION - Testing champion alphas on Out-of-Sample data")
        print(f"Out-of-Sample Period: {out_of_sample_start_date} to {full_end_date}")
        print(f"Selected Alphas for Validation: {champion_alphas_to_validate}")
        print("="*50)
        
        # Filter the data for the Out-of-Sample period
        oos_price_data = full_price_data.loc[pd.IndexSlice[out_of_sample_start_date:, :]]
        
        # We need a calculator with the full data history to avoid lookback period errors
        full_alpha_calculator = Alpha101(full_price_data)
        
        # Use the detailed interval report function to generate a report for the OOS period
        # We'll just use one interval: the entire OOS period.
        oos_intervals = [(out_of_sample_start_date, full_end_date)]
        
        # Run the validation
        run_oos_validation_report(full_alpha_calculator, oos_price_data, champion_alphas_to_validate, oos_intervals)
