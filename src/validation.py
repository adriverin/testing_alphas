import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
import pandas_datareader.data as web 
import statsmodels.api as sm

from src.reporting import analyze_performance, generate_date_intervals
from src.backtests import run_rank_backtest



def run_oos_validation_report(alpha_calc, full_price_data, alpha_list, intervals, report_dir="reports/oos_validation"):
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
                    
                    if not oos_price_data_interval.index.is_monotonic_increasing:
                        oos_price_data_interval = oos_price_data_interval.sort_index()

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

def run_is_validation_report(alpha_calculator, price_data, alpha_list, report_dir='reports/in_sample_analysis'):
    """
    Runs and plots In-Sample validation for a given list of alphas.
    """
    print(f"\n--- Generating In-Sample Analysis Report ---")
    
    for alpha_name in alpha_list:
        print(f"\nProcessing {alpha_name}...")
        pdf_path = os.path.join(report_dir, f"{alpha_name}_in_sample_analysis.pdf")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        with backend_pdf.PdfPages(pdf_path) as pdf:
                if hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name)):
                    print(f"\nProcessing {alpha_name}...")
                    try:
                        alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                        
                        if alpha_series.empty:
                            print(f"  -> Skipping {alpha_name}, no valid signals.")
                            continue

                        strategy_returns, portfolio_info = run_rank_backtest(price_data, alpha_series)
                        
                        fig = plt.figure(figsize=(11.69, 8.27))                    

                        analyze_performance(strategy_returns, portfolio_info, price_data, fig=fig, title=alpha_name)
                        
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    except Exception as e:
                        # (Error handling remains the same)
                        print(f"  -> FAILED to process {alpha_name}: {e}")
                        fig_err, ax_err = plt.subplots(figsize=(11.69, 8.27))
                        ax_err.text(0.5, 0.5, f'Failed to process {alpha_name}\nError: {e}', ha='center', va='center', color='red')
                        pdf.savefig(fig_err)
                        plt.close(fig_err)

    print("\n--- Full Alpha Report Generated at ---")
    print(f"{pdf_path}")

def run_is_validation_report(alpha_calculator, full_price_data, alpha_list, interval_start_date, interval_end_date, report_dir='reports/in_sample_analysis'):
    """
    Runs and plots In-Sample validation for a given list of alphas.
    """
    print(f"\n--- Generating In-Sample Analysis Report ---")
    
    for alpha_name in alpha_list:
        print(f"\nProcessing {alpha_name}...")
        pdf_path = os.path.join(report_dir, f"{alpha_name}_in_sample_analysis.pdf")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        if not (hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name))):
            continue

        with backend_pdf.PdfPages(pdf_path) as pdf:
            try:
                full_alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                if not full_alpha_series.index.is_monotonic_increasing:
                    full_alpha_series = full_alpha_series.sort_index()
            except Exception as e:
                print(f"  -> FAILED to calculate alpha series for {alpha_name}: {e}")
                continue

            if full_alpha_series.empty:
                print(f"  -> Skipping {alpha_name}, no valid signals.")
                continue


            # Loop through each date interval
            try:
                # Filter both price data and the pre-calculated alpha series for the interval
                interval_price_data = full_price_data.loc[pd.IndexSlice[interval_start_date:interval_end_date, :]]
                interval_alpha_series = full_alpha_series.loc[pd.IndexSlice[interval_start_date:interval_end_date, :]]
                
                if interval_alpha_series.empty or interval_price_data.empty:
                    print(f"    -> No data in this interval for {alpha_name}. Skipping.")
                    continue
                    
                # Backtest on the interval data
                strategy_returns, portfolio_info = run_rank_backtest(interval_price_data, interval_alpha_series)
                
                if strategy_returns.empty:
                    print(f"    -> Backtest resulted in no returns for this interval. Skipping.")
                    continue
                
                fig = plt.figure(figsize=(11.69, 8.27))
                analyze_performance(strategy_returns, portfolio_info, interval_price_data, fig=fig, 
                                    title=f"{alpha_name}\n{interval_start_date} to {interval_end_date}")
                pdf.savefig(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"    -> FAILED to process interval: {e}")
                fig_err, ax_err = plt.subplots(figsize=(11.69, 8.27))
                ax_err.text(0.5, 0.5, f'Failed to process interval\n{interval_start_date} to {interval_end_date}\nError: {e}', 
                            ha='center', va='center', color='red')
                pdf.savefig(fig_err)
                plt.close(fig_err)

    print("\n--- In-Sample Analysis Report Generated at ---")
    print(f"{pdf_path}")





# --- Helper Function to Get Factor Data ---

def get_fama_french_factors(start_date, end_date):
    """
    Downloads Fama-French 3-factor daily data using pandas-datareader.
    Handles potential errors and ensures correct date formatting.
    """
    print("\n--- Downloading Fama-French 3-Factor Data ---")
    try:
        # Fetch the daily data from Ken French's library
        ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
        
        # The library returns a dictionary; the daily data is in the first element (index 0)
        ff_df = ff_data[0]
        
        # Convert values from percentages to decimals
        ff_df = ff_df / 100
        
        # Convert the index to a standard DatetimeIndex if it's not already
        if not isinstance(ff_df.index, pd.DatetimeIndex):
            ff_df.index = ff_df.index.to_timestamp()
            
        print("Fama-French data downloaded successfully.")
        return ff_df
    except Exception as e:
        print(f"Could not download Fama-French data. Error: {e}")
        print("Please ensure you have the 'lxml' package installed (`pip install lxml`). It is required by pandas-datareader.")
        return None
    



# --- Main Factor Analysis Function ---

def run_factor_analysis(strategy_returns_net, start_date, end_date):
    """
    Performs a Fama-French 3-factor regression on a given strategy's returns.

    Args:
        strategy_returns_net (pd.Series): A Series of daily net returns for the strategy.
                                          The index must be a DatetimeIndex.
        start_date (str): The start date for the analysis period.
        end_date (str): The end date for the analysis period.
    """
    print("\n--- Running Fama-French 3-Factor Regression ---")

    # 1. Get the Fama-French factor data for the period
    ff_data = get_fama_french_factors(start_date, end_date)
    
    if ff_data is None:
        print("Halting factor analysis due to data download failure.")
        return

    # 2. Prepare the data for regression
    # Rename the strategy returns Series for clarity in the merge
    strategy_returns_net.name = 'strategy_net_returns'
    
    # Merge the strategy returns with the factor data on the date index
    df_merged = pd.merge(strategy_returns_net, ff_data, left_index=True, right_index=True, how='inner')
    
    # Calculate the strategy's excess return (return above the risk-free rate)
    df_merged['strategy_excess_ret'] = df_merged['strategy_net_returns'] - df_merged['RF']
    
    if df_merged.empty:
        print("No overlapping data between strategy returns and factor data. Cannot run regression.")
        return

    # 3. Run the Ordinary Least Squares (OLS) regression
    
    # Define the independent variables (the factors)
    X = df_merged[['Mkt-RF', 'SMB', 'HML']]
    # Define the dependent variable (our strategy's excess return)
    y = df_merged['strategy_excess_ret']
    
    # Add a constant (the intercept) to the independent variables. This is crucial.
    X = sm.add_constant(X)
    
    # Fit the regression model
    model = sm.OLS(y, X).fit()
    
    # 4. Print and interpret the results
    print("\n" + "="*80)
    print("                        FACTOR ANALYSIS RESULTS")
    print("="*80)
    print(model.summary())
    print("="*80)

    # Extract and explain the key metrics
    # The intercept ('const') from the regression is the daily alpha
    daily_alpha = model.params['const']
    # Annualize it by multiplying by 252 trading days
    alpha_annualized = daily_alpha * 252 * 100
    alpha_p_value = model.pvalues['const']
    
    print("\n--- INTERPRETATION GUIDE ---")
    print(f"Annualized Alpha (Intercept): {alpha_annualized:.2f}% per year.")
    print(f"   -> This is the portion of your strategy's return NOT explained by the market, size, or value factors.")
    print(f"Alpha P-value: {alpha_p_value:.4f}")
    if alpha_p_value < 0.05:
        print("   -> INTERPRETATION: The alpha is statistically significant. This is a strong positive result, suggesting a genuine edge.")
    else:
        print("   -> INTERPRETATION: The alpha is NOT statistically significant. The performance could be due to random chance.")
    
    print(f"\nMarket Beta (Mkt-RF Coefficient): {model.params['Mkt-RF']:.3f}")
    print(f"   -> A value close to 0 indicates the strategy is well-hedged against market movements.")

    print(f"\nSize Beta (SMB Coefficient): {model.params['SMB']:.3f}")
    print(f"   -> Positive suggests a tilt towards small-cap stocks. Negative suggests a tilt towards large-caps.")
    
    print(f"\nValue Beta (HML Coefficient): {model.params['HML']:.3f}")
    print(f"   -> Positive suggests a tilt towards value stocks. Negative suggests a tilt towards growth stocks.")

    print(f"\nR-squared: {model.rsquared:.3f}")
    print(f"   -> This means {model.rsquared:.1%} of your strategy's daily returns are explained by these 3 common factors.")
    print(f"   -> A lower R-squared is generally better, as it indicates a more unique and diversified source of returns.")