import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from src.backtests import run_backtest, run_rank_backtest
from src.alpha101 import Alpha101
import os


def analyze_performance(returns_series, portfolio_info, price_data, fig, title='Strategy Performance', transaction_cost_bps=5):
    """
    Calculates performance metrics and creates a 2-panel plot on a given figure.
    - Uses a "floating" benchmark aligned with the alpha's active period.
    - Adds a clear text annotation specifying the active date range.
    - Adds a "Net of Fees" curve to the equity plot.
    - Displays daily portfolio turnover in the second panel.
    
    Args:
        returns_series (pd.Series): The daily returns of the strategy.
        portfolio_info (pd.DataFrame): DataFrame with weights and turnover.
        price_data (pd.DataFrame): The full price data for the relevant period, used for the benchmark.
        fig (matplotlib.figure.Figure): The matplotlib figure object to plot on.
        title (str): The title for the overall figure.
        transaction_cost_bps (int): The assumed one-way transaction cost in basis points (1 bps = 0.01%).
        
    Returns:
        dict: A dictionary of performance metrics based on net returns.
    """
    fig.suptitle(title, fontsize=16)
    
    # --- Top Panel: Equity Curve ---
    ax1 = fig.add_subplot(2, 1, 1)
    
    # If there are no returns, do not proceed with plotting
    if returns_series.empty:
        ax1.text(0.5, 0.5, 'No returns to analyze for this period.', ha='center', va='center')
        return {}
        
    # --- 1. Define the active date range based on the strategy returns ---
    active_start_date = returns_series.index.min()
    active_end_date = returns_series.index.max()
    
    # --- 2. Align the Benchmark to this active date range ---
    benchmark_returns = price_data['returns'].groupby(level='date').mean()
    benchmark_returns_aligned = benchmark_returns.loc[active_start_date:active_end_date]
    cumulative_benchmark_returns = (1 + benchmark_returns_aligned).cumprod()
    ax1.plot(cumulative_benchmark_returns.index, cumulative_benchmark_returns.values, label='Buy & Hold Benchmark', color='gray', linestyle='--')

    # 3. Gross Strategy Performance (already aligned)
    cumulative_returns_gross = (1 + returns_series).cumprod()
    ax1.plot(cumulative_returns_gross.index, cumulative_returns_gross.values, label='Alpha Strategy (Gross)', color='blue', linewidth=2)

    # 4. Net Strategy Performance (already aligned)
    daily_turnover = portfolio_info['turnover'].groupby('date').first()
    daily_cost = daily_turnover * (transaction_cost_bps / 10000.0)
    returns_series_net = returns_series - daily_cost
    cumulative_returns_net = (1 + returns_series_net).cumprod()
    ax1.plot(cumulative_returns_net.index, cumulative_returns_net.values, label=f'Alpha Strategy (Net, {transaction_cost_bps} bps)', color='green', linewidth=2)
    
    # --- Calculate Comparative Metrics (IR) ---
    excess_returns = returns_series_net - benchmark_returns_aligned
    tracking_error = np.std(excess_returns) * np.sqrt(252)
    # Avoid division by zero if tracking error is zero
    if tracking_error == 0:
        information_ratio = np.nan
    else:
        annualized_excess_return = np.mean(excess_returns) * 252
        information_ratio = annualized_excess_return / tracking_error

    # --- Calculate Standard Performance Metrics (on Net Returns) ---
    std_dev_net = np.std(returns_series_net); std_dev_net = 1e-6 if std_dev_net == 0 else std_dev_net
    sharpe_ratio_net = np.mean(returns_series_net) / std_dev_net * np.sqrt(252)
    cumulative_returns_net = (1 + returns_series_net).cumprod()
    annualized_return_net = (cumulative_returns_net.iloc[-1]) ** (252 / len(cumulative_returns_net)) - 1
    peak_net = cumulative_returns_net.expanding(min_periods=1).max()
    drawdown_net = (cumulative_returns_net / peak_net) - 1
    max_drawdown_net = drawdown_net.min()
    
    # --- Add the Date Range Annotation ---
    date_range_text = f"Active Period: {active_start_date.strftime('%Y-%m-%d')} to {active_end_date.strftime('%Y-%m-%d')}"
    stats_text = (f"Net Sharpe: {sharpe_ratio_net:.2f}\n"
                  f"Information Ratio: {information_ratio:.2f}\n"
                  f"Net Ann. Return: {annualized_return_net:.2%}\n"
                  f"Net Max Drawdown: {max_drawdown_net:.2%}\n")
                #   f"{date_range_text}") 
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend(loc='upper center')
    ax1.grid(True)
    
    # --- Bottom Panel: Daily Turnover ---
    # (This part remains the same)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    turnover_series = portfolio_info['turnover'].groupby('date').first().dropna()
    ax2.plot(turnover_series.index, turnover_series.values, label='Daily Turnover', color='orange', alpha=0.6, linewidth=1)
    rolling_turnover = turnover_series.rolling(window=22).mean()
    ax2.plot(rolling_turnover.index, rolling_turnover.values, label='1-Month Avg. Turnover', color='darkred', linestyle='-')
    avg_turnover = turnover_series.mean()
    ax2.axhline(avg_turnover, color='black', linestyle=':', label=f'Avg Turnover: {avg_turnover:.2%}')
    
    ax2.set_ylabel('Portfolio Turnover')
    ax2.set_xlabel('Date')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return {
        'sharpe': sharpe_ratio_net,
        'ir': information_ratio, 
        'annual_return': annualized_return_net,
        'max_drawdown': max_drawdown_net
    }

def generate_full_report(alpha_calculator, price_data, pdf_path='reports/alpha_report.pdf', first_alpha=1, last_alpha=106):
    """
    Calculates all implemented alphas and backtests them.
    - Adds a Buy & Hold benchmark to each plot.
    """
    print(f"\n--- Generating Full Alpha Report to '{pdf_path}' ---")
    
    with backend_pdf.PdfPages(pdf_path) as pdf:
        for i in range(first_alpha, last_alpha):
            alpha_name = f'alpha{i:03d}'
            if hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name)):
                print(f"\nProcessing {alpha_name}...")
                try:
                    alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                    
                    if alpha_series.empty:
                        print(f"  -> Skipping {alpha_name}, no valid signals.")
                        continue

                    # strategy_returns, portfolio_info = run_backtest(price_data, alpha_series, quantiles=5)
                    strategy_returns, portfolio_info = run_rank_backtest(price_data, alpha_series)
                    
                    # last_day_df = portfolio_info.loc[portfolio_info.index.get_level_values('date').max()]
                    # print(last_day_df)

                    # long_positions = last_day_df[last_day_df['weights'] > 0].index.tolist()
                    # short_positions = last_day_df[last_day_df['weights'] < 0].index.tolist()
                    # print(f"  -> Portfolio Snapshot for Last Day:")
                    # print(f"     Longs: {long_positions}")
                    # print(f"     Shorts: {short_positions}")
                    
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

    print("\n--- Full Alpha Report Generation Complete ---")






def generate_interval_report(alpha_calculator, full_price_data, date_intervals, report_dir="reports/interval_reports", first_alpha=1, last_alpha=106):
    """
    Performs a chunked backtest for each alpha over specified date intervals.
    Generates one PDF report per alpha, with each page showing performance in one interval.
    """
    print(f"\n--- Starting Interval-Based Analysis ---")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    # Loop through each alpha in the calculator
    for i in range(first_alpha, last_alpha):
        alpha_name = f'alpha{i:03d}'
        if not (hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name))):
            continue
            
        print(f"\n--- Analyzing {alpha_name} ---")
        pdf_path = os.path.join(report_dir, f"{alpha_name}_interval_report.pdf")
        
        with backend_pdf.PdfPages(pdf_path) as pdf:
            # Calculate the full alpha series once to save time
            try:
                full_alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                # Ensure the calculated alpha series is also sorted before slicing.
                if not full_alpha_series.index.is_monotonic_increasing:
                    full_alpha_series = full_alpha_series.sort_index()
                                    
            except Exception as e:
                print(f"  -> FAILED to calculate alpha series for {alpha_name}: {e}")
                continue

            if full_alpha_series.empty:
                print(f"  -> Skipping {alpha_name}, no valid signals.")
                continue

            # Loop through each date interval
            for start_str, end_str in date_intervals:
                interval_title = f"Interval: {start_str} to {end_str}"
                print(f"  Processing {interval_title}")
                
                start_dt = pd.to_datetime(start_str)
                end_dt = pd.to_datetime(end_str)
                
                try:
                    # Filter both price data and the pre-calculated alpha series for the interval
                    interval_price_data = full_price_data.loc[pd.IndexSlice[start_dt:end_dt, :]]
                    interval_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_dt:end_dt, :]]
                    
                    if interval_alpha_series.empty or interval_price_data.empty:
                        print(f"    -> No data in this interval for {alpha_name}. Skipping.")
                        continue
                        
                    # Backtest on the interval data
                    strategy_returns, portfolio_info = run_rank_backtest(interval_price_data, interval_alpha_series)
                    
                    if strategy_returns.empty:
                        print(f"    -> Backtest resulted in no returns for this interval. Skipping.")
                        continue
                    
                    # Create and save the plot for this interval
                    fig = plt.figure(figsize=(11.69, 8.27))
                    analyze_performance(strategy_returns, portfolio_info, interval_price_data, fig=fig, 
                                        title=f"{alpha_name}\n{interval_title}")
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"    -> FAILED to process interval: {e}")
                    fig_err, ax_err = plt.subplots(figsize=(11.69, 8.27))
                    ax_err.text(0.5, 0.5, f'Failed to process interval\n{interval_title}\nError: {e}', 
                                ha='center', va='center', color='red')
                    pdf.savefig(fig_err)
                    plt.close(fig_err)

    print("\n--- Interval-Based Analysis Complete ---")







def generate_summary_html_report(alpha_calculator, full_price_data, date_intervals, report_dir="reports/summary_reports", first_alpha=1, last_alpha=106):
    """
    Performs a chunked backtest and generates a single, interactive HTML summary report.
    - The main metric displayed and color-coded is the Information Ratio (IR).
    - Tooltips show Sharpe Ratio, Return, Max Drawdown, and Turnover.
    """
    print(f"\n--- Starting HTML Summary Report Generation (centered on Information Ratio) ---")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    all_metrics_data = []
    interval_names = [f"{pd.to_datetime(s).year}-{pd.to_datetime(e).year}" for s, e in date_intervals]

    # --- 1. Gather all performance data ---
    for i in range(first_alpha, last_alpha):
        alpha_name = f'alpha{i:03d}'
        if not (hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name))):
            continue
            
        print(f"Processing {alpha_name} for summary... ", end='', flush=True)
        
        try:
            full_alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
            if not full_alpha_series.index.is_monotonic_increasing:
                full_alpha_series = full_alpha_series.sort_index()
        except Exception as e:
            print(f"-> FAILED to calculate alpha series: {e}")
            continue

        if full_alpha_series.empty:
            print(f"-> Skipping, no signals.")
            continue
            
        for j, (start_str, end_str) in enumerate(date_intervals):
            start_dt = pd.to_datetime(start_str)
            end_dt = pd.to_datetime(end_str)
            
            try:
                interval_price_data = full_price_data.loc[pd.IndexSlice[start_dt:end_dt]]
                interval_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_dt:end_dt]]
                
                if interval_alpha_series.empty or interval_price_data.empty: continue
                    
                strategy_returns, portfolio_info = run_rank_backtest(interval_price_data, interval_alpha_series)
                
                if strategy_returns.empty: continue

                # --- METRIC CALCULATION (WITH IR) ---
                daily_turnover = portfolio_info['turnover'].groupby('date').first()
                daily_cost = daily_turnover * (5 / 10000.0)
                returns_series_net = strategy_returns - daily_cost.reindex(strategy_returns.index).fillna(0)
                
                benchmark_returns = interval_price_data['returns'].groupby(level='date').mean().reindex(returns_series_net.index).fillna(0)
                excess_returns = returns_series_net - benchmark_returns
                
                std_dev_net = np.std(returns_series_net); std_dev_net = 1e-6 if std_dev_net == 0 else std_dev_net
                tracking_error = np.std(excess_returns); tracking_error = 1e-6 if tracking_error == 0 else tracking_error * np.sqrt(252)
                
                sharpe = np.mean(returns_series_net) * 252 / (std_dev_net * np.sqrt(252))
                annualized_excess_return = np.mean(excess_returns) * 252
                ir = annualized_excess_return / tracking_error
                
                total_ret = (1 + returns_series_net).prod() - 1
                cum_ret_net = (1 + returns_series_net).cumprod()
                peak = cum_ret_net.expanding(min_periods=1).max()
                drawdown = (cum_ret_net / peak) - 1
                max_dd = drawdown.min()
                avg_turnover = daily_turnover.mean()
                
                all_metrics_data.append({
                    'alpha': alpha_name, 'interval': interval_names[j], 'ir': ir,
                    'sharpe': sharpe, 'return': total_ret, 'max_drawdown': max_dd, 'turnover': avg_turnover
                })
                
            except Exception as e:
                print(f"\n  -> FAILED to process interval for {alpha_name}: {e}")
        
        print("-> Done.")
        
    if not all_metrics_data:
        print("No data was generated. Cannot create report.")
        return

    # --- 2. Structure the data into pivot tables for each metric ---
    metrics_df = pd.DataFrame(all_metrics_data)
    
    # Main pivot is now Information Ratio
    ir_pivot = metrics_df.pivot_table(index='alpha', columns='interval', values='ir')
    
    # Create corresponding pivot tables for all tooltip data
    sharpe_pivot = metrics_df.pivot_table(index='alpha', columns='interval', values='sharpe')
    return_pivot = metrics_df.pivot_table(index='alpha', columns='interval', values='return')
    drawdown_pivot = metrics_df.pivot_table(index='alpha', columns='interval', values='max_drawdown')
    turnover_pivot = metrics_df.pivot_table(index='alpha', columns='interval', values='turnover')

    # Create the rich text for the tooltips (hover text)
    tooltip_df = pd.DataFrame(
        'Sharpe: ' + sharpe_pivot.map('{:.2f}'.format) +
        ', Total Return: ' + return_pivot.map('{:.2%}'.format) + 
        ', Max Drawdown: ' + drawdown_pivot.map('{:.2%}'.format) +
        ', Avg Turnover: ' + turnover_pivot.map('{:.2%}'.format),
        index=ir_pivot.index, columns=ir_pivot.columns
    ).fillna('')

    # --- 3. Style the DataFrame based on IR ---
    styled_df = ir_pivot.style \
        .set_caption("Alpha Performance Summary: Information Ratio (Net of Fees)") \
        .background_gradient(cmap='RdYlGn', axis=None, vmin=-1.0, vmax=1.0) \
        .format('{:.2f}', na_rep='-') \
        .set_tooltips(tooltip_df)
    
    # Get the raw HTML for the table body and head from the Styler
    html_table = styled_df.to_html()

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Alpha Performance Summary</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            
            body {{
                font-family: 'Roboto', sans-serif;
                background-color: #f4f7f6;
                color: #333;
                margin: 20px;
            }}
            .container {{
                max-width: 95%;
                margin: auto;
                background: #fff;
                padding: 20px 40px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .table-container {{
                overflow-x: auto; /* Makes the table scrollable horizontally */
                width: 100%;
            }}
            table {{
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                width: 100%;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            table caption {{
                caption-side: top;
                font-size: 1.5em;
                font-weight: bold;
                margin: 1em 0 .75em;
                color: #34495e;
            }}
            thead tr {{
                background-color: #34495e;
                color: #ffffff;
                text-align: center;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #dddddd;
                text-align: center;
            }}
            tbody th {{
                font-weight: bold;
                background-color: #f3f3f3;
            }}
            tbody tr:hover {{
                background-color: #f1f1f1;
            }}

            /* Tooltip styling from pandas Styler */
            {html_table.split('<style type="text/css">')[1].split('</style>')[0]}
            
            /* Overriding default tooltip for better appearance */
            .pd-tooltip {{
                position:relative;
            }}
            .pd-tooltip .pd-tooltip-text {{
                visibility: hidden;
                position: absolute;
                z-index: 100;
                width: 200px;
                background-color: #333;
                color: #fff;
                text-align: left;
                padding: 10px;
                border-radius: 5px;
                bottom: 110%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
                pointer-events: none;
            }}
            .pd-tooltip:hover .pd-tooltip-text {{
                visibility: visible;
                opacity: 1;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Alpha Performance Analysis</h1>
            <p>This report summarizes alpha performance across different time intervals. The primary metric displayed is the <b>Information Ratio (IR)</b>.</p>
            <p><em>Hover over any cell to see detailed metrics for that period.</em></p>
            
            <!-- Inject the table HTML generated by pandas -->
            <div class="table-container">
                {html_table.split('</style>')[1]}
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(report_dir, "alpha_summary_IR_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"\n--- HTML Summary Report Generated at '{report_path}' ---")



# def run_walk_forward_analysis(alpha_calculator, full_price_data, in_sample_years=3, out_of_sample_years=1):
#     """
#     Performs a walk-forward analysis for each alpha.
#     Generates one PDF report per alpha showing the performance in each OOS fold.
#     """
#     print("\n--- Starting Walk-Forward Analysis ---")
    
#     # Create a directory to store the reports
#     report_dir = "walk_forward_reports"
#     if not os.path.exists(report_dir):
#         os.makedirs(report_dir)
        
#     # Get the overall date range from the data
#     all_dates = full_price_data.index.get_level_values('date').unique()
#     start_date = all_dates.min()
#     end_date = all_dates.max()
    
#     # Define the size of our rolling windows
#     in_sample_period = pd.DateOffset(years=in_sample_years)
#     out_of_sample_period = pd.DateOffset(years=out_of_sample_years)
    
#     # Loop through each alpha in the calculator
#     for i in range(1, 102):
#         alpha_name = f'alpha{i:03d}'
#         if not (hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name))):
#             continue
            
#         print(f"\n--- Analyzing {alpha_name} ---")
#         pdf_path = os.path.join(report_dir, f"{alpha_name}_walk_forward.pdf")
        
#         with backend_pdf.PdfPages(pdf_path) as pdf:
#             all_oos_returns = []
            
#             # Define the start of the first in-sample period
#             current_is_start = start_date
            
#             # "Walk" through the data
#             while current_is_start + in_sample_period + out_of_sample_period <= end_date:
#                 # Define the dates for the current fold
#                 current_is_end = current_is_start + in_sample_period
#                 current_oos_start = current_is_end + pd.DateOffset(days=1)
#                 current_oos_end = current_oos_start + out_of_sample_period
                
#                 fold_title = f"OOS: {current_oos_start.date()} to {current_oos_end.date()}"
#                 print(f"  Processing fold: {fold_title}")
                
#                 # Get the data for the out-of-sample period
#                 oos_price_data = full_price_data.loc[pd.IndexSlice[current_oos_start:current_oos_end, :]]
                
#                 # We need to calculate the alpha on a slightly larger dataset to have enough history
#                 # for rolling functions at the start of the OOS period.
#                 alpha_calc_start = current_oos_start - pd.DateOffset(days=365) # 1 year buffer
#                 alpha_calc_data = full_price_data.loc[pd.IndexSlice[alpha_calc_start:current_oos_end, :]]
                
#                 try:
#                     # Calculate alpha signals for this extended period
#                     alpha_series = getattr(Alpha101(alpha_calc_data), alpha_name)().dropna()
#                     # Filter signals to just the OOS period
#                     alpha_series_oos = alpha_series.loc[pd.IndexSlice[current_oos_start:current_oos_end, :]]

#                     if alpha_series_oos.empty:
#                         print("    -> No valid signals in this OOS period. Skipping fold.")
#                         current_is_start += out_of_sample_period # Move to the next fold
#                         continue
                        
#                     # Backtest on the OOS data
#                     strategy_returns, portfolio_info = run_rank_backtest(oos_price_data, alpha_series_oos)
#                     all_oos_returns.append(strategy_returns)
                    
#                     # Create and save the plot for this fold
#                     fig = plt.figure(figsize=(11.69, 8.27))
#                     analyze_performance(strategy_returns, portfolio_info, oos_price_data, fig=fig, title=f"{alpha_name}\n{fold_title}")
#                     pdf.savefig(fig)
#                     plt.close(fig)
                    
#                 except Exception as e:
#                     print(f"    -> FAILED to process fold: {e}")
#                     fig_err, ax_err = plt.subplots(figsize=(11.69, 8.27))
#                     ax_err.text(0.5, 0.5, f'Failed to process fold\n{fold_title}\nError: {e}', ha='center', va='center', color='red')
#                     pdf.savefig(fig_err)
#                     plt.close(fig_err)

#                 # Move the window forward by the OOS period length
#                 current_is_start += out_of_sample_period
            
#             # After looping through all folds, plot the stitched results
#             if all_oos_returns:
#                 print("  Stitching OOS periods for overall performance plot...")
#                 stitched_returns = pd.concat(all_oos_returns)
#                 stitched_portfolio_info = pd.DataFrame({'turnover': stitched_returns * 0}) # Dummy for now
                
#                 fig_stitched = plt.figure(figsize=(11.69, 8.27))
#                 analyze_performance(stitched_returns, stitched_portfolio_info, full_price_data, fig=fig_stitched, title=f"{alpha_name}\nOverall Stitched OOS Performance")
#                 pdf.savefig(fig_stitched)
#                 plt.close(fig_stitched)