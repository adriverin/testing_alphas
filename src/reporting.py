import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
import os
from .backtests import run_rank_backtest


def _detect_timezone_format(data_index):
    """
    Detect if the data index is timezone-aware or naive.
    
    Args:
        data_index: MultiIndex with 'date' level
        
    Returns:
        bool: True if timezone-aware, False if timezone-naive
    """
    date_level = data_index.get_level_values('date')
    if len(date_level) > 0:
        return date_level[0].tz is not None
    return False


def _create_compatible_datetime(date_str, is_timezone_aware):
    """
    Create datetime that matches the data's timezone format.
    
    Args:
        date_str: Date string to convert
        is_timezone_aware: Whether to create timezone-aware datetime
        
    Returns:
        pd.Timestamp: Compatible datetime
    """
    if is_timezone_aware:
        return pd.to_datetime(date_str, utc=True)
    else:
        return pd.to_datetime(date_str)


def generate_date_intervals(start_date, end_date, n):
    """
    Generate n equally spaced date intervals between start_date and end_date.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format 
        n (int): Number of intervals to generate
        
    Returns:
        list: List of tuples (start_str, end_str) for each interval
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Calculate interval length
    total_duration = end_dt - start_dt
    interval_duration = total_duration / n
    
    intervals = []
    for i in range(n):
        interval_start = start_dt + i * interval_duration
        interval_end = start_dt + (i + 1) * interval_duration
        
        # Format as strings
        start_str = interval_start.strftime('%Y-%m-%d')
        end_str = interval_end.strftime('%Y-%m-%d')
        
        intervals.append((start_str, end_str))
    
    return intervals



def _detect_data_frequency(returns_series):
    """
    Detect the frequency of returns data and return the appropriate annualization factor.
    
    Args:
        returns_series: Time series of returns
        
    Returns:
        tuple: (periods_per_year, frequency_label)
    """
    if len(returns_series) < 2:
        return 252, "daily"  # Default fallback
    
    # Calculate median time difference between consecutive observations
    time_diffs = returns_series.index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    
    # Convert to hours for comparison
    hours = median_diff.total_seconds() / 3600
    
    if hours <= 1.5:  # 1 hour or less
        return 8760, "hourly"  # 24*365 hours per year
    elif hours <= 6:  # 4 hours or less
        return 2190, "4-hourly"  # 6*365 periods per year (4h intervals)
    elif hours <= 25:  # 1 day or less  
        return 252, "daily"  # 252 trading days per year
    elif hours <= 168:  # 1 week or less
        return 52, "weekly"  # 52 weeks per year
    else:
        return 12, "monthly"  # 12 months per year

def analyze_performance(returns_series, portfolio_info, price_data, fig, title='Strategy Performance', transaction_cost_bps=50, log_scale=False):
    """
    Calculates performance metrics and creates a 3-panel plot on a given figure.
    - Top panel: Cumulative Returns (Equity Curve) vs. Benchmark.
    - Middle panel (NEW): Drawdown from Peak over time.
    - Bottom panel: Daily Portfolio Turnover.
    
    Enhanced with automatic data frequency detection for proper annualization.
    """
    fig.suptitle(title, fontsize=16)
    
    # Detect data frequency for proper annualization
    periods_per_year, freq_label = _detect_data_frequency(returns_series)
    print(f"📊 Detected data frequency: {freq_label} ({periods_per_year} periods/year)")
    
    # --- Change subplot layout to a 3x1 grid ---
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1) # New middle plot
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1) # Old bottom plot is now ax3

    if returns_series.empty:
        ax1.text(0.5, 0.5, 'No returns to analyze for this period.', ha='center', va='center')
        return {}
        
    active_start_date = returns_series.index.min()
    active_end_date = returns_series.index.max()
    
    daily_turnover = portfolio_info['turnover'].groupby('date').first()
    daily_cost = daily_turnover * (transaction_cost_bps / 10000.0)
    # Ensure cost series is aligned with returns before subtracting
    returns_series_net = returns_series - daily_cost.reindex(returns_series.index).fillna(0)
    
    # --- Top Panel: Equity Curve (ax1) ---
    benchmark_returns = price_data['returns'].groupby(level='date').mean()
    benchmark_returns_aligned = benchmark_returns.loc[active_start_date:active_end_date]
    cumulative_benchmark_returns = (1 + benchmark_returns_aligned).cumprod()
    ax1.plot(cumulative_benchmark_returns.index, cumulative_benchmark_returns.values, label='Buy & Hold Benchmark', color='gray', linestyle='--')

    cumulative_returns_gross = (1 + returns_series).cumprod()
    ax1.plot(cumulative_returns_gross.index, cumulative_returns_gross.values, label='Alpha Strategy (Gross)', color='blue', linewidth=2)

    cumulative_returns_net = (1 + returns_series_net).cumprod()
    ax1.plot(cumulative_returns_net.index, cumulative_returns_net.values, label=f'Alpha Strategy (Net, {transaction_cost_bps} bps)', color='green', linewidth=2)
    
    # --- Metrics Calculations ---
    excess_returns = returns_series_net - benchmark_returns_aligned
    tracking_error = np.std(excess_returns) * np.sqrt(periods_per_year)
    if tracking_error == 0:
        information_ratio = np.nan
    else:
        annualized_excess_return = np.mean(excess_returns) * periods_per_year
        information_ratio = annualized_excess_return / tracking_error

    std_dev_net = np.std(returns_series_net); std_dev_net = 1e-6 if std_dev_net == 0 else std_dev_net
    sharpe_ratio_net = np.mean(returns_series_net) / std_dev_net * np.sqrt(periods_per_year)
    annualized_return_net = (cumulative_returns_net.iloc[-1]) ** (periods_per_year / len(cumulative_returns_net)) - 1
    
    peak_net = cumulative_returns_net.expanding(min_periods=1).max()
    drawdown_series = (cumulative_returns_net / peak_net) - 1
    max_drawdown_net = drawdown_series.min()
    avg_drawdown = drawdown_series.mean()

    stats_text = (f"Net Sharpe: {sharpe_ratio_net:.2f}\n"
                  f"Information Ratio: {information_ratio:.2f}\n"
                  f"Net Ann. Return: {annualized_return_net:.2%}\n"
                  f"Net Max Drawdown: {max_drawdown_net:.2%}\n")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    
    if log_scale:
        ax1.set_yscale('log')
    
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend(loc='upper center')
    ax1.grid(True)
    
    # --- Middle Panel: Drawdown Plot (ax2) ---
    ax2.plot(drawdown_series.index, drawdown_series.values, label='Drawdown', color='red', alpha=0.7)
    ax2.fill_between(drawdown_series.index, drawdown_series.values, 0, color='red', alpha=0.1)
    ax2.plot(drawdown_series.index, np.full(len(drawdown_series), avg_drawdown), label=f'Avg Drawdown: {avg_drawdown:.2%}', color='black', linestyle='--')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.grid(True)

    # --- Bottom Panel: Daily Turnover (now ax3) ---
    turnover_series = portfolio_info['turnover'].groupby('date').first().dropna()
    ax3.plot(turnover_series.index, turnover_series.values, label='Daily Turnover', color='orange', alpha=0.6, linewidth=1)
    rolling_turnover = turnover_series.rolling(window=22).mean()
    ax3.plot(rolling_turnover.index, rolling_turnover.values, label='1-Month Avg. Turnover', color='darkred', linestyle='-')
    avg_turnover = turnover_series.mean()
    ax3.axhline(avg_turnover, color='black', linestyle=':', label=f'Avg Turnover: {avg_turnover:.2%}')
    
    ax3.set_ylabel('Portfolio Turnover')
    ax3.set_xlabel('Date')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # --- Final Layout Adjustments ---
    # Hide x-axis labels on the top and middle plots for a cleaner look
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return {
        'sharpe': sharpe_ratio_net,
        'ir': information_ratio, 
        'annual_return': annualized_return_net,
        'max_drawdown': max_drawdown_net
    }



def generate_full_report(alpha_calculator, price_data, pdf_path='reports/alpha_report.pdf', first_alpha=1, last_alpha=106, log_scale=False):
    """
    Calculates all implemented alphas and backtests them.
    - Adds a Buy & Hold benchmark to each plot.
    """
    print(f"\n--- Generating Full Alpha Report ---")
    
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

                    strategy_returns, portfolio_info = run_rank_backtest(price_data, alpha_series)
                    
                    fig = plt.figure(figsize=(11.69, 8.27))
                    
                    

                    analyze_performance(strategy_returns, portfolio_info, price_data, fig=fig, title=alpha_name, log_scale=log_scale)
                    
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






# legacy function; still working but not useful

def generate_interval_report(alpha_calculator, full_price_data, date_intervals, report_dir="reports/interval_reports", first_alpha=1, last_alpha=106, stop_loss_pct=None, backtest_func=run_rank_backtest, log_scale=False):    
    """
    Performs a chunked backtest for each alpha over specified date intervals.
    Generates one PDF report per alpha, with each page showing performance in one interval.
    """
    print(f"\n--- Starting Interval-Based Analysis ---")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Detect timezone format of the data once
    is_timezone_aware = _detect_timezone_format(full_price_data.index)
    print(f"📅 Data timezone format: {'timezone-aware' if is_timezone_aware else 'timezone-naive'}")
        
    # Loop through each alpha in the calculator
    for i in range(first_alpha, last_alpha + 1):  # +1 to make last_alpha inclusive
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
                
                # Create compatible datetime objects based on data timezone format
                start_dt = _create_compatible_datetime(start_str, is_timezone_aware)
                end_dt = _create_compatible_datetime(end_str, is_timezone_aware)
                
                try:
                    # Filter both price data and the pre-calculated alpha series for the interval
                    interval_price_data = full_price_data.loc[pd.IndexSlice[start_dt:end_dt, :]]
                    interval_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_dt:end_dt, :]]
                    
                    if interval_alpha_series.empty or interval_price_data.empty:
                        print(f"    -> No data in this interval for {alpha_name}. Skipping.")
                        continue
                        
                    # Backtest on the interval data
                    strategy_returns, portfolio_info = backtest_func(interval_price_data, interval_alpha_series, stop_loss_pct=stop_loss_pct)
                    
                    if strategy_returns.empty:
                        print(f"    -> Backtest resulted in no returns for this interval. Skipping.")
                        continue
                    
                    # Create and save the plot for this interval
                    fig = plt.figure(figsize=(11.69, 8.27))
                    analyze_performance(strategy_returns, portfolio_info, interval_price_data, fig=fig, 
                                        title=f"{alpha_name}\n{interval_title}", log_scale=log_scale)
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"    -> FAILED to process interval: {e}")
                    fig_err, ax_err = plt.subplots(figsize=(11.69, 8.27))
                    ax_err.text(0.5, 0.5, f'Failed to process interval\n{interval_title}\nError: {e}', 
                                ha='center', va='center', color='red')
                    pdf.savefig(fig_err)
                    plt.close(fig_err)

    print("\n--- Interval-Based Analysis Generated at ---")
    print(f"{report_dir}")







def generate_summary_html_report(alpha_calculator, full_price_data, date_intervals, report_dir="reports/summary_reports", first_alpha=1, last_alpha=106, stop_loss_pct=None):
    """
    Performs a chunked backtest and generates a single, interactive HTML summary report.
    - The main metric displayed and color-coded is the Information Ratio (IR).
    - Tooltips show Sharpe Ratio, Return, Max Drawdown, and Turnover.
    """
    print(f"\n--- Starting HTML Summary Report Generation (centered on Information Ratio) ---")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Detect timezone format of the data once
    is_timezone_aware = _detect_timezone_format(full_price_data.index)
    print(f"📅 Data timezone format: {'timezone-aware' if is_timezone_aware else 'timezone-naive'}")
        
    all_metrics_data = []
    interval_names = [f"{pd.to_datetime(s).year}-{pd.to_datetime(e).year}" for s, e in date_intervals]

    # --- 1. Gather all performance data ---
    for i in range(first_alpha, last_alpha + 1):  # +1 to make last_alpha inclusive
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
            # Create compatible datetime objects based on data timezone format
            start_dt = _create_compatible_datetime(start_str, is_timezone_aware)
            end_dt = _create_compatible_datetime(end_str, is_timezone_aware)
            
            try:
                interval_price_data = full_price_data.loc[pd.IndexSlice[start_dt:end_dt]]
                interval_alpha_series = full_alpha_series.loc[pd.IndexSlice[start_dt:end_dt]]
                
                if interval_alpha_series.empty or interval_price_data.empty: continue
                    
                strategy_returns, portfolio_info = run_rank_backtest(interval_price_data, interval_alpha_series, stop_loss_pct=stop_loss_pct)
                
                if strategy_returns.empty: continue

                # --- METRIC CALCULATION (WITH IR) ---
                daily_turnover = portfolio_info['turnover'].groupby('date').first()
                daily_cost = daily_turnover * (5 / 10000.0)
                returns_series_net = strategy_returns - daily_cost.reindex(strategy_returns.index).fillna(0)
                
                benchmark_returns = interval_price_data['returns'].groupby(level='date').mean().reindex(returns_series_net.index).fillna(0)
                excess_returns = returns_series_net - benchmark_returns
                
                # Detect frequency for proper annualization
                periods_per_year_summary, _ = _detect_data_frequency(returns_series_net)
                
                std_dev_net = np.std(returns_series_net); std_dev_net = 1e-6 if std_dev_net == 0 else std_dev_net
                tracking_error = np.std(excess_returns); tracking_error = 1e-6 if tracking_error == 0 else tracking_error * np.sqrt(periods_per_year_summary)
                
                sharpe = np.mean(returns_series_net) * periods_per_year_summary / (std_dev_net * np.sqrt(periods_per_year_summary))
                annualized_excess_return = np.mean(excess_returns) * periods_per_year_summary
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
        
    print(f"\n--- HTML Summary Report Generated at ---")
    print(f"{report_path}")