# 101 Formulaic Alphas: A Quantitative Research Framework

This repository contains a full-featured Python framework for the implementation, backtesting, and analysis of quantitative trading strategies, based on the classic finance paper **"101 Formulaic Alphas"** by Zura Kakushadze.

The project is designed to be a robust, end-to-end "alpha factory," allowing a researcher to:
1.  Automatically download and cache historical stock data.
2.  Implement and compute complex alpha signals.
3.  Backtest strategies using a sophisticated, rank-based, dollar-neutral portfolio construction method.
4.  Analyze performance with realistic transaction costs and benchmark comparisons.
5.  Generate detailed reports, including per-alpha interval analysis and a high-level interactive summary heatmap.

## Project Structure

The codebase is organized into a modular structure to promote clarity and maintainability.

*   `main.py`: The main entry point of the application. This script orchestrates the entire workflow from data loading to report generation.
*   `src/`: A directory containing the core logic modules.
    *   `alpha101.py`: Contains the `Alpha101` class, where all 101 alpha formulas are implemented as methods.
    *   `data_loader.py`: Contains the `get_stock_data` function responsible for downloading data from Yahoo Finance (other sources will be implemented when relevant) and caching it locally using Parquet for efficient re-runs.
    *   `reporting.py`: Contains the high-level reporting functions (`generate_interval_report`, `generate_summary_html_report`, `analyze_performance`).
    *   `backtests.py`: Contains the backtesting engines (`run_rank_backtest`) that translate alpha signals into portfolio returns and performance metrics.
*   `reports/`: A directory containing the analysis reports
    *   `summary_reports/`: This directory is created automatically to store the interactive HTML summary reports.
    *   `interval_reports/`: This directory is created automatically to store the detailed, per-alpha PDF reports showing performance across different time intervals.

## Core Features

### 1. Data Pipeline
The `get_stock_data` function uses `yfinance` (other sources will be implemented when relevant) to fetch daily OHLCV data and `pandas` for processing. It features a caching system:
-   Data is saved locally to a `stock_data.parquet` file.
-   On subsequent runs, the script checks if the cache is up-to-date for the requested date range. It only downloads new data if necessary, making repeated analysis fast and efficient.
-   Handles survivor-bias by design (if a broader stock universe is provided).

### 2. Alpha Implementation
The `Alpha101` class provides a full, working implementation of the formulas described in the original paper ("https://arxiv.org/pdf/1601.00991"). Each alpha is a separate method, making the code easy to read and verify against the source material. Next will be to implement alphas from other sources (e.g. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865) and playing with my own ideas.


### 3. Backtesting Engine
The project has evolved from a simple quantile-based backtester (this lead to bad results) to a more robust **rank-based, dollar-neutral, long-short portfolio construction** model (`run_rank_backtest`). This method:
-   Uses the full information content of the alpha signal by weighting positions based on their cross-sectional rank.
-   Maintains dollar neutrality to hedge against broad market movements (beta).
-   Includes a transaction cost model based on daily turnover, providing a "net-of-fees" performance view.

### 4. Reporting & Analysis
The framework generates two types of powerful reports:
-   **Per-Alpha Interval Reports (PDFs):** For each alpha, a detailed PDF is generated, with each page showing a full performance analysis (equity curve, benchmark comparison, turnover) for a different historical time period. This is to test an alpha's stability across different market regimes.
-   **Interactive Summary Report (HTML):** A single HTML file that displays a performance heatmap of all alphas across all tested intervals. The cells are color-coded by Sharpe Ratio, and hovering over any cell reveals detailed metrics (Return, Max Drawdown), allowing for at-a-glance comparison and identification of the most robust strategies.
-   **Per-Interval Reports (PDFs):** For an interval, a detailed PDF is generated, with each page showing a full performance analysis (equity curve, benchmark comparison, turnover) for each alpha. This is compare alphas in a single time interval. Note that this was the first implementation, thus less sofisticated than the previous two.

## How to Run the Code

1.  **Install Dependencies:** Ensure you have the required Python libraries installed.
    ```bash
    pip install pandas numpy yfinance scipy matplotlib seaborn
    ```
2.  **Configure the Main Script:** Open `alpha_testing.py` and configure the main execution block:
    ```python
    from src.get_stock_data import get_stock_data
    from src.alpha101 import Alpha101
    from src.analysis import generate_full_report, generate_interval_report, generate_summary_html_report
    from src.utils import generate_date_intervals

    # Define your stock universe
    sp100_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', ...] 
    
    # Define the full backtest period
    start = '2011-01-01'
    end = '2025-05-31'

    # Load data (will use cache if available)
    price_data = get_stock_data(sp100_tickers, start, end)

    if not price_data.empty:
        # Initialize the calculator
        alpha_calculator = Alpha101(price_data)

        # --- CHOOSE YOUR ANALYSIS (can of course be run together)---
        
        # Option 1: Generate the HTML summary report
        number_of_intervals = 4
        intervals_to_test = generate_date_intervals(start, end, number_of_intervals)
        generate_summary_html_report(alpha_calculator, price_data, intervals_to_test)
        
        # Option 2: Generate detailed PDF reports for each alpha in number_of_intervals intervals between (start, end).
        generate_interval_report(alpha_calculator, price_data, intervals_to_test)

        # Option 3: Generate one detailed PDF reports for all alphas in the interval (start, end).
        generate_full_report(alpha_calculator, price_data, intervals_to_test)



    ```
3.  **Run the script:**
    ```bash
    python alpha_testing.py
    ```
4.  **Check the Output:** Look for the generated reports in the `reports/summary_reports/` and `reports/interval_reports/` directories. Open the `.html` file in a browser like Google Chrome for the best experience (my oldish version of Safari did not work).

*(Note: To force a full re-download of all data, simply delete the `stock_data.parquet` file.)*

## Future Steps & Roadmap for Improvement

This framework serves as a foundation. The next steps focus on increasing the rigor of the research process to build even more confidence in the results.

### Priority 0: Implementation of more alphas
Constant priority for now.
- Use https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865.
- Research ideas.
- Think of ideas.


### Priority 1: Rigorous Out-of-Sample Validation
The most critical next step.
- **Task:** Split the dataset into a historical **In-Sample (IS)** period (e.g., 2011-2020) and a "future" **Out-of-Sample (OOS)** period (e.g., 2021-present).
- **Process:** Use the IS data to run the summary report and *select* the top 5-10 best-performing alphas. Then, test *only* those selected alphas on the OOS data.
- **Goal:** To verify if an alpha's past performance was due to skill or luck. A robust alpha will continue to perform well on the unseen OOS data.

### Priority 2: Alpha Combination (Mega-Alpha)
Individual alphas can be noisy. Combining them creates a more stable, diversified portfolio.
- **Task:** Implement an alpha combination function.
- **Process:**
    1.  Select a basket of robust, preferably uncorrelated, alphas identified from the OOS validation.
    2.  Combine their signals using the **simple average of ranks** method.
    3.  Backtest this new "mega-alpha" signal.
- **Goal:** To create a strategy with a smoother equity curve and a better risk-adjusted return than any single alpha component.

### Priority 3: Factor Risk Analysis
Understand *why* an alpha makes money.
- **Task:** Correlate the alpha strategy's returns to common academic risk factors (e.g., Fama-French factors).
- **Process:** Download factor data and use a library like `statsmodels` to run a linear regression.
- **Goal:** To determine if an alpha is truly unique or just a complicated proxy for a well-known factor like Value or Momentum. A high R-squared is a red flag; a statistically significant intercept (the "alpha" in the regression) is the desired outcome.

