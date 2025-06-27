#
# factor_analysis.py - Fama-French 3-Factor Model Analysis
#
import pandas as pd
import pandas_datareader.data as web # For Fama-French data
import statsmodels.api as sm
import os

# Import our existing functions
from src.data_loader import get_stock_data
from src.alpha101 import Alpha101
from src.combiner import combine_alphas
from src.backtests import run_rank_backtest
from src.utils import get_fama_french_factors




if __name__ == '__main__':
    
    # --- 1. Define dataset and strategy parameters ---
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
        'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT'
    ]
    start_date = '2011-01-01'
    end_date = '2025-05-31'
    
    # The basket of your best alphas, identified from previous validation
    core_alphas = ['alpha003', 'alpha041', 'alpha042', 'alpha054', 'alpha083', 'alpha101']
    
    # --- 2. Generate Strategy Returns ---
    print("--- Loading Price Data and Generating Strategy Returns ---")
    price_data = get_stock_data(sp100_tickers, start_date=start_date, end_date=end_date)
    
    if not price_data.empty:
        alpha_calculator = Alpha101(price_data)
        mega_alpha_signal = combine_alphas(alpha_calculator, core_alphas)
        strategy_returns_gross, portfolio_info = run_rank_backtest(price_data, mega_alpha_signal)
        
        # Calculate net returns after transaction costs
        daily_turnover = portfolio_info['turnover'].groupby('date').first()
        daily_cost = daily_turnover * (5 / 10000.0) # 5 bps cost
        strategy_returns_net = strategy_returns_gross - daily_cost.reindex(strategy_returns_gross.index).fillna(0)
        strategy_returns_net.name = 'strategy_net_returns'
        
        # --- 3. Get Fama-French Factor Data ---
        ff_data = get_fama_french_factors(start_date, end_date)
        
        if ff_data is not None:
            # --- 4. Merge Strategy Returns with Factor Data ---
            print("\n--- Merging Strategy and Factor Data ---")
            
            # The risk-free rate (RF) is needed to calculate excess returns
            df_merged = pd.merge(strategy_returns_net, ff_data, left_index=True, right_index=True, how='inner')
            
            # Calculate the strategy's excess return
            df_merged['strategy_excess_ret'] = df_merged['strategy_net_returns'] - df_merged['RF']
            
            # --- 5. Run the Factor Regression ---
            print("\n--- Running Fama-French 3-Factor Regression ---")
            
            # Define the independent variables (the factors)
            X = df_merged[['Mkt-RF', 'SMB', 'HML']]
            # Define the dependent variable (our strategy's excess return)
            y = df_merged['strategy_excess_ret']
            
            # Add a constant (the intercept) to the independent variables
            X = sm.add_constant(X)
            
            # Run the Ordinary Least Squares (OLS) regression
            model = sm.OLS(y, X).fit()
            
            # --- 6. Print and Interpret the Results ---
            print("\n" + "="*80)
            print("                        FACTOR ANALYSIS RESULTS")
            print("="*80)
            print(model.summary())
            print("="*80)

            # --- Interpretation Guide ---
            alpha_annualized = model.params['const'] * 252 * 100
            alpha_p_value = model.pvalues['const']
            
            print("\n--- KEY TAKEAWAYS ---")
            print(f"Annualized Alpha (Intercept): {alpha_annualized:.2f}% per year.")
            print(f"Alpha P-value: {alpha_p_value:.4f}")
            if alpha_p_value < 0.05:
                print("  -> The alpha is statistically significant. This is a strong positive result.")
            else:
                print("  -> The alpha is NOT statistically significant. The result could be due to random chance.")
            
            print(f"\nMarket Beta (Mkt-RF Coefficient): {model.params['Mkt-RF']:.3f}")
            print(f"  -> A value close to 0 indicates the strategy is well-hedged against market movements.")

            print(f"\nSize Beta (SMB Coefficient): {model.params['SMB']:.3f}")
            print(f"  -> A positive value suggests a tilt towards small-cap stocks.")
            
            print(f"\nValue Beta (HML Coefficient): {model.params['HML']:.3f}")
            print(f"  -> A positive value suggests a tilt towards value stocks (vs growth stocks).")

            print(f"\nR-squared: {model.rsquared:.3f}")
            print(f"  -> This means {model.rsquared:.1%} of your strategy's daily returns are explained by these 3 common factors.")
            print("  -> A lower R-squared is generally better, indicating a more unique strategy.")