import pandas as pd


def run_backtest(price_data, alpha_series, quantiles=5):
    """
    Runs a simple dollar-neutral, long-short backtest on a given alpha series.

    This is a very simple backtest that just forms portfolios based on quantiles of the alpha series. 

        Comment: It does not seem to work well at all.
    """
    # Create a DataFrame with alpha and forward returns
    df = pd.DataFrame({
        'alpha': alpha_series,
        'fwd_returns': price_data['returns'].groupby(level='asset').shift(-1)
    })
    df.dropna(inplace=True)
    
    # Form Portfolios based on Quantiles
    if not df.empty:
        df['quantile'] = df.groupby(level='date')['alpha'] \
                           .transform(lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop'))
    else:
        return pd.Series(dtype=float), pd.DataFrame(columns=['weights', 'quantile', 'turnover'])
        
    # Calculate Portfolio Weights
    df['weights'] = 0.0
    df.loc[df['quantile'] == 0, 'weights'] = -1.0
    df.loc[df['quantile'] == (quantiles - 1), 'weights'] = 1.0
    
    # Normalize weights to be dollar-neutral
    daily_abs_sum_weights = df.groupby(level='date')['weights'].transform(lambda x: x.abs().sum())
    df.loc[:, 'weights'] = df['weights'] / daily_abs_sum_weights.replace(0, 1)
    
    # --- THIS IS THE FIX for the FutureWarning ---
    # Change the inplace operation to an explicit assignment
    df['weights'] = df['weights'].fillna(0)
    # --- END OF FIX ---
    
    # Calculate Strategy Returns
    strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())
    
    # Calculate Turnover
    df['weights_change'] = (df['weights'] - df.groupby(level='asset')['weights'].shift(1)).fillna(df['weights'])
    daily_turnover = df['weights_change'].abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'
    
    # Create the portfolio_info DataFrame and join the turnover Series
    portfolio_info = df[['weights', 'quantile']]
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    return strategy_returns, portfolio_info


def run_alpha999_backtest(price_data, alpha_series):
    """
    Special backtest function for alpha999 that uses ML signals directly as position sizing.
    This avoids the ranking system degeneracy for single-asset portfolios.
    
    Args:
        price_data: DataFrame with price data
        alpha_series: Series with ML signals (-1000, 0, 1000) for alpha999
        
    Returns:
        tuple: (strategy_returns, portfolio_info)
    """
    print("--------------------------------")
    print(f"Running alpha999 backtest with alpha_series: {alpha_series.name}")
    print("--------------------------------")
    # Create a DataFrame with alpha and forward returns
    df = pd.DataFrame({
        'alpha': alpha_series,
        'fwd_returns': price_data['returns'].groupby(level='asset').shift(-1)
    })
    df.dropna(inplace=True)

    if df.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=['weights', 'turnover'])

    # Convert ML signals directly to position sizes
    # -1000 -> -1 (short), 0 -> 0 (neutral), 1000 -> 1 (long)
    df['ml_signal'] = df['alpha'] / 1000.0  # Convert back to -1, 0, 1
    
    # Use ML signals directly as weights (no ranking needed)
    df['weights'] = df['ml_signal'].copy()
    
    # For single asset, we can use the signal directly as position size
    # No normalization needed since we're not doing cross-sectional ranking
    
    # Calculate Strategy Returns
    strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())

    # Calculate Turnover
    df['weights_change'] = (df['weights'] - df.groupby(level='asset')['weights'].shift(1)).fillna(df['weights'])
    daily_turnover = df['weights_change'].abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'

    portfolio_info = df[['weights']].copy()
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    return strategy_returns, portfolio_info


def run_rank_backtest(price_data, alpha_series):
    """
    Runs a dollar-neutral, long-short backtest using rank-based weighting.
    This version includes the fix for the pandas FutureWarning.
    """
    # Special handling for alpha999
    if alpha_series.name == 'alpha999' or (alpha_series.abs() > 110).any():
        print("Detected alpha999 signals - using special ML backtest")
        return run_alpha999_backtest(price_data, alpha_series)
    
    # Create a DataFrame with alpha and forward returns
    df = pd.DataFrame({
        'alpha': alpha_series,
        'fwd_returns': price_data['returns'].groupby(level='asset').shift(-1)
    })
    df.dropna(inplace=True)

    if df.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=['weights', 'turnover'])

    # 1. Rank the alpha signals cross-sectionally for each day (from 0.0 to 1.0)
    df['rank'] = df.groupby(level='date')['alpha'].rank(pct=True)
    
    # 2. Center the ranks to create a spread from -0.5 to 0.5
    df['centered_rank'] = df['rank'] - 0.5
    
    # 3. Normalize the weights to be dollar-neutral with unit leverage
    # We divide by the sum of all positive centered ranks for each day.
    daily_positive_rank_sum = df[df['centered_rank'] > 0].groupby(level='date')['centered_rank'].transform('sum')
    df['weights'] = df['centered_rank'] / daily_positive_rank_sum.replace(0, 1)
    

    df['weights'] = df['weights'].fillna(0)

    # Calculate Strategy Returns
    strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())

    # Calculate Turnover
    df['weights_change'] = (df['weights'] - df.groupby(level='asset')['weights'].shift(1)).fillna(df['weights'])
    daily_turnover = df['weights_change'].abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'

    portfolio_info = df[['weights']].copy() # We don't have 'quantile' in this backtester
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    return strategy_returns, portfolio_info
