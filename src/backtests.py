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
    This properly interprets ML signals as position changes and holds positions until the next signal.
    
    Args:
        price_data: DataFrame with price data
        alpha_series: Series with ML signals (-1000, 0, 1000) for alpha999
        
    Returns:
        tuple: (strategy_returns, portfolio_info)
    """
    print("--------------------------------")
    print(f"Running alpha999 backtest with alpha_series: {alpha_series.name}")
    print("--------------------------------")
    
    # Convert ML signals directly to position sizes
    # -1000 -> -1 (short), 0 -> 0 (neutral), 1000 -> 1 (long)
    ml_signals = alpha_series / 1000.0  # Convert back to -1, 0, 1
    
    # Get the underlying asset returns for holding period calculation
    asset_returns = price_data['returns']
    
    # FIXED LOGIC: Interpret ML signals as position targets that should be held
    # When we get a non-zero signal, hold that position until next non-zero signal
    positions = pd.Series(0.0, index=ml_signals.index)
    
    for asset in ml_signals.index.get_level_values('asset').unique():
        asset_signals = ml_signals.xs(asset, level='asset')
        current_position = 0.0
        
        for date, signal in asset_signals.items():
            if signal != 0.0:
                # Non-zero signal: change position
                current_position = signal
            # Always set the current position (whether it changed or not)
            positions.loc[(date, asset)] = current_position
    
    # Strategy returns = position * asset return for each day
    # This properly accounts for holding period returns
    strategy_returns_by_asset = positions * asset_returns
    
    # Aggregate across assets (for multi-asset portfolios)
    strategy_returns = strategy_returns_by_asset.groupby(level='date').sum()
    
    # Calculate turnover based on position changes
    position_changes = positions.groupby(level='asset').diff().fillna(positions)
    daily_turnover = position_changes.abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'
    
    # Create portfolio info
    portfolio_info = pd.DataFrame({
        'weights': positions,
        'turnover': daily_turnover.reindex(positions.index, level='date')
    })
    
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
