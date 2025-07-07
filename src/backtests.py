import pandas as pd
import numpy as np


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


def track_position_pnl(weights, returns, price_data):
    """
    Track individual position P&L for stop-loss calculations.
    
    Args:
        weights: Series with position weights
        returns: Series with asset returns
        price_data: DataFrame with price data (for entry prices)
        
    Returns:
        dict: Position tracking data
    """
    position_tracker = {}
    
    # Get all dates and assets
    dates = weights.index.get_level_values('date').unique().sort_values()
    assets = weights.index.get_level_values('asset').unique()
    
    for asset in assets:
        asset_weights = weights.xs(asset, level='asset')
        asset_returns = returns.xs(asset, level='asset')
        
        # Get prices for this asset
        if 'close' in price_data.columns:
            asset_prices = price_data['close'].xs(asset, level='asset')
        else:
            # Calculate approximate prices from returns
            asset_prices = (1 + asset_returns).cumprod() * 100
        
        current_position = 0.0
        entry_date = None
        entry_price = None
        entry_weight = None
        cumulative_return = 0.0
        
        for date in dates:
            if date not in asset_weights.index:
                continue
                
            new_weight = asset_weights[date]
            new_position = 1 if new_weight > 0 else (-1 if new_weight < 0 else 0)
            
            # Check for position changes
            if new_position != current_position:
                # Close existing position
                if current_position != 0 and entry_date is not None:
                    position_key = f"{asset}_{entry_date.strftime('%Y%m%d')}"
                    position_tracker[position_key] = {
                        'asset': asset,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': asset_prices[date] if date in asset_prices.index else entry_price,
                        'entry_weight': entry_weight,
                        'position_type': 'LONG' if current_position > 0 else 'SHORT',
                        'cumulative_return': cumulative_return,
                        'pnl_percent': (cumulative_return / abs(entry_weight)) * 100 if entry_weight != 0 else 0,
                        'days_held': (date - entry_date).days,
                        'status': 'CLOSED'
                    }
                
                # Open new position
                if new_position != 0:
                    current_position = new_position
                    entry_date = date
                    entry_weight = new_weight
                    entry_price = asset_prices[date] if date in asset_prices.index else 100
                    cumulative_return = 0.0
                else:
                    current_position = 0
                    entry_date = None
                    entry_weight = None
                    entry_price = None
            
            # Update cumulative return for existing position
            if current_position != 0 and date in asset_returns.index:
                daily_return = asset_returns[date]
                cumulative_return += new_weight * daily_return
        
        # Handle any remaining open position
        if current_position != 0 and entry_date is not None:
            position_key = f"{asset}_{entry_date.strftime('%Y%m%d')}"
            last_date = dates[-1]
            position_tracker[position_key] = {
                'asset': asset,
                'entry_date': entry_date,
                'exit_date': last_date,
                'entry_price': entry_price,
                'exit_price': asset_prices[last_date] if last_date in asset_prices.index else entry_price,
                'entry_weight': entry_weight,
                'position_type': 'LONG' if current_position > 0 else 'SHORT',
                'cumulative_return': cumulative_return,
                'pnl_percent': (cumulative_return / abs(entry_weight)) * 100 if entry_weight != 0 else 0,
                'days_held': (last_date - entry_date).days,
                'status': 'OPEN'
            }
    
    return position_tracker


def apply_stop_loss(df, price_data, stop_loss_pct):
    """
    Apply individual position stop-loss to backtest data.
    
    Args:
        df: Backtest DataFrame with weights
        price_data: Price data for P&L calculation
        stop_loss_pct: Stop-loss percentage (e.g., -5.0 for 5% loss)
        
    Returns:
        tuple: (modified_df, stop_loss_info, modified_strategy_returns)
    """
    if stop_loss_pct is None:
        # Calculate strategy returns for baseline
        df['fwd_returns'] = price_data['returns'].groupby(level='asset').shift(-1)
        strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())
        return df, {'stop_loss_triggers': 0, 'stopped_positions': []}, strategy_returns
    
    print(f"🛡️ Applying individual position stop-loss at {stop_loss_pct}%")
    
    # Create a copy to modify
    df_modified = df.copy()
    
    # Track position states
    position_states = {}  # {(asset, entry_date): {'entry_weight', 'cumulative_pnl', 'entry_price'}}
    stopped_positions = []
    stop_loss_triggers = 0
    
    dates = df.index.get_level_values('date').unique().sort_values()
    assets = df.index.get_level_values('asset').unique()
    
    for date in dates:
        for asset in assets:
            if (date, asset) not in df.index:
                continue
                
            current_weight = df.loc[(date, asset), 'weights']
            current_position = 1 if current_weight > 0 else (-1 if current_weight < 0 else 0)
            daily_return = price_data['returns'].loc[(date, asset)] if (date, asset) in price_data['returns'].index else 0
            
            # Find active position for this asset
            active_position_key = None
            for key in position_states:
                if key[0] == asset and position_states[key]['active']:
                    active_position_key = key
                    break
            
            # Check for new position
            if current_position != 0 and active_position_key is None:
                # New position entry
                position_key = (asset, date)
                position_states[position_key] = {
                    'entry_weight': current_weight,
                    'cumulative_pnl': 0.0,
                    'active': True,
                    'position_type': 'LONG' if current_position > 0 else 'SHORT'
                }
                active_position_key = position_key
            
            # Update existing position P&L
            if active_position_key and position_states[active_position_key]['active']:
                # Calculate daily contribution to P&L
                daily_contribution = current_weight * daily_return
                position_states[active_position_key]['cumulative_pnl'] += daily_contribution
                
                # Calculate P&L percentage
                entry_weight = position_states[active_position_key]['entry_weight']
                if abs(entry_weight) > 0:
                    pnl_percent = (position_states[active_position_key]['cumulative_pnl'] / abs(entry_weight)) * 100
                    

                    # Check stop-loss condition
                    # For losses: pnl_percent is negative (e.g., -3.5%)
                    # Stop-loss threshold is negative (e.g., -5.0%)
                    # Trigger if loss is worse than threshold: -3.5% < -5.0% is FALSE, but -6.0% < -5.0% is TRUE
                    if pnl_percent < stop_loss_pct:
                        # Trigger stop-loss
                        print(f"🔴 Stop-loss triggered: {asset} on {date.strftime('%Y-%m-%d')}, P&L: {pnl_percent:.2f}%")
                        
                        # Set weight to zero (exit position)
                        df_modified.loc[(date, asset), 'weights'] = 0.0
                        
                        # Mark position as stopped
                        position_states[active_position_key]['active'] = False
                        stopped_positions.append({
                            'asset': asset,
                            'entry_date': active_position_key[1],
                            'stop_date': date,
                            'pnl_percent': pnl_percent,
                            'position_type': position_states[active_position_key]['position_type']
                        })
                        stop_loss_triggers += 1
            
            # Check for position exit (due to signal change)
            if current_position == 0 and active_position_key:
                position_states[active_position_key]['active'] = False
    
    # Recalculate returns with modified weights
    df_modified['fwd_returns'] = price_data['returns'].groupby(level='asset').shift(-1)
    strategy_returns_modified = df_modified.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())
    
    # Recalculate turnover
    df_modified['weights_change'] = (df_modified['weights'] - df_modified.groupby(level='asset')['weights'].shift(1)).fillna(df_modified['weights'])
    
    stop_loss_info = {
        'stop_loss_triggers': stop_loss_triggers,
        'stopped_positions': stopped_positions,
        'stop_loss_pct': stop_loss_pct
    }
    
    return df_modified, stop_loss_info, strategy_returns_modified


def run_alpha999_backtest(price_data, alpha_series, stop_loss_pct=None):
    """
    Special backtest function for alpha999 that uses ML signals directly as position sizing.
    This properly interprets ML signals as position changes and holds positions until the next signal.
    
    Args:
        price_data: DataFrame with price data
        alpha_series: Series with ML signals (-1000, 0, 1000) for alpha999
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0 for 5% loss)
        
    Returns:
        tuple: (strategy_returns, portfolio_info)
    """
    print("--------------------------------")
    print(f"Running alpha999 backtest with alpha_series: {alpha_series.name}")
    if stop_loss_pct is not None:
        print(f"🛡️ Individual position stop-loss enabled: {stop_loss_pct}%")
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
    
    # Create temporary DataFrame for stop-loss processing
    df_temp = pd.DataFrame({
        'weights': positions,
        'returns': asset_returns
    })
    
    # Apply stop-loss if enabled - now returns modified strategy returns
    stop_loss_info = {'stop_loss_triggers': 0, 'stopped_positions': []}
    if stop_loss_pct is not None:
        df_temp, stop_loss_info, strategy_returns = apply_stop_loss(df_temp, price_data, stop_loss_pct)
        positions = df_temp['weights']
    else:
        # Calculate strategy returns for baseline case
        strategy_returns_by_asset = positions * asset_returns
        strategy_returns = strategy_returns_by_asset.groupby(level='date').sum()
    
    # Calculate turnover based on position changes
    position_changes = positions.groupby(level='asset').diff().fillna(positions)
    daily_turnover = position_changes.abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'
    
    # Create portfolio info with stop-loss metadata
    portfolio_info = pd.DataFrame({
        'weights': positions,
        'turnover': daily_turnover.reindex(positions.index, level='date')
    })
    
    # Add stop-loss information
    for key, value in stop_loss_info.items():
        portfolio_info.attrs[key] = value
    
    return strategy_returns, portfolio_info


def run_rank_backtest(price_data, alpha_series, stop_loss_pct=None):
    """
    Runs a dollar-neutral, long-short backtest using rank-based weighting.
    Enhanced with optional individual position stop-loss functionality.
    
    Args:
        price_data: Price data DataFrame
        alpha_series: Alpha signal series
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0 for 5% loss)
                      None = disabled (default)
    """
    # Special handling for alpha999
    if alpha_series.name == 'alpha999' or (alpha_series.abs() > 110).any():
        print("Detected alpha999 signals - using special ML backtest")
        return run_alpha999_backtest(price_data, alpha_series, stop_loss_pct)
    
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
    # FIXED: Proper normalization that preserves negative weights for short positions
    daily_abs_rank_sum = df['centered_rank'].abs().groupby(level='date').transform('sum')
    df['weights'] = df['centered_rank'] / daily_abs_rank_sum.replace(0, 1)
    
    df['weights'] = df['weights'].fillna(0)
    
    # Apply stop-loss if enabled - now returns modified strategy returns
    stop_loss_info = {'stop_loss_triggers': 0, 'stopped_positions': []}
    if stop_loss_pct is not None:
        df, stop_loss_info, strategy_returns = apply_stop_loss(df, price_data, stop_loss_pct)
    else:
        # Calculate strategy returns for baseline case
        strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())

    # Calculate Turnover
    df['weights_change'] = (df['weights'] - df.groupby(level='asset')['weights'].shift(1)).fillna(df['weights'])
    daily_turnover = df['weights_change'].abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'

    portfolio_info = df[['weights']].copy() # We don't have 'quantile' in this backtester
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    # Add stop-loss information to portfolio_info
    for key, value in stop_loss_info.items():
        portfolio_info.attrs[key] = value
    
    if stop_loss_info['stop_loss_triggers'] > 0:
        print(f"🛡️ Stop-loss summary: {stop_loss_info['stop_loss_triggers']} positions stopped out")
    
    return strategy_returns, portfolio_info
