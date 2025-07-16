import pandas as pd
import numpy as np


def run_backtest(price_data, alpha_series, quantiles=5):
    """
    From alpha values for each asset and date, this backtest forms portfolios based on top and bottom quantiles of the alpha series. 
    It then computes the strategy returns and turnover.

    It is SUPPOSED to be dollar-neutral, i.e. the sum of the long and short positions is 0.
    (this IS NOT THE CASE, unless the quantiles lead to the same number of long and short positions, which is not clear it does).

        Comment: It does not seem to work well at all. 
        (To check) Possibly due to only taking top and bottom quantiles and using 1 or -1 weights.

    Args:
        price_data: Price data DataFrame
        alpha_series: Alpha signal series
        quantiles: Number of quantiles to use (default: 5)

    Returns:
        tuple of DataFrames: (strategy_returns, portfolio_info)
        - strategy_returns: Daily returns of the strategy
        - portfolio_info: Daily portfolio weights and turnover
    """

    # Step 1: Create a Combined DataFrame with alpha and forward returns
    df = pd.DataFrame({
        'alpha': alpha_series,
        # Shift to compute returns for day t+1 (fwd_returns) with respect to day t alpha; this can lead to issues such as lookahead bias
        'fwd_returns': price_data['returns'].groupby(level='asset').shift(-1) 
    })
    df.dropna(inplace=True)
    
    # Step 2: Assign Quantiles Based on Alpha
    if not df.empty:
        # Per date, we assign quantiles to the asset based on their alpha values
        # pd.qcut() is used to assign quantiles to the alpha series
        # labels=False to get quantile order (0, 1, 2, 3, 4) instead of bins for each quantile
        # duplicates='drop' allows for less quantiles than the number set (5)
        df['quantile'] = df.groupby(level='date')['alpha'].transform(lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop'))
    else:
        return pd.Series(dtype=float), pd.DataFrame(columns=['weights', 'quantile', 'turnover'])
        
    # Step 3: Calculate Portfolio (Long/Short) Weights
    # We only use the top and bottom quantiles to form long and short positions
    # This is a simple way to form portfolios, but it is not the best way to form portfolios
    df['weights'] = 0.0
    df.loc[df['quantile'] == 0, 'weights'] = -1.0
    df.loc[df['quantile'] == (quantiles - 1), 'weights'] = 1.0
    
    # Step 4: Normalize Weights to Be Dollar-Neutral (i.e. long/short sum to 0)
    # THIS IS WRONG! does not lead to a dollar-neutral portfolio
    daily_abs_sum_weights = df.groupby(level='date')['weights'].transform(lambda x: x.abs().sum())
    df.loc[:, 'weights'] = df['weights'] / daily_abs_sum_weights.replace(0, 1)
    # Remaining NaNs are set to 0
    df['weights'] = df['weights'].fillna(0)

    # Step 5: Compute Daily Strategy Returns
    strategy_returns = df.groupby(level='date').apply(lambda x: (x['weights'] * x['fwd_returns']).sum())
    
    # Step 6: Calculate Turnover
    df['weights_change'] = (df['weights'] - df.groupby(level='asset')['weights'].shift(1)).fillna(df['weights'])
    daily_turnover = df['weights_change'].abs().groupby(level='date').sum() / 2.0
    daily_turnover.name = 'turnover'

    # This appears to me wrong because changing one asset from 1 to -1 should not change the turnover by 100% (right????)
    # TO CHECK FURTHER - Proposed fix:
    # Sum of abs position changes
    # daily_trade = df['weights_change'].abs().groupby(level='date').sum()

    # # Gross exposure per day (total invested capital)
    # gross_exposure = df['weights'].abs().groupby(level='date').sum()

    # # Turnover: total traded / total exposure
    # daily_turnover = (daily_trade / gross_exposure).fillna(0)
    # END PROPOSED FIX

    # Step 7: Package Portfolio Info
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
    ml_signals = alpha_series / 1000.0  
    
    # Get the underlying asset returns for holding period calculation
    asset_returns = price_data['returns']
    
    # Interpret ML signals as position targets that should be held
    # When we get a non-zero signal, hold that position until next non-zero signal
    positions = pd.Series(0.0, index=ml_signals.index)
    
    # create a DataFrame with (date, asset, signal) the signal to next time only changes when the signal changes
    # Example: signal = [-1.0, 0.0, 1.0] -> [-1.0, -1.0, 1.0]
    for asset in ml_signals.index.get_level_values('asset').unique():
        asset_signals = ml_signals.xs(asset, level='asset')
        current_position = 0.0
        
        for date, signal in asset_signals.items():
            if signal != 0.0:
                # Non-zero signal: change position
                current_position = signal
            # Always set the current position (whether it changed or not) 
            # i.e. if the signal is 0.0, the position is the same as the previous signal
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
    Runs a long-short backtest using rank-based weighting.
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
    # ========================= 
    # TO CHECK: THIS IS NOT DOLLAR-NEUTRAL FOR THE SIMPLE BACKTEST run_backtest()
    # ========================= 
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


# =============================
# True dollar-neutral backtest
# =============================
def run_rank_dollar_neutral_backtest(price_data, alpha_series, stop_loss_pct=None, weight_threshold=None):
    """
    Runs a truly dollar-neutral, long-short backtest using rank-based weighting.
    Ensures that the sum of weights equals zero on each date (dollar-neutral).
    Enhanced with optional individual position stop-loss functionality and weight filtering.
    
    Args:
        price_data: Price data DataFrame
        alpha_series: Alpha signal series
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0 for 5% loss)
                      None = disabled (default)
        weight_threshold: Optional minimum absolute weight threshold (e.g., 0.01)
                         Weights with absolute value below this threshold are set to 0
                         None = no threshold filtering (default)
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
    
    # 3. Create truly dollar-neutral weights
    # Method: Use centered ranks as base weights, then adjust to ensure sum = 0
    def make_dollar_neutral(group):
        """Ensure the group of weights sums to exactly zero."""
        weights = group['centered_rank'].copy()
        
        # If all weights are the same (edge case), return zeros
        if weights.std() == 0:
            return pd.Series(0.0, index=weights.index)
        
        # Adjust weights to sum to zero while preserving relative magnitudes
        # Method: subtract the mean from each weight
        weights_adjusted = weights - weights.mean()
        
        # Normalize to achieve desired leverage (sum of absolute weights)
        # Target: sum of absolute weights = 1.0 (unit leverage)
        total_abs_weight = weights_adjusted.abs().sum()
        if total_abs_weight > 0:
            weights_adjusted = weights_adjusted / total_abs_weight
        
        return weights_adjusted
    
    # Apply dollar-neutral transformation to each date
    df['weights'] = df.groupby(level='date').apply(make_dollar_neutral).values
    df['weights'] = df['weights'].fillna(0)
    
    # Apply weight threshold filtering if specified
    if weight_threshold is not None:
        print(f"🎯 Applying weight threshold filter: |weight| < {weight_threshold} → 0")
        
        # Count positions before filtering
        positions_before = (df['weights'] != 0).sum()
        
        # Apply threshold filter: set small weights to zero
        small_weights_mask = df['weights'].abs() < weight_threshold
        df.loc[small_weights_mask, 'weights'] = 0.0
        
        # Count positions after filtering
        positions_after = (df['weights'] != 0).sum()
        positions_filtered = positions_before - positions_after
        
        print(f"📊 Positions filtered: {positions_filtered} out of {positions_before} ({positions_filtered/positions_before*100:.1f}%)")
        
        # Re-normalize to maintain dollar neutrality after filtering
        def renormalize_dollar_neutral(group):
            """Re-normalize weights to maintain dollar neutrality after filtering."""
            weights = group['weights'].copy()
            
            # Skip if all weights are zero or only one non-zero weight
            non_zero_weights = weights[weights != 0]
            if len(non_zero_weights) <= 1:
                return weights
            
            # Adjust to sum to zero while preserving relative magnitudes
            weights_mean = weights.mean()
            weights_adjusted = weights - weights_mean
            
            # Only adjust non-zero weights to maintain the threshold filtering
            non_zero_mask = weights != 0
            if non_zero_mask.sum() > 0:
                # Normalize only the non-zero weights to maintain unit leverage
                total_abs_weight = weights_adjusted[non_zero_mask].abs().sum()
                if total_abs_weight > 0:
                    weights_adjusted[non_zero_mask] = weights_adjusted[non_zero_mask] / total_abs_weight
            
            return weights_adjusted
        
        # Re-apply dollar-neutral normalization after filtering
        df['weights'] = df.groupby(level='date').apply(renormalize_dollar_neutral).values
    
    # Verify dollar neutrality (for debugging)
    daily_weight_sums = df.groupby(level='date')['weights'].sum()
    max_sum_deviation = daily_weight_sums.abs().max()
    if max_sum_deviation > 1e-10:  # Allow for floating point precision
        print(f"⚠️ Warning: Maximum daily weight sum deviation from zero: {max_sum_deviation:.2e}")
    else:
        print("✅ Dollar neutrality verified: All daily weight sums ≈ 0")
    
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

    portfolio_info = df[['weights']].copy()
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    # Add stop-loss information to portfolio_info
    for key, value in stop_loss_info.items():
        portfolio_info.attrs[key] = value
    
    if stop_loss_info['stop_loss_triggers'] > 0:
        print(f"🛡️ Stop-loss summary: {stop_loss_info['stop_loss_triggers']} positions stopped out")
    
    # Verify final dollar neutrality
    final_daily_sums = portfolio_info['weights'].groupby(level='date').sum()
    print(f"📊 Final verification - Daily weight sums range: [{final_daily_sums.min():.6f}, {final_daily_sums.max():.6f}]")

    return strategy_returns, portfolio_info


def run_alpha_value_dollar_neutral_backtest(price_data, alpha_series, stop_loss_pct=None, weight_threshold=None):
    """
    Runs a dollar-neutral backtest using raw alpha values (not ranks) for weighting.
    This preserves the magnitude differences between alpha signals - stronger alphas get proportionally larger weights.
    
    Example: 
    - Alpha 0.9 vs Alpha 0.1 → weights will reflect this 9:1 ratio
    - Rank-based would treat them as just "high" vs "low" (1 vs 0)
    
    Args:
        price_data: Price data DataFrame
        alpha_series: Alpha signal series (raw values, not ranks)
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0 for 5% loss)
                      None = disabled (default)
        weight_threshold: Optional minimum absolute weight threshold (e.g., 0.01)
                         Weights with absolute value below this threshold are set to 0
                         None = no threshold filtering (default)
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

    # Use raw alpha values directly for weighting (preserving magnitude differences)
    def create_alpha_value_weights(group):
        """Create weights based on raw alpha values, ensuring dollar neutrality."""
        alphas = group['alpha'].copy()
        
        # Handle edge cases
        if len(alphas) == 0 or alphas.std() == 0:
            return pd.Series(0.0, index=alphas.index)
        
        # Method 1: Center alphas around their mean to create dollar-neutral base
        # This preserves relative magnitudes while ensuring sum = 0
        centered_alphas = alphas - alphas.mean()
        
        # Method 2: Scale to desired range [-1, 1] while preserving ratios
        max_abs_alpha = centered_alphas.abs().max()
        if max_abs_alpha > 0:
            # Scale so the largest absolute weight is 1.0
            scaled_weights = centered_alphas / max_abs_alpha
        else:
            scaled_weights = centered_alphas
        
        # Method 3: Normalize to unit leverage (sum of absolute weights = 1.0)
        total_abs_weight = scaled_weights.abs().sum()
        if total_abs_weight > 0:
            normalized_weights = scaled_weights / total_abs_weight
        else:
            normalized_weights = scaled_weights
            
        return normalized_weights
    
    # Apply alpha-value-based weighting to each date
    df['weights'] = df.groupby(level='date').apply(create_alpha_value_weights).values
    df['weights'] = df['weights'].fillna(0)
    
    # Apply weight threshold filtering if specified
    if weight_threshold is not None:
        print(f"🎯 Applying weight threshold filter: |weight| < {weight_threshold} → 0")
        
        # Count positions before filtering
        positions_before = (df['weights'] != 0).sum()
        
        # Apply threshold filter: set small weights to zero
        small_weights_mask = df['weights'].abs() < weight_threshold
        df.loc[small_weights_mask, 'weights'] = 0.0
        
        # Count positions after filtering
        positions_after = (df['weights'] != 0).sum()
        positions_filtered = positions_before - positions_after
        
        print(f"📊 Positions filtered: {positions_filtered} out of {positions_before} ({positions_filtered/positions_before*100:.1f}%)")
        
        # Re-normalize to maintain dollar neutrality after filtering
        def renormalize_alpha_weights(group):
            """Re-normalize weights to maintain dollar neutrality after filtering."""
            weights = group['weights'].copy()
            
            # Skip if all weights are zero or only one non-zero weight
            non_zero_weights = weights[weights != 0]
            if len(non_zero_weights) <= 1:
                return weights
            
            # Adjust to sum to zero while preserving relative magnitudes
            weights_mean = weights.mean()
            weights_adjusted = weights - weights_mean
            
            # Only adjust non-zero weights to maintain the threshold filtering
            non_zero_mask = weights != 0
            if non_zero_mask.sum() > 0:
                # Normalize only the non-zero weights to maintain unit leverage
                total_abs_weight = weights_adjusted[non_zero_mask].abs().sum()
                if total_abs_weight > 0:
                    weights_adjusted[non_zero_mask] = weights_adjusted[non_zero_mask] / total_abs_weight
            
            return weights_adjusted
        
        # Re-apply dollar-neutral normalization after filtering
        df['weights'] = df.groupby(level='date').apply(renormalize_alpha_weights).values
    
    # Verify dollar neutrality and weight range
    daily_weight_sums = df.groupby(level='date')['weights'].sum()
    max_sum_deviation = daily_weight_sums.abs().max()
    weight_range = [df['weights'].min(), df['weights'].max()]
    
    print(f"✅ Alpha-value weighting complete:")
    print(f"   📊 Weight range: [{weight_range[0]:.4f}, {weight_range[1]:.4f}]")
    print(f"   💰 Dollar neutrality: max deviation = {max_sum_deviation:.2e}")
    
    if max_sum_deviation > 1e-10:
        print(f"   ⚠️ Warning: Daily weight sums deviate from zero")
    
    # Apply stop-loss if enabled
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

    portfolio_info = df[['weights']].copy()
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    # Add stop-loss information to portfolio_info
    for key, value in stop_loss_info.items():
        portfolio_info.attrs[key] = value
    
    if stop_loss_info['stop_loss_triggers'] > 0:
        print(f"🛡️ Stop-loss summary: {stop_loss_info['stop_loss_triggers']} positions stopped out")
    
    # Final verification
    final_daily_sums = portfolio_info['weights'].groupby(level='date').sum()
    print(f"📊 Final verification - Daily weight sums range: [{final_daily_sums.min():.6f}, {final_daily_sums.max():.6f}]")

    return strategy_returns, portfolio_info




def run_alpha_value_backtest(price_data, alpha_series, stop_loss_pct=None, weight_threshold=None):
    """
    Runs a backtest using raw alpha values (not ranks) for weighting WITHOUT dollar neutrality.
    This preserves the magnitude differences between alpha signals and allows net long/short exposure.
    
    Unlike the dollar-neutral version, this function:
    - Does NOT force the sum of weights to equal zero
    - Allows net long exposure when alphas are mostly positive
    - Allows net short exposure when alphas are mostly negative
    - Preserves the original alpha signal direction and magnitude
    
    Example: 
    - If all alphas are positive → net long portfolio
    - If alphas are [0.9, 0.1, -0.2] → mostly long with small short position
    - Alpha 0.9 vs Alpha 0.1 → weights will reflect this 9:1 ratio
    
    Args:
        price_data: Price data DataFrame
        alpha_series: Alpha signal series (raw values, not ranks)
        stop_loss_pct: Optional stop-loss percentage (e.g., -5.0 for 5% loss)
                      None = disabled (default)
        weight_threshold: Optional minimum absolute weight threshold (e.g., 0.01)
                         Weights with absolute value below this threshold are set to 0
                         None = no threshold filtering (default)
    """
    # Special handling for alpha999
    if alpha_series.name == 'alpha999' or (alpha_series.abs() > 500).any():
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

    # Use raw alpha values directly for weighting (NO dollar neutrality constraint)
    def create_alpha_weights_no_neutrality(group):
        """Create weights based on raw alpha values, preserving net exposure."""
        alphas = group['alpha'].copy()
        
        # Handle edge cases
        if len(alphas) == 0:
            return pd.Series(0.0, index=alphas.index)
        
        # If all alphas are the same, distribute equally
        if alphas.std() == 0:
            if alphas.iloc[0] == 0:
                return pd.Series(0.0, index=alphas.index)
            else:
                # All same non-zero value - equal weights with same sign
                equal_weight = 1.0 / len(alphas)
                return pd.Series(equal_weight * np.sign(alphas.iloc[0]), index=alphas.index)
        
        # Method 1: Scale alphas to desired range [-1, 1] while preserving signs and ratios
        max_abs_alpha = alphas.abs().max()
        if max_abs_alpha > 0:
            # Scale so the largest absolute weight is 1.0
            scaled_weights = alphas / max_abs_alpha
        else:
            scaled_weights = alphas
        
        # Method 2: Normalize to unit leverage (sum of absolute weights = 1.0)
        # This maintains relative magnitudes and signs while standardizing total exposure
        total_abs_weight = scaled_weights.abs().sum()
        if total_abs_weight > 0:
            normalized_weights = scaled_weights / total_abs_weight
        else:
            normalized_weights = scaled_weights
            
        return normalized_weights
    
    # Apply alpha-value-based weighting to each date (NO centering around mean)
    df['weights'] = df.groupby(level='date').apply(create_alpha_weights_no_neutrality).values
    df['weights'] = df['weights'].fillna(0)
    
    # Apply weight threshold filtering if specified
    if weight_threshold is not None:
        print(f"🎯 Applying weight threshold filter: |weight| < {weight_threshold} → 0")
        
        # Count positions before filtering
        positions_before = (df['weights'] != 0).sum()
        
        # Apply threshold filter: set small weights to zero
        small_weights_mask = df['weights'].abs() < weight_threshold
        df.loc[small_weights_mask, 'weights'] = 0.0
        
        # Count positions after filtering
        positions_after = (df['weights'] != 0).sum()
        positions_filtered = positions_before - positions_after
        
        print(f"📊 Positions filtered: {positions_filtered} out of {positions_before} ({positions_filtered/positions_before*100:.1f}%)")
        
        # Re-normalize to maintain unit leverage after filtering (but NOT dollar neutrality)
        def renormalize_weights_no_neutrality(group):
            """Re-normalize weights to maintain unit leverage after filtering."""
            weights = group['weights'].copy()
            
            # Skip if all weights are zero
            non_zero_weights = weights[weights != 0]
            if len(non_zero_weights) == 0:
                return weights
            
            # Simply renormalize the absolute sum to 1.0 without changing net exposure
            total_abs_weight = weights.abs().sum()
            if total_abs_weight > 0:
                weights = weights / total_abs_weight
            
            return weights
        
        # Re-apply normalization after filtering (maintaining net exposure)
        df['weights'] = df.groupby(level='date').apply(renormalize_weights_no_neutrality).values
    
    # Calculate and report net exposure statistics
    daily_weight_sums = df.groupby(level='date')['weights'].sum()
    daily_gross_exposure = df.groupby(level='date')['weights'].apply(lambda x: x.abs().sum())
    weight_range = [df['weights'].min(), df['weights'].max()]
    net_exposure_range = [daily_weight_sums.min(), daily_weight_sums.max()]
    
    print(f"✅ Alpha-value weighting (no neutrality) complete:")
    print(f"   📊 Weight range: [{weight_range[0]:.4f}, {weight_range[1]:.4f}]")
    print(f"   📈 Net exposure range: [{net_exposure_range[0]:.4f}, {net_exposure_range[1]:.4f}]")
    print(f"   💼 Average gross exposure: {daily_gross_exposure.mean():.4f}")
    print(f"   🎯 Average net exposure: {daily_weight_sums.mean():.4f}")
    
    # Apply stop-loss if enabled
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

    portfolio_info = df[['weights']].copy()
    portfolio_info = portfolio_info.join(daily_turnover, on='date')
    
    # Add exposure metrics to portfolio info
    portfolio_info['net_exposure'] = daily_weight_sums.reindex(portfolio_info.index, level='date')
    portfolio_info['gross_exposure'] = daily_gross_exposure.reindex(portfolio_info.index, level='date')
    
    # Add stop-loss information to portfolio_info
    for key, value in stop_loss_info.items():
        portfolio_info.attrs[key] = value
    
    if stop_loss_info['stop_loss_triggers'] > 0:
        print(f"🛡️ Stop-loss summary: {stop_loss_info['stop_loss_triggers']} positions stopped out")
    
    # Final verification
    final_net_exposure = portfolio_info['net_exposure'].dropna()
    if len(final_net_exposure) > 0:
        print(f"📊 Final net exposure range: [{final_net_exposure.min():.6f}, {final_net_exposure.max():.6f}]")

    return strategy_returns, portfolio_info
