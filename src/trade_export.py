import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

def extract_trades_from_backtest(strategy_returns, portfolio_info, price_data, alpha_name="Strategy"):
    """
    Extract individual trades from backtest results.
    
    Args:
        strategy_returns: Daily strategy returns
        portfolio_info: DataFrame with weights, turnover data
        price_data: DataFrame with asset price and return data
        alpha_name: Name of the strategy/alpha
        
    Returns:
        pd.DataFrame: Detailed trade records
    """
    trades = []
    
    # Get asset price data
    if 'close' not in price_data.columns:
        # Calculate close prices from returns if not available
        if 'returns' in price_data.columns:
            returns = price_data['returns'].unstack(level='asset').fillna(0)
            # Assume starting price of 100 for each asset
            prices = (1 + returns).cumprod() * 100
        else:
            print("⚠️ No price data available to calculate trade P&L")
            return pd.DataFrame()
    else:
        prices = price_data['close'].unstack(level='asset')
    
    # Get position weights
    weights = portfolio_info['weights']
    
    # Process each asset separately
    for asset in weights.index.get_level_values('asset').unique():
        asset_weights = weights.xs(asset, level='asset')
        asset_prices = prices[asset] if asset in prices.columns else None
        
        if asset_prices is None:
            continue
            
        # Align weights and prices
        common_dates = asset_weights.index.intersection(asset_prices.index)
        asset_weights = asset_weights.reindex(common_dates).fillna(0)
        asset_prices = asset_prices.reindex(common_dates).fillna(method='ffill')
        
        # Track current position
        current_position = 0.0
        entry_date = None
        entry_price = None
        entry_weight = None
        
        for date in common_dates:
            new_weight = asset_weights[date]
            new_position = 1 if new_weight > 0 else (-1 if new_weight < 0 else 0)
            current_price = asset_prices[date]
            
            # Check for position changes
            if new_position != current_position:
                # Close existing position if any
                if current_position != 0 and entry_date is not None:
                    exit_date = date
                    exit_price = current_price
                    exit_weight = abs(entry_weight) if entry_weight else 0
                    
                    # Calculate trade metrics
                    days_held = (exit_date - entry_date).days
                    
                    if current_position == 1:  # Long position
                        pnl_percent = (exit_price / entry_price - 1) * 100
                        trade_type = "LONG"
                    else:  # Short position
                        pnl_percent = (entry_price / exit_price - 1) * 100
                        trade_type = "SHORT"
                    
                    # Portfolio weight impact
                    weight_impact = exit_weight * pnl_percent / 100
                    
                    trades.append({
                        'Asset': asset,
                        'Strategy': alpha_name,
                        'Trade_Type': trade_type,
                        'Entry_Date': entry_date,
                        'Exit_Date': exit_date,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Entry_Weight': entry_weight,
                        'Days_Held': days_held,
                        'PnL_Percent': pnl_percent,
                        'Weight_Impact': weight_impact,
                        'Trade_Result': 'WIN' if pnl_percent > 0 else 'LOSS'
                    })
                
                # Open new position if not flat
                if new_position != 0:
                    current_position = new_position
                    entry_date = date
                    entry_price = current_price
                    entry_weight = new_weight
                else:
                    current_position = 0
                    entry_date = None
                    entry_price = None
                    entry_weight = None
        
        # Handle any remaining open position
        if current_position != 0 and entry_date is not None:
            exit_date = common_dates[-1]
            exit_price = asset_prices[exit_date]
            
            days_held = (exit_date - entry_date).days
            
            if current_position == 1:  # Long position
                pnl_percent = (exit_price / entry_price - 1) * 100
                trade_type = "LONG"
            else:  # Short position
                pnl_percent = (entry_price / exit_price - 1) * 100
                trade_type = "SHORT"
            
            weight_impact = abs(entry_weight) * pnl_percent / 100 if entry_weight else 0
            
            trades.append({
                'Asset': asset,
                'Strategy': alpha_name,
                'Trade_Type': trade_type,
                'Entry_Date': entry_date,
                'Exit_Date': exit_date,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Entry_Weight': entry_weight,
                'Days_Held': days_held,
                'PnL_Percent': pnl_percent,
                'Weight_Impact': weight_impact,
                'Trade_Result': 'WIN' if pnl_percent > 0 else 'LOSS',
                'Status': 'OPEN'
            })
    
    if not trades:
        print("⚠️ No trades found in backtest data")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by entry date
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('Entry_Date').reset_index(drop=True)
    
    return trades_df

def calculate_trade_statistics(trades_df):
    """Calculate comprehensive trade statistics"""
    if trades_df.empty:
        return {}
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['PnL_Percent'] > 0])
    losing_trades = len(trades_df[trades_df['PnL_Percent'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['PnL_Percent'] > 0]['PnL_Percent'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['PnL_Percent'] < 0]['PnL_Percent'].mean() if losing_trades > 0 else 0
    
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else np.inf
    
    avg_days_held = trades_df['Days_Held'].mean()
    max_win = trades_df['PnL_Percent'].max()
    max_loss = trades_df['PnL_Percent'].min()
    
    # Calculate by trade type
    long_trades = trades_df[trades_df['Trade_Type'] == 'LONG']
    short_trades = trades_df[trades_df['Trade_Type'] == 'SHORT']
    
    stats = {
        'Total_Trades': total_trades,
        'Winning_Trades': winning_trades,
        'Losing_Trades': losing_trades,
        'Win_Rate_Percent': win_rate,
        'Average_Win_Percent': avg_win,
        'Average_Loss_Percent': avg_loss,
        'Profit_Factor': profit_factor,
        'Average_Days_Held': avg_days_held,
        'Best_Trade_Percent': max_win,
        'Worst_Trade_Percent': max_loss,
        'Long_Trades': len(long_trades),
        'Long_Win_Rate': (len(long_trades[long_trades['PnL_Percent'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
        'Short_Trades': len(short_trades),
        'Short_Win_Rate': (len(short_trades[short_trades['PnL_Percent'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0,
        'Total_PnL_Percent': trades_df['PnL_Percent'].sum(),
        'Total_Weight_Impact': trades_df['Weight_Impact'].sum()
    }
    
    return stats

def export_trades_to_csv(trades_df, stats, output_path, alpha_name="Strategy"):
    """
    Export trades and statistics to CSV files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Main trades file
    trades_path = output_path.with_suffix('.csv')
    trades_df.to_csv(trades_path, index=False)
    
    # Statistics file
    stats_path = output_path.with_name(f"{output_path.stem}_statistics.csv")
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(stats_path, index=False)
    
    # Winning trades
    winning_trades = trades_df[trades_df['PnL_Percent'] > 0]
    if not winning_trades.empty:
        wins_path = output_path.with_name(f"{output_path.stem}_winning_trades.csv")
        winning_trades.to_csv(wins_path, index=False)
    
    # Losing trades
    losing_trades = trades_df[trades_df['PnL_Percent'] < 0]
    if not losing_trades.empty:
        losses_path = output_path.with_name(f"{output_path.stem}_losing_trades.csv")
        losing_trades.to_csv(losses_path, index=False)
    
    print(f"📊 Trade analysis exported to: {trades_path.parent}")
    return trades_path

def export_trades_to_excel(trades_df, stats, output_path, alpha_name="Strategy"):
    """
    Export trades and statistics to a formatted Excel file.
    """
    if not EXCEL_AVAILABLE:
        print("⚠️ openpyxl not available, falling back to CSV export")
        return export_trades_to_csv(trades_df, stats, output_path, alpha_name)
    
    output_path = Path(output_path).with_suffix('.xlsx')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create Excel workbook
    wb = openpyxl.Workbook()
    
    # Remove default sheet
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])
    
    # Create Summary sheet
    ws_summary = wb.create_sheet("Summary")
    
    # Summary formatting
    header_font = Font(bold=True, size=12)
    title_font = Font(bold=True, size=14)
    
    # Add title
    ws_summary['A1'] = f"{alpha_name} - Trading Results Summary"
    ws_summary['A1'].font = title_font
    ws_summary.merge_cells('A1:B1')
    
    # Add timestamp
    ws_summary['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws_summary['A2'].font = Font(size=10, italic=True)
    
    # Add statistics
    row = 4
    for key, value in stats.items():
        ws_summary[f'A{row}'] = key.replace('_', ' ').title()
        ws_summary[f'A{row}'].font = header_font
        
        if isinstance(value, (int, float)):
            if 'Percent' in key or 'Rate' in key:
                ws_summary[f'B{row}'] = f"{value:.2f}%"
            elif 'Factor' in key:
                ws_summary[f'B{row}'] = f"{value:.2f}" if value != np.inf else "∞"
            elif 'Days' in key:
                ws_summary[f'B{row}'] = f"{value:.1f}"
            else:
                ws_summary[f'B{row}'] = int(value) if value == int(value) else f"{value:.2f}"
        else:
            ws_summary[f'B{row}'] = str(value)
        
        row += 1
    
    # Auto-adjust column widths
    for column in ws_summary.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws_summary.column_dimensions[column_letter].width = adjusted_width
    
    # Create Trades sheet
    if not trades_df.empty:
        ws_trades = wb.create_sheet("All Trades")
        
        # Add headers
        headers = list(trades_df.columns)
        for i, header in enumerate(headers, 1):
            cell = ws_trades.cell(row=1, column=i)
            cell.value = header.replace('_', ' ')
            cell.font = header_font
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
        
        # Add data
        for row_idx, (_, row) in enumerate(trades_df.iterrows(), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws_trades.cell(row=row_idx, column=col_idx)
                
                if isinstance(value, pd.Timestamp):
                    cell.value = value.strftime('%Y-%m-%d')
                elif isinstance(value, (int, float)) and not pd.isna(value):
                    if 'Percent' in headers[col_idx-1] or 'Weight' in headers[col_idx-1]:
                        cell.value = f"{value:.2f}%"
                    elif 'Price' in headers[col_idx-1]:
                        cell.value = f"${value:.2f}"
                    else:
                        cell.value = value
                else:
                    cell.value = str(value) if value is not None else ""
                
                # Color coding for P&L
                if 'PnL' in headers[col_idx-1]:
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if value > 0:
                            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                        elif value < 0:
                            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Auto-adjust column widths for trades sheet
        for column in ws_trades.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            ws_trades.column_dimensions[column_letter].width = adjusted_width
        
        # Create filtered views for wins/losses
        winning_trades = trades_df[trades_df['PnL_Percent'] > 0]
        losing_trades = trades_df[trades_df['PnL_Percent'] < 0]
        
        if not winning_trades.empty:
            ws_wins = wb.create_sheet("Winning Trades")
            # Copy headers
            for i, header in enumerate(headers, 1):
                cell = ws_wins.cell(row=1, column=i)
                cell.value = header.replace('_', ' ')
                cell.font = header_font
                cell.fill = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
                cell.font = Font(color="FFFFFF", bold=True)
            
            # Copy winning trades data
            for row_idx, (_, row) in enumerate(winning_trades.iterrows(), 2):
                for col_idx, value in enumerate(row, 1):
                    cell = ws_wins.cell(row=row_idx, column=col_idx)
                    if isinstance(value, pd.Timestamp):
                        cell.value = value.strftime('%Y-%m-%d')
                    elif isinstance(value, (int, float)) and not pd.isna(value):
                        if 'Percent' in headers[col_idx-1] or 'Weight' in headers[col_idx-1]:
                            cell.value = f"{value:.2f}%"
                        elif 'Price' in headers[col_idx-1]:
                            cell.value = f"${value:.2f}"
                        else:
                            cell.value = value
                    else:
                        cell.value = str(value) if value is not None else ""
        
        if not losing_trades.empty:
            ws_losses = wb.create_sheet("Losing Trades")
            # Copy headers
            for i, header in enumerate(headers, 1):
                cell = ws_losses.cell(row=1, column=i)
                cell.value = header.replace('_', ' ')
                cell.font = header_font
                cell.fill = PatternFill(start_color="C5504B", end_color="C5504B", fill_type="solid")
                cell.font = Font(color="FFFFFF", bold=True)
            
            # Copy losing trades data
            for row_idx, (_, row) in enumerate(losing_trades.iterrows(), 2):
                for col_idx, value in enumerate(row, 1):
                    cell = ws_losses.cell(row=row_idx, column=col_idx)
                    if isinstance(value, pd.Timestamp):
                        cell.value = value.strftime('%Y-%m-%d')
                    elif isinstance(value, (int, float)) and not pd.isna(value):
                        if 'Percent' in headers[col_idx-1] or 'Weight' in headers[col_idx-1]:
                            cell.value = f"{value:.2f}%"
                        elif 'Price' in headers[col_idx-1]:
                            cell.value = f"${value:.2f}"
                        else:
                            cell.value = value
                    else:
                        cell.value = str(value) if value is not None else ""
    
    # Save workbook
    wb.save(output_path)
    print(f"📊 Trade analysis exported to: {output_path}")
    return output_path

def export_backtest_trades(alpha_calculator, price_data, alpha_name="alpha998", output_dir="export_trades_to_csv/trade_exports", export_format="excel"):
    """
    Main function to export trades from a specific alpha's backtest.
    
    Args:
        alpha_calculator: Alpha101 instance
        price_data: Price data DataFrame
        alpha_name: Name of alpha to analyze (default: alpha999)
        output_dir: Directory to save CSV files
    
    Returns:
        Path to exported CSV file
    """
    from src.backtests import run_rank_backtest
    
    print(f"🔍 Extracting trades for {alpha_name}...")
    
    # Get alpha signals
    if not hasattr(alpha_calculator, alpha_name):
        print(f"❌ Alpha {alpha_name} not found")
        return None
    
    alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
    
    if alpha_series.empty:
        print(f"❌ No signals generated for {alpha_name}")
        return None
    
    # Run backtest
    strategy_returns, portfolio_info = run_rank_backtest(price_data, alpha_series)
    
    if strategy_returns.empty:
        print(f"❌ No returns generated for {alpha_name}")
        return None
    
    # Extract trades
    trades_df = extract_trades_from_backtest(strategy_returns, portfolio_info, price_data, alpha_name)
    
    if trades_df.empty:
        print(f"❌ No trades extracted for {alpha_name}")
        return None
    
    # Calculate statistics
    stats = calculate_trade_statistics(trades_df)
    
    # Export based on format
    output_path = Path(output_dir) / f"{alpha_name}_trades_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if export_format.lower() == "excel":
        export_path = export_trades_to_excel(trades_df, stats, output_path, alpha_name)
    else:
        export_path = export_trades_to_csv(trades_df, stats, output_path, alpha_name)
    
    # Print summary
    print(f"\n📈 {alpha_name} Trade Summary:")
    print(f"   Total Trades: {stats['Total_Trades']}")
    print(f"   Win Rate: {stats['Win_Rate_Percent']:.1f}%")
    print(f"   Profit Factor: {stats['Profit_Factor']:.2f}")
    print(f"   Average Days Held: {stats['Average_Days_Held']:.1f}")
    print(f"   Best Trade: {stats['Best_Trade_Percent']:.2f}%")
    print(f"   Worst Trade: {stats['Worst_Trade_Percent']:.2f}%")
    
    return export_path 