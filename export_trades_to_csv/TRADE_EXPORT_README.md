# Trade Export System

This system extracts individual trades from backtesting results and exports them to detailed spreadsheet files for analysis.

## Features

✅ **Comprehensive Trade Details**: Entry/exit dates, prices, P&L, holding periods  
✅ **Multiple Export Formats**: Excel (.xlsx) with formatting or CSV  
✅ **Detailed Statistics**: Win rate, profit factor, average holding time, etc.  
✅ **Organized Sheets**: Summary, All Trades, Winning Trades, Losing Trades  
✅ **Visual Formatting**: Color-coded P&L, professional styling  
✅ **Batch Processing**: Export trades for all alphas at once  

## Quick Start

### Export Single Alpha
```bash
# Export alpha999 trades to Excel (default)
python export_trades.py alpha999

# Export to CSV format
python export_trades.py alpha999 --format csv

# Export different alpha
python export_trades.py alpha003
```

### Export All Alphas
```bash
# Export trades for all available alphas
python export_trades.py --all-alphas

# Export all to CSV format
python export_trades.py --all-alphas --format csv
```

### View Results
```bash
# Open most recent Excel file
python view_trades.py

# Open specific alpha's trades
python view_trades.py alpha999
```

## Trade Information Included

Each trade record contains:

| Field | Description |
|-------|-------------|
| **Asset** | Ticker symbol (e.g., BTC-USD, ETH-USD) |
| **Strategy** | Alpha name (e.g., alpha999, alpha003) |
| **Trade_Type** | LONG or SHORT position |
| **Entry_Date** | When the position was opened |
| **Exit_Date** | When the position was closed |
| **Entry_Price** | Asset price at entry |
| **Exit_Price** | Asset price at exit |
| **Entry_Weight** | Portfolio weight allocated |
| **Days_Held** | Number of days position was held |
| **PnL_Percent** | Profit/Loss percentage |
| **Weight_Impact** | Contribution to portfolio return |
| **Trade_Result** | WIN or LOSS |
| **Status** | OPEN (for current positions) |

## Statistics Calculated

| Metric | Description |
|--------|-------------|
| **Total Trades** | Number of completed trades |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Ratio of gross profit to gross loss |
| **Average Win/Loss** | Mean profit/loss percentage |
| **Best/Worst Trade** | Highest gain and largest loss |
| **Average Days Held** | Mean holding period |
| **Long/Short Stats** | Separate analytics by direction |

## Excel File Structure

The Excel export creates multiple sheets:

### 📊 Summary Sheet
- Strategy performance overview
- Key statistics and metrics
- Generation timestamp

### 📈 All Trades Sheet
- Complete trade history
- Color-coded P&L (green=profit, red=loss)
- Formatted prices and percentages

### 🟢 Winning Trades Sheet
- Only profitable trades
- Green header styling
- Performance analysis

### 🔴 Losing Trades Sheet
- Only loss-making trades
- Red header styling
- Risk analysis

## Example Output

```
📈 alpha999 Trade Summary:
   Total Trades: 44
   Win Rate: 34.1%
   Profit Factor: 0.75
   Average Days Held: 20.3
   Best Trade: 57.21%
   Worst Trade: -22.85%
```

## File Organization

```
trade_exports/
├── alpha999_trades_20250704_1713.xlsx    # Excel format
├── alpha003_trades_20250704_1713.xlsx    # Another alpha
├── alpha999_trades_20250704_1710.csv     # CSV format
├── alpha999_trades_20250704_1710_statistics.csv
├── alpha999_trades_20250704_1710_winning_trades.csv
└── alpha999_trades_20250704_1710_losing_trades.csv
```

## Command Options

### `export_trades.py`
```
python export_trades.py [alpha_name] [options]

Arguments:
  alpha_name              Alpha to analyze (default: alpha999)

Options:
  --all-alphas           Export trades for all available alphas
  --format {csv,excel}   Export format (default: excel)
  --output-dir DIR       Output directory (default: trade_exports)
```

### `view_trades.py`
```
python view_trades.py [alpha_name]

Arguments:
  alpha_name              Open trades for specific alpha (optional)
```

## Advanced Usage

### Custom Data Range
To analyze different time periods, modify the date range in `export_trades.py`:

```python
# Line 43-45
tickers = ['BTC-USD', 'ETH-USD']
start_date = '2024-03-31'  # Modify this
end_date = '2025-06-30'    # Modify this
```

### Add New Assets
Add more cryptocurrencies or stocks to analyze:

```python
# Line 42
tickers = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
```

### Performance Analysis
Use the exported data for:
- Risk management analysis
- Strategy optimization
- Trade pattern identification
- Portfolio allocation studies

## Sample Trade Record

```csv
Asset,Strategy,Trade_Type,Entry_Date,Exit_Date,Entry_Price,Exit_Price,Entry_Weight,Days_Held,PnL_Percent,Weight_Impact,Trade_Result
ETH-USD,alpha999,SHORT,2025-01-12,2025-03-24,3265.95,2077.48,-1.0,71,57.21,0.57,WIN
BTC-USD,alpha999,LONG,2024-11-09,2024-11-23,76778.87,97777.28,1.0,14,27.35,0.27,WIN
```

## Troubleshooting

### No Trades Found
- Check if the alpha generates any signals
- Verify the data date range covers the alpha's active period
- Ensure the alpha has position changes (not constant neutral)

### Missing Price Data
- The system calculates prices from returns if close prices unavailable
- Assumes starting price of $100 for relative calculations
- Check data_loader.py configuration

### Excel Issues
- Install openpyxl: `pip install openpyxl`
- Falls back to CSV if Excel unavailable
- Use `--format csv` to force CSV export

## Technical Details

### Trade Detection Logic
1. **Position Changes**: Detects when portfolio weights change
2. **Entry/Exit Matching**: Pairs position opens with closes
3. **Holding Period**: Calculates days between entry and exit
4. **P&L Calculation**: Accounts for long/short directions
5. **Open Positions**: Marks unclosed trades as "OPEN"

### Alpha999 Special Handling
- Uses ML signals (-1000, 0, 1000) directly as positions
- Forward-fills signals until next change
- Supports both single and multi-asset portfolios

### Data Sources
- Price data: Yahoo Finance via yfinance
- Alpha signals: Generated from alpha101.py functions
- Backtest results: From backtests.py framework

## Integration

The trade export system integrates with:
- ✅ All existing alpha functions
- ✅ ML-based alpha999 strategy  
- ✅ Multi-asset portfolios
- ✅ Existing data loading pipeline
- ✅ Backtest framework

---

**Need Help?** Check the example usage above or run with `--help` for command-line options. 