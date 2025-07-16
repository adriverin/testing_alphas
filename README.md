# 101 Formulaic Alphas: A Quantitative Trading Framework

This repository contains a **production-ready Python framework** for implementing, backtesting, and analyzing quantitative trading strategies. Originally based on the **"101 Formulaic Alphas"** paper by Zura Kakushadze, it is evolving into a comprehensive alpha research platform with **mid-frequency crypto capabilities**, **ML integration**, and **risk management**.





## Running summary
**ML training** 

Fits exponential moving averages, exponential moving standard deviations, and relative strength index (RSI) (based on 18.2 Strategy: Artificial neural network (ANN) of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865).


Adapt configuration form `config.py` (directly in `multi_crypto_ml_training.py`) and run:

`python multi_crypto_ml_training.py`

**Plot generation**

Then generate plot in `reports/interval_reports` showing (net) earnings vs benchmark, underwater and turnover plots. Adapt date interval, number of plots in the interval, etc. in `main.py`. Run:

`python main.py (--flags) interval`

Available (currently useful) --flags:
- `--interval 1d` or `-i 4h` default is daily data for prices (`1d`); Available intervals: `1m, 5m, 15m, 1h, 4h, 1d`.  
- `--stop-loss -5.0` or `-sl -3.5` to set a stop loss in the backtesting. (Note: I sometimes get strange results; this feature must be checked)
- `--log-scale` set a logarithmic scale in the earnings vs benchmark plot.
- `--stock-mode` switch data fetching to `yfinance` to get daily stock data prices; without this flag the default is crypto data on Binance through ccxt.


**HTML dashboard report**

Run a full report as HTML. 

`python run_dashboard.py`

Follow instructions.













## New Major Features 

### **Mid-Frequency Crypto Backtesting**
- **Minute-level precision**: Test strategies at 1m, 5m, 15m intervals
- **Binance integration**: Professional exchange data via `ccxt`
- **Massive scale**: Handle 250K+ data points seamlessly
- **Same interface**: All existing alphas work unchanged

### **Advanced Risk Management**
- **Individual position stop-loss**: Precise risk control per trade
- **Real-time P&L tracking**: Monitor position performance
- **Multiple stop-loss strategies**: Portfolio vs individual position levels

### **Comprehensive Trade Analysis**
- **Individual trade extraction**: Export every trade with full details
- **Excel/CSV exports**: Spreadsheet analysis
- **Trade statistics**: Win rate, profit factor, holding periods
- **P&L validation**: Verified against portfolio returns

### **Machine Learning Integration**
- **ML-powered alphas**: Alpha998/999 with neural network signals
- **Multi-crypto models**: Asset-specific ML strategies
- **Cross-asset validation**: Multi-crypto portfolios

## Project Architecture

The framework is organized into specialized modules for maximum flexibility:

```
testing_alphas/
├── main.py                          # Central command interface
├── src/
│   ├── alpha101.py                  # 101+ alpha implementations
│   ├── data_loader.py               # Multi-source data pipeline
│   ├── backtests.py                 # Advanced backtesting engines
│   ├── reporting.py                 # Comprehensive reporting suite
│   ├── validation.py                # Rigorous validation framework
│   ├── trade_export.py              # Professional trade analysis
│   └── combiner.py                  # Alpha combination strategies
├── export_trades_to_csv/
│   └── export_trades.py             # Standalone trade export tool
├── reports/                         # Generated analysis reports
├── artefacts/                       # ML models and signals
└── requirements.txt                 # Dependencies
```

## Core Capabilities

### **1. Multi-Source Data Pipeline**
```python
# Traditional stock data (daily)
price_data = get_stock_data(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31')

# High-frequency crypto data (minute-level)
crypto_data = get_crypto_data(['BTC-USD', 'ETH-USD'], '2024-01-01', '2024-12-31', interval='5m')
```

**Features:**
- **Yahoo Finance**: Stocks, ETFs, daily data
- **Binance Exchange**: 100+ crypto pairs, 1m-1d intervals
- **Caching**: Fast subsequent runs
- **Automatic validation**: Data quality checks

### **2. Advanced Alpha Implementation**
```python
# Classic formulaic alphas (101+ implemented)
alpha003 = alpha_calculator.alpha003()  # Price momentum
alpha054 = alpha_calculator.alpha054()  # Mean reversion

# ML-powered alphas
alpha998 = alpha_calculator.alpha998()  # Multi-crypto ML signals
alpha999 = alpha_calculator.alpha999()  # Neural network forecasts
```

**Alpha Categories:**
- **Momentum**: Trend-following strategies
- **Mean Reversion**: Contrarian strategies  
- **Cross-Sectional**: Relative value strategies
- **Machine Learning**: AI-powered signals

### **3. Backtesting Engine**
```python
# Run backtest with stop-loss
strategy_returns, portfolio_info = run_rank_backtest(
    price_data, 
    alpha_signals,
    stop_loss_pct=-5.0  # 5% individual position stop-loss
)
```

**Backtesting Features:**
- **Rank-based weighting**: Full signal information utilization
- **Dollar-neutral portfolios**: Market-neutral construction
- **Transaction cost modeling**: Realistic fee simulation
- **Position tracking**: Individual trade monitoring
- **Risk management**: Multiple stop-loss strategies

### **4. Comprehensive Analysis Suite**

#### **Command-Line Interface**
```bash
# Generate interval reports with stop-loss
python main.py interval --stop-loss -3.0

# Higher-frequency crypto analysis
python main.py combine --crypto-mode --interval 1h

# Out-of-sample validation 
python main.py oos --stop-loss -5.0

# Interactive HTML summary
python main.py summary
```

#### **Trade Export & Analysis**
```bash
# Export all trades to Excel with stop-loss
python export_trades.py alpha003 --stop-loss -2.5 --format excel

# High-frequency crypto trade analysis
python export_trades.py alpha998 --crypto-mode --interval 5m --format csv

# Batch export all alphas
python export_trades.py --all-alphas --stop-loss -3.0
```

## Report Types

### **1. Interval Reports (PDF)**
- **Per-alpha analysis** across multiple time periods
- **Performance stability** testing across market regimes
- **Risk-adjusted metrics** with stop-loss impact
- **Benchmark comparisons** with transaction costs

### **2. Interactive Summary (HTML)**
- **Heatmap visualization** of all alphas vs time periods
- **Color-coded Sharpe ratios** for quick identification
- **Hover details**: Returns, max drawdown, win rates
- **Filter capabilities**: By performance, risk, asset class

### **3. Trade Analysis (Excel/CSV)**
- **Individual trade records**: Entry/exit dates, prices, P&L
- **Trade statistics**: Win rate, profit factor, holding periods
- **Performance validation**: Trade impacts vs portfolio returns
- **Stop-loss analysis**: Risk management effectiveness

### **4. Out-of-Sample Validation** (Possibly not needed anymore)
- **Rigorous time-split validation**: In-sample discovery → OOS testing
- **Statistical significance**: Avoiding overfitting
- **Performance degradation analysis**: Real-world robustness
- **Factor attribution**: Understanding return sources

## Quick Start Guide

### **1. Installation**
```bash
# Clone repository
git clone <repository-url>
cd testing_alphas

# Install dependencies
pip install -r requirements.txt
```

### **2. Basic Stock Analysis**
```bash
# Quick performance analysis
python main.py combine

# Detailed interval analysis
python main.py interval

# Interactive summary report
python main.py summary
```

### **3. High-Frequency Crypto Analysis**
```bash
# Hourly crypto backtesting
python main.py combine --crypto-mode --interval 1h

# 5-minute precision with stop-loss
python main.py combine --crypto-mode --interval 5m --stop-loss -2.0

# Export minute-level trades
cd export_trades_to_csv
python export_trades.py alpha998 --crypto-mode --interval 15m
```

### **4. Professional Trade Analysis**
```bash
# Export comprehensive trade data
python export_trades.py alpha003 --stop-loss -3.0 --format excel

# Batch analysis of all alphas
python export_trades.py --all-alphas --format csv
```

## 🔧 Advanced Configuration

### **Data Sources**
```python
# Configure tickers in main.py
tickers = ['BTC-USD', 'ETH-USD', 'DOGE-USD']  # Crypto
tickers = ['AAPL', 'MSFT', 'GOOGL']           # Stocks

# Date ranges
start_date = '2024-01-01'
end_date = '2024-12-31'
```

### **Risk Management**
```python
# Individual position stop-loss
--stop-loss -5.0    # 5% loss limit per position

# Portfolio-level controls (in backtests.py)
daily_turnover_limit = 0.5    # 50% max daily turnover
transaction_cost = 5 / 10000  # 5 basis points
```

### **ML Model Integration**
```bash
# Train multi-crypto ML models
python multi_crypto_ml_training.py

# Generate trading signals
python src/ml_forecast_prob_dist.py
```

## Performance Examples

### **Traditional vs High-Frequency**
| Mode | Interval | Data Points | Precision | Use Case |
|------|----------|-------------|-----------|----------|
| **Stock** | 1d | 455 | Daily signals | Position trading |
| **Crypto** | 1h | 4,488 | Hourly rebalancing | Swing trading |
| **Crypto** | 5m | 53,856 | Minute precision | Scalping |
| **Crypto** | 1m | 269,280 | Ultra-high frequency | HFT strategies |

### **Real Performance Results**
```
   Alpha003 + Stop-Loss Performance:
   Total Return: 7.54% (vs 9.03% without stop-loss)
   Volatility: 47.07% (vs 48.65% without stop-loss)  
   Sharpe Ratio: 0.32 (vs 0.34 without stop-loss)
   Positions Stopped: 19 out of 156 trades
   Risk Reduction: -1.59% volatility, controlled downside
```

## Risk Management Features

### **Stop-Loss Implementation**
- **Individual position tracking**: Monitor each trade's P&L
- **Real-time risk control**: Automatic position exit on losses
- **Multiple strategies**: Portfolio (to implement) vs position-level stops
- **Performance impact analysis**: Risk vs return trade-offs

### **Trade Validation**
- **P&L reconciliation**: Trade impacts match portfolio returns
- **Price validation**: Entry/exit prices verified
- **Signal integrity**: ML signals properly interpreted
- **Statistical validation**: Significance testing

## Machine Learning Integration

### **ML Alpha Strategies**
- **Alpha998**: Multi-crypto ML signals with regime detection
- **Alpha999**: Neural network probability forecasts
- **Cross-asset models**: Asset-specific trained models
- **Signal aggregation**: Ensemble methods

### **Model Pipeline**
```python
# Train models
python multi_crypto_ml_training.py

# Generate signals  
python src/ml_forecast_prob_dist.py

# Backtest ML strategies
python main.py combine  # Uses alpha998/999 automatically
```

## Validation Framework

### **Out-of-Sample Testing** (again, possibly not needed; I mainly use interval)
```bash
# Rigorous validation with time splits
python main.py oos --stop-loss -3.0
```

**Process:**
1. **In-Sample Discovery** (2011-2020): Identify top alphas
2. **Out-of-Sample Testing** (2021-present): Validate on unseen data  
3. **Statistical Analysis**: Measure performance degradation
4. **Factor Attribution**: Understand return sources

### **Alpha Combination**
```python
# Combine top-performing alphas
core_alphas = ['alpha003', 'alpha041', 'alpha054', 'alpha083']
mega_alpha = combine_alphas(alpha_calculator, core_alphas)
```

## Research Workflow

### **1. Alpha Discovery**
```bash
# Test all alphas across multiple intervals
python main.py summary --stop-loss -2.5
```

### **2. Deep Analysis**
```bash
# Detailed performance analysis
python main.py interval --stop-loss -3.0

# Export trades for inspection
python export_trades.py alpha003 --stop-loss -3.0
```

### **3. Validation** (keep using interval)
```bash
# Out-of-sample validation
python main.py oos

# Factor analysis
python main.py factor
```

### **4. Production**
```bash
# High-frequency crypto deployment
python main.py combine --crypto-mode --interval 1h --stop-loss -2.0
```

## Future Roadmap

### **Short- and Mid-term Priorities**
- [ ] **Portfolio optimization**: Modern portfolio theory integration
- [ ] **Multi-exchange support**: Coinbase, Kraken integration
- [ ] **Real-time streaming**: Live data feeds
- [ ] **Options strategies**: Derivatives backtesting

### **Advanced Features**
- [ ] **Risk factor models**: Multi-factor risk attribution
- [ ] **Regime detection**: Market state identification
- [ ] **Alternative data**: Sentiment, news, social media
- [ ] **Cloud deployment**: AWS/GCP scalable infrastructure

### **Research Extensions**
- [ ] **Cross-market arbitrage**: Crypto vs traditional markets
- [ ] **Microstructure analysis**: Order book dynamics
- [ ] **Behavioral factors**: Investor sentiment integration
- [ ] **ESG strategies**: Sustainable investing alphas

## Dependencies

**Core Libraries:**
```
pandas>=1.5.0        # Data manipulation
numpy>=1.20.0        # Numerical computing
yfinance>=0.2.0      # Stock data
ccxt>=4.0.0          # Crypto exchange data
scipy>=1.9.0         # Statistical analysis
matplotlib>=3.5.0    # Visualization
seaborn>=0.11.0      # Statistical plotting
statsmodels>=0.13.0  # Econometric analysis
```

**Optional Libraries:**
```
openpyxl>=3.0.0      # Excel export
lxml>=4.6.0          # XML parsing
pandas-datareader    # Factor data
torch>=1.12.0        # ML models (for alpha998/999)
```

## Getting Started

### **For Beginners**
1. Start with `python main.py combine` for basic analysis
2. Try `python main.py summary` for visual overview
3. Export trades with `python export_trades.py alpha003`

### **For Advanced Users**
1. Implement custom alphas in `src/alpha101.py`
2. Use crypto mode: `--crypto-mode --interval 5m`
3. Add stop-loss strategies: `--stop-loss -3.0`
4. Conduct OOS validation: `python main.py oos`

### **For Researchers**
1. Modify tickers and date ranges in `main.py`
2. Implement new risk management in `src/backtests.py`
3. Add custom validation in `src/validation.py`
4. Extend reporting in `src/reporting.py`

---


<<<<<<< HEAD

=======
>>>>>>> 483c880 (add missing files)

