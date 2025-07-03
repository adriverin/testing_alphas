# Complete Guide: ML-Based Alpha999 Trading Strategy

This guide explains how to use the ML forecasting system (`ml_forecast_prob_dist.py`) with the existing alpha trading infrastructure to create and deploy the alpha999 ML-based trading strategy.

## Overview

The alpha999 system combines machine learning price forecasting with the existing alpha research framework. It follows this workflow:

1. **Train ML Model**: Use `ml_forecast_prob_dist.py` to train a neural network that predicts future normalized returns
2. **Generate Signals**: The model creates trading signals (-1, 0, 1) based on prediction confidence
3. **Deploy Strategy**: The `alpha999()` function in `alpha101.py` loads these signals for backtesting
4. **Analyze Performance**: Use `main.py` to run comprehensive performance analysis

## Step 1: Understanding the ML Forecasting System

### Core Architecture

```python
# Neural Network: Input → Hidden Layers → Quantile Classification → Trading Signals
Features (17) → [128, 64, 32] → 4 Quantiles → {-1, 0, 1} Signals
```

### Signal Generation Logic

The ML model works by:
1. **Classification**: Predicts which quantile future returns will fall into (4 bins)
2. **Confidence Scoring**: Combines extreme quantile probabilities
3. **Thresholding**: Uses percentile-based cutoffs to generate discrete signals

```python
# Signal generation (simplified)
bottom_scores = P[:, 0] + P[:, 1]  # Bottom 2 quantiles (bearish)
top_scores = P[:, -2] + P[:, -1]   # Top 2 quantiles (bullish)
extreme_preference = top_scores - bottom_scores

# Top/bottom 5% become signals
if score > 95th_percentile: signal = 1   # Long
elif score < 5th_percentile: signal = -1 # Short
else: signal = 0                         # Neutral
```

## Step 2: Configuration Parameters Guide

### Core Data Parameters

```python
@dataclass
class Config:
    # Data source configuration
    symbol: str = "BTC-USD"           # Trading pair to analyze
    start: str = "2020-01-01"         # Training data start date
    end: str = "2025-01-01"           # Training data end date
    interval: str = "1d"              # Bar interval: "1h", "4h", "1d"
    
    # Prediction parameters
    forecast_horizon_hours: int = 24  # How far ahead to predict (hours)
    vol_window_hours: int = 240       # Volatility estimation window
```

**How to Choose:**
- **Symbol**: Use liquid crypto pairs (`BTC-USD`, `ETH-USD`) or major stocks
- **Date Range**: Minimum 2-3 years for stable training (current: 5 years)
- **Interval**: Daily (`1d`) recommended for stability; hourly for more data
- **Forecast Horizon**: 24 hours (1 day) balances predictability vs reaction time

### Feature Engineering Parameters

```python
    # Moving average windows (in hours, converted to bars)
    ema_windows = (24, 48, 72, 120, 168)  # 1d, 2d, 3d, 5d, 7d
    rsi_windows = (48, 96, 144, 192)      # 2d, 4d, 6d, 8d
```

**Current Features (17 total):**
- Simple Moving Averages (5): 2, 5, 10, 20, 30 day
- Volatility features (3): 5, 10, 20 day rolling volatility  
- Momentum features (4): 1, 3, 7, 14 day returns
- RSI indicators (3): 7, 14, 21 day RSI
- Base features (2): normalized returns, volatility

**How to Choose:**
- **Shorter windows**: More reactive, higher noise
- **Longer windows**: More stable, slower to adapt
- **Current setup**: Balanced for daily trading with 1-4 week patterns

### Model Architecture Parameters

```python
    # Model configuration
    n_quantiles: int = 4              # Number of return bins
    hidden_sizes: tuple = (128, 64, 32)  # Neural network layers
    n_epochs: int = 50                # Training iterations
    lr: float = 5e-5                  # Learning rate
    batch_size: int = 256             # Training batch size
```

**How to Choose:**
- **n_quantiles**: 4-5 optimal (too few = crude, too many = overfitting)
- **hidden_sizes**: Current setup good for ~2000 samples; scale with data size
- **n_epochs**: 50 with early stopping prevents overfitting
- **lr**: 5e-5 conservative for stable training; increase for faster convergence

### Signal Generation Parameters

```python
    # Signal parameters
    test_fraction: float = 0.20       # Holdout for validation (20%)
    threshold: float = 0.4            # Legacy parameter (not used in current logic)
```

**How to Choose:**
- **test_fraction**: 0.15-0.25 provides sufficient validation without losing training data
- **threshold**: Not used in percentile-based approach; kept for compatibility

## Step 3: Using the ML System

### Basic Usage

```bash
# Step 1: Train model and generate signals
cd /path/to/testing_alphas
python src/ml_forecast_prob_dist.py

# Step 2: Run backtests with alpha999
python main.py interval     # Detailed performance analysis
python main.py summary      # Quick overview across time periods
```

### Advanced Configuration

```python
# Custom configuration example
from src.ml_forecast_prob_dist import Config, main

# High-frequency setup
config_hourly = Config(
    symbol="BTC-USD",
    interval="1h",
    forecast_horizon_hours=4,
    n_epochs=100,
    hidden_sizes=(256, 128, 64)
)

# Conservative long-term setup  
config_daily = Config(
    symbol="BTC-USD", 
    interval="1d",
    forecast_horizon_hours=168,  # 1 week
    vol_window_hours=720,        # 30 days
    n_quantiles=3               # Simpler classification
)

# Run with custom config
signals = main(config_hourly)
```

## Step 4: Understanding Alpha999 Integration

### How Alpha999 Works

The `alpha999()` function in `alpha101.py` serves as the bridge between ML predictions and the trading framework:

```python
def alpha999(self):
    # 1. Load pre-computed ML signals from artifacts
    signals_df = pd.read_parquet("artefacts/trading_signals_threshold_40.parquet")
    
    # 2. Align signals with portfolio dates/assets
    # Handles timezone matching and forward-filling
    
    # 3. Amplify signals for detection (×1000)
    signal_value = signal_value * 1000.0  # -1→-1000, 0→0, 1→1000
    
    # 4. Return properly formatted series
    return result  # MultiIndex series with amplified signals
```

### Signal Flow

```
ML Model → Parquet File → Alpha999 → Backtesting → Performance Analysis
   ↓            ↓            ↓           ↓              ↓
{-1,0,1}   signals.parquet  {-1000,0,1000}  Position   Reports
```

### Key Features

1. **Automatic Detection**: Amplified signals trigger custom backtesting logic
2. **Position Holding**: Signals held until next non-zero signal (not daily decisions)
3. **Multi-Asset Support**: Same signal applied to all assets in portfolio
4. **Fallback Safety**: Returns neutral signals if ML data unavailable

## Step 5: Performance Optimization

### Data Quality

```python
# Monitor these metrics during training
print(f"Data cleaned: {initial_len} -> {final_len} rows")  # Should keep >90%
print(f"Features shape: {out[feature_cols].shape}")       # Verify feature count
print(f"Test acc: {acc:.2%}")                             # >25% for 4-class
```

### Signal Quality

```python
# Check signal distribution after generation
signal_counts = signals.value_counts()
print(f"Short (-1): {signal_counts.get(-1, 0)/len(signals)*100:.1f}%")   # Target: 3-7%
print(f"Neutral(0): {signal_counts.get(0, 0)/len(signals)*100:.1f}%")    # Target: 85-95%  
print(f"Long (+1):  {signal_counts.get(1, 0)/len(signals)*100:.1f}%")    # Target: 3-7%
```

### Performance Tuning

**For Better Signals:**
- Increase training data (longer date range)
- Add more relevant features (order book, macro indicators)
- Experiment with different forecast horizons
- Try ensemble methods (multiple models)

**For Better Backtests:**
- Reduce transaction costs if model trades frequently
- Adjust position sizing based on confidence scores
- Implement stop-losses for risk management
- Add regime detection (bull/bear market adaptation)

## Step 6: File Structure and Artifacts

### Key Files

```
testing_alphas/
├── src/
│   ├── ml_forecast_prob_dist.py    # ML model training and signal generation
│   ├── alpha101.py                 # Alpha999 integration function
│   ├── backtests.py               # Custom alpha999 backtesting logic
│   └── ...                        # Other alpha infrastructure
├── artefacts/                     # Generated ML artifacts
│   ├── return_model.pt            # Trained PyTorch model
│   ├── trading_signals_threshold_40.parquet  # Generated signals
│   └── reliability_test.png       # Model calibration plots
├── main.py                        # Main analysis runner
└── ML_ALPHA999_GUIDE.md          # This guide
```

### Artifact Files

**return_model.pt**: Contains the trained model state and metadata
```python
{
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'quantile_edges': q,
    'feature_names': X.columns.tolist(),
    'input_dim': X.shape[1]
}
```

**trading_signals_threshold_40.parquet**: Daily trading signals
```python
# DataFrame with DatetimeIndex and 'signal' column
#                   signal
# 2020-01-04           0
# 2020-01-05          -1
# 2020-01-06           0
# ...
```

## Step 7: Troubleshooting

### Common Issues

**1. No ML signals file found**
```bash
# Solution: Generate signals first
python src/ml_forecast_prob_dist.py
```

**2. Training divergence**
```
RuntimeError: Training diverged.
```
```python
# Solution: Reduce learning rate
cfg.lr = 1e-5  # Instead of 5e-5
```

**3. Poor model accuracy (<20%)**
- Check for data leakage
- Verify feature engineering
- Increase model complexity or training data

**4. Zero turnover in backtests**
- Confirm alpha999 detection is working
- Check signal amplification (should see ±1000 values)
- Verify custom backtest function is called

**5. CCXT connection errors**
```python
# Solution: Check internet connection and API limits
# The system caches data, so this only affects initial downloads
```

### Monitoring Checklist

✅ **Data Pipeline**
- [ ] Price data loads successfully
- [ ] Features have reasonable distributions  
- [ ] No excessive NaN values after cleaning

✅ **Model Training**
- [ ] Training converges without divergence
- [ ] Validation accuracy > random (25% for 4-class)
- [ ] No obvious overfitting (train/val gap <10%)

✅ **Signal Generation**
- [ ] Signal distribution is reasonable (5-10% non-neutral)
- [ ] Signals are properly saved to parquet file
- [ ] Date alignment works correctly

✅ **Backtesting Integration**
- [ ] Alpha999 detection triggers ("Detected alpha999 signals")
- [ ] Custom backtest function runs
- [ ] Position holding works (no step functions)
- [ ] Turnover is reasonable (5-20% average)

## Step 8: Example Workflows

### Basic Workflow

```bash
# Complete workflow example
cd testing_alphas

# 1. Generate ML signals (one-time setup)
python src/ml_forecast_prob_dist.py

# 2. Run comprehensive analysis  
python main.py interval    # Detailed PDF reports
python main.py summary     # HTML overview

# 3. Validate performance
python main.py oos         # Out-of-sample testing

# 4. Compare with other strategies
python main.py combine     # Multi-alpha comparison
```

### Advanced Analysis

```bash
# Generate multiple threshold signals
python -c "
from src.ml_forecast_prob_dist import Config, main
import os

# Test different confidence thresholds
for threshold in [0.3, 0.4, 0.5, 0.6]:
    cfg = Config(threshold=threshold)
    main(cfg)
    # This creates trading_signals_threshold_XX.parquet files
"

# Compare different models
python main.py interval     # Test all generated models
```

### Custom Model Development

```python
# Example: Create a custom model for different market conditions
from src.ml_forecast_prob_dist import Config, main

# Bull market model (shorter horizons)
bull_config = Config(
    symbol="BTC-USD",
    start="2020-01-01",
    end="2021-12-31",  # Bull market period
    forecast_horizon_hours=12,
    n_quantiles=5,
    cache_dir=Path("artefacts/bull_market")
)

# Bear market model (longer horizons, defensive)
bear_config = Config(
    symbol="BTC-USD", 
    start="2022-01-01",
    end="2023-12-31",  # Bear market period
    forecast_horizon_hours=48,
    n_quantiles=3,
    cache_dir=Path("artefacts/bear_market")
)

# Train both models
bull_signals = main(bull_config)
bear_signals = main(bear_config)
```

## Step 9: Performance Interpretation

### Key Metrics to Monitor

**Training Metrics:**
- **Accuracy**: >25% for 4-class classification (random = 25%)
- **Loss Convergence**: Should decrease steadily, plateau around epoch 30-40
- **Validation Gap**: <10% difference between train/validation accuracy

**Signal Metrics:**
- **Signal Frequency**: 5-15% non-neutral signals optimal
- **Signal Persistence**: Average hold period 5-20 days
- **Signal Balance**: Long/short ratio between 0.5-2.0

**Backtest Metrics:**
- **Information Ratio**: >0.5 good, >1.0 excellent
- **Sharpe Ratio**: >1.0 good for crypto, >1.5 excellent
- **Max Drawdown**: <20% acceptable, <10% good
- **Turnover**: 5-25% average, depends on strategy frequency

### Interpreting Results

**Good Alpha999 Performance:**
```
Strategy Performance Metrics:
- Information Ratio: 1.2
- Sharpe Ratio: 1.8
- Max Drawdown: -12%
- Average Turnover: 15%
- Active Days: 45/252 (18%)
```

**Warning Signs:**
```
Strategy Performance Metrics:
- Information Ratio: 0.1  ← Too low
- Sharpe Ratio: 0.3       ← Too low  
- Max Drawdown: -35%      ← Too high
- Average Turnover: 85%   ← Too high (overtrading)
- Active Days: 200/252    ← Too frequent
```

## Step 10: Next Steps and Extensions

### Model Improvements

1. **Feature Engineering**:
   - Add macro-economic indicators
   - Include sentiment data (fear/greed index)
   - Cross-asset correlations
   - Order book features

2. **Architecture Enhancements**:
   - LSTM/GRU for sequence modeling
   - Attention mechanisms
   - Ensemble methods
   - Transfer learning from other assets

3. **Signal Processing**:
   - Dynamic position sizing based on confidence
   - Regime-aware predictions
   - Multi-timeframe signals
   - Risk-adjusted signal weighting

### Integration Enhancements

1. **Risk Management**:
   - Stop-loss integration
   - Position sizing rules
   - Correlation-based portfolio construction
   - Dynamic hedging

2. **Production Features**:
   - Real-time data feeds
   - Model retraining schedules
   - Performance monitoring alerts
   - A/B testing framework

### Research Extensions

1. **Multi-Asset Models**:
   - Cross-asset prediction
   - Portfolio-level optimization
   - Sector rotation strategies

2. **Alternative Architectures**:
   - Reinforcement learning
   - Graph neural networks
   - Transformer models
   - Generative approaches

This comprehensive guide provides everything needed to understand, deploy, and extend the ML-based alpha999 trading strategy within the existing infrastructure. 