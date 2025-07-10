# ML Forecasting Module Migration Guide

This guide helps you transition from the original ML files to the new centralized `ml_forecasting` module.

## Overview

The following files have been consolidated into a unified module:

### Original Files ➜ New Module Structure

```
src/ml_forecast_prob_dist.py     ➜ src/ml_forecasting/
src/ml_forecast_improved.py      ➜   ├── config.py
multi_crypto_ml_training.py      ➜   ├── data_loader.py
                                 ➜   ├── feature_engineering.py
                                 ➜   ├── models.py
                                 ➜   ├── training.py
                                 ➜   ├── evaluation.py
                                 ➜   ├── signal_generation.py
                                 ➜   ├── multi_asset.py
                                 ➜   └── __init__.py
```

## Key Benefits

✅ **Reduced Code Duplication** - Single source of truth for each functionality  
✅ **Unified Configuration** - One config class instead of overlapping classes  
✅ **Better Maintainability** - Changes in one place affect all usage  
✅ **Enhanced Modularity** - Clear separation of concerns  
✅ **Backward Compatibility** - Original class names still work  

## Migration Steps

### 1. Update Imports

**Before:**
```python
from src.ml_forecast_improved import run_improved_training, ImprovedConfig
from src.ml_forecast_prob_dist import Config, main
```

**After:**
```python
from src.ml_forecasting import MLConfig, train_model
# Or for backward compatibility:
from src.ml_forecasting import MLConfig as ImprovedConfig
```

### 2. Configuration Changes

**Before:**
```python
# Original Config
config = Config(
    symbol="BTC-USD",
    start="2020-01-01",
    end="2024-01-01",
    n_quantiles=5,
    threshold=0.4
)

# Original ImprovedConfig
improved_config = ImprovedConfig(
    symbol="BTC-USD", 
    min_train_samples=500,
    enable_regime_features=True
)
```

**After:**
```python
# Unified MLConfig
config = MLConfig(
    symbol="BTC-USD",
    start="2020-01-01", 
    end="2024-01-01",
    training_mode="simple",  # or "improved"
    n_quantiles=5,
    threshold=0.4,
    min_train_samples=500,
    enable_regime_features=True
)

# Or use factory methods:
simple_config = MLConfig.for_simple_training(symbol="BTC-USD")
improved_config = MLConfig.for_improved_training(symbol="BTC-USD")
```

### 3. Training Function Changes

**Before:**
```python
# Simple training
from src.ml_forecast_prob_dist import main
signals = main(config)

# Improved training  
from src.ml_forecast_improved import run_improved_training
results = run_improved_training(config)
```

**After:**
```python
# Unified training interface
from src.ml_forecasting import train_model

# Simple training
config = MLConfig.for_simple_training(symbol="BTC-USD")
results = train_model(config)

# Improved training
config = MLConfig.for_improved_training(symbol="BTC-USD") 
results = train_model(config)

# Access signals
signals = results['signals']
model = results['model']
metadata = results['metadata']
```

### 4. Multi-Asset Training

**Before:**
```python
from multi_crypto_ml_training import train_multi_crypto_models
from src.ml_forecast_improved import ImprovedConfig

base_config = ImprovedConfig(
    start="2020-01-01",
    end="2024-01-01",
    n_quantiles=5
)

assets = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
signals_df = train_multi_crypto_models(assets, base_config)
```

**After:**
```python
from src.ml_forecasting import train_multi_crypto_models, MLConfig

base_config = MLConfig.for_improved_training(
    start="2020-01-01",
    end="2024-01-01", 
    n_quantiles=5
)

assets = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
results = train_multi_crypto_models(assets, base_config, parallel=True)

signals_df = results['signals_df']
summary = results['summary']
```

### 5. Feature Engineering

**Before:**
```python
from src.ml_forecast_prob_dist import add_features
df_features = add_features(df, config)
```

**After:**
```python
from src.ml_forecasting.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(config)
df_features = engineer.engineer_features(df)
feature_names = engineer.get_feature_names()

# Or use convenience function for backward compatibility:
from src.ml_forecasting.feature_engineering import add_features
df_features = add_features(df, config)
```

### 6. Model Creation

**Before:**
```python
from src.ml_forecast_prob_dist import MLPClassifier
model = MLPClassifier(input_dim, config)
```

**After:**
```python
from src.ml_forecasting.models import create_model

model = create_model(input_dim, config, model_type="auto")
# model_type options: "mlp", "simple", "ensemble", "auto"
```

### 7. Signal Generation

**Before:**
```python
# Signals were generated as part of training
signals = main(config)  # or from results['signals']
```

**After:**
```python
from src.ml_forecasting import generate_trading_signals

# Generate signals from trained model
signals = generate_trading_signals(model, dataset, config)

# Or with confidence filtering
from src.ml_forecasting.signal_generation import generate_signals_with_confidence
signals_with_conf = generate_signals_with_confidence(model, dataset, config, confidence_threshold=0.6)
```

## Complete Example Migration

### Before (Original Code)
```python
from src.ml_forecast_improved import run_improved_training, ImprovedConfig
import pandas as pd

# Old way
config = ImprovedConfig(
    symbol="DOGE-USD",
    start="2020-01-01",
    end="2024-01-01",
    interval="4h",
    forecast_horizon_hours=6,
    vol_window_hours=60,
    n_quantiles=5,
    hidden_sizes=(64, 32, 16),
    n_epochs=30,
    lr=1e-4,
    enable_regime_features=True
)

results = run_improved_training(config)
signals = results['signals']
```

### After (New Centralized Code)
```python
from src.ml_forecasting import MLConfig, train_model
import pandas as pd

# New way - more explicit and flexible
config = MLConfig.for_improved_training(
    symbol="DOGE-USD",
    start="2020-01-01", 
    end="2024-01-01",
    interval="4h",
    forecast_horizon_hours=6,
    vol_window_hours=60,
    n_quantiles=5,
    hidden_sizes=(64, 32, 16),
    n_epochs=30,
    lr=1e-4,
    enable_regime_features=True
)

results = train_model(config)
signals = results['signals']
model = results['model']
metadata = results['metadata']

# Additional capabilities now available:
from src.ml_forecasting.evaluation import evaluate_model
from src.ml_forecasting.signal_generation import analyze_signal_quality

# Evaluate model performance
test_results = evaluate_model(model, test_dataset, config)

# Analyze signal quality
signal_analysis = analyze_signal_quality(signals, future_returns, config)
```

## Backward Compatibility

The new module maintains backward compatibility:

```python
# These still work:
from src.ml_forecasting.config import Config, ImprovedConfig

# Legacy class names are aliases to MLConfig
config = Config(symbol="BTC-USD")  # Same as MLConfig
improved = ImprovedConfig(symbol="BTC-USD")  # Same as MLConfig
```

## Advanced Features

The new module includes enhanced features:

### 1. Asset-Specific Configurations
```python
# Optimized configs for specific cryptocurrencies
btc_config = MLConfig.for_crypto_asset("BTC-USD")
eth_config = MLConfig.for_crypto_asset("ETH-USD") 
doge_config = MLConfig.for_crypto_asset("DOGE-USD")
```

### 2. Parallel Multi-Asset Training
```python
# Train multiple assets in parallel
results = train_multi_crypto_models(
    assets=['BTC-USD', 'ETH-USD', 'XRP-USD'],
    base_config=base_config,
    parallel=True,
    max_workers=3
)
```

### 3. Enhanced Evaluation
```python
from src.ml_forecasting.evaluation import compare_models

# Compare multiple models
models = {
    'simple': simple_model,
    'improved': improved_model,
    'ensemble': ensemble_model
}
comparison = compare_models(models, test_dataset, config)
```

### 4. Signal Analysis Tools
```python
from src.ml_forecasting.signal_generation import SignalGenerator

# Class-based signal generation with state management
generator = SignalGenerator(model, config)
signals = generator.generate(dataset)
quality_analysis = generator.analyze_quality(returns)
generator.save("experiment_1")
```

## Testing

Run the test script to verify migration:

```bash
python test_centralized_ml.py
```

This will test all functionality and ensure compatibility.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to update import paths to `src.ml_forecasting`

2. **Config Parameter Mismatches**: Some parameter names may have changed. Check the MLConfig class documentation.

3. **Return Value Differences**: The new `train_model()` returns a dictionary with `model`, `signals`, and `metadata` keys.

### Getting Help

If you encounter issues during migration:

1. Check the test script output for specific errors
2. Review the MLConfig class for available parameters
3. Use the factory methods (`for_simple_training`, `for_improved_training`) for quick setup
4. Enable `verbose=True` in config for detailed logging

## Summary

The centralized ML forecasting module provides:

- **Cleaner API** with unified configuration and training interface
- **Better Performance** with optimized data loading and parallel training  
- **Enhanced Features** like signal analysis and model comparison
- **Easier Maintenance** with reduced code duplication
- **Backward Compatibility** to ease migration

Start by migrating one script at a time, using the examples above as templates. 