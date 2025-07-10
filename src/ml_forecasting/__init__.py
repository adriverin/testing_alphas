"""
ML Forecasting Module
====================

A centralized module for cryptocurrency price forecasting using machine learning.

This module consolidates functionality from:
- ml_forecast_prob_dist.py (base implementation)
- ml_forecast_improved.py (enhanced time-series validation)
- multi_crypto_ml_training.py (multi-asset training)

Key Components:
- config: Unified configuration management
- data_loader: Price data loading and preprocessing
- feature_engineering: Technical indicator computation
- models: Neural network architectures
- training: Training pipelines (simple and improved)
- evaluation: Model evaluation and metrics
- signal_generation: Trading signal generation
- multi_asset: Multi-cryptocurrency training orchestration

Usage:
    from src.ml_forecasting import MLConfig, train_model, generate_signals
    
    config = MLConfig(symbol="BTC-USD", training_mode="improved")
    results = train_model(config)
    signals = generate_signals(results['model'], config)
"""

from .config import MLConfig
from .training import train_model, run_simple_training, run_improved_training
from .signal_generation import generate_trading_signals
from .multi_asset import train_multi_crypto_models

__version__ = "1.0.0"
__all__ = [
    "MLConfig",
    "train_model", 
    "run_simple_training",
    "run_improved_training",
    "generate_trading_signals",
    "train_multi_crypto_models"
] 