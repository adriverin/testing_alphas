"""
Unified ML Configuration
========================

Centralized configuration management for ML forecasting models.
Combines and enhances Config and ImprovedConfig from the original files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import torch


@dataclass
class MLConfig:
    """
    Unified configuration for ML forecasting models.
    
    Combines functionality from original Config and ImprovedConfig classes
    with improved organization and documentation.
    """
    
    # ==================== DATA PARAMETERS ====================
    symbol: str = "BTC-USD"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    interval: str = "1d"  # Supported: 1m, 5m, 15m, 1h, 4h, 1d
    price_column: str = "close"  # Price column for ML training: "open", "high", "low", "close", "vwap"
    
    # ==================== FORECASTING PARAMETERS ====================
    forecast_horizon_hours: int = 24  # How far ahead to predict
    vol_window_hours: int = 240       # Volatility estimation window (10 days default)
    
    # ==================== FEATURE ENGINEERING ====================
    # Technical indicator windows (in bars/periods, not hours)
    sma_windows: Tuple[int, ...] = (2, 5, 10, 20, 30)
    volatility_windows: Tuple[int, ...] = (5, 10, 20)  
    momentum_windows: Tuple[int, ...] = (1, 3, 7, 14)
    rsi_windows: Tuple[int, ...] = (7, 14, 21)
    
    # Regime detection features
    enable_regime_features: bool = True
    volatility_regime_window: int = 60
    feature_stability_window: int = 90
    max_feature_drift: float = 0.3
    
    # ==================== MODEL ARCHITECTURE ====================
    n_quantiles: int = 5
    hidden_sizes: Tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.2
    
    # ==================== TRAINING PARAMETERS ====================
    training_mode: str = "improved"  # "simple" or "improved"
    n_epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-4
    batch_size: int = 256
    
    # Simple training parameters
    test_fraction: float = 0.20  # For simple mode
    
    # Improved training parameters  
    train_ratio: float = 0.8     # For improved mode
    val_ratio: float = 0.05       # For improved mode
    test_ratio: float = 0.15      # For improved mode
    min_train_samples: int = 500
    validation_months: int = 6
    walk_forward_step: int = 30
    n_ensemble_models: int = 3
    
    # Early stopping
    early_stopping_patience: int = 10
    min_improvement: float = 1e-4
    
    # ==================== SIGNAL GENERATION ====================
    threshold: float = 0.4  # Probability threshold for simple mode
    signal_percentiles: Tuple[int, int] = (5, 95)  # For improved mode signal generation
    
    # ==================== INFRASTRUCTURE ====================
    cache_dir: Path = field(default_factory=lambda: Path("artefacts"))
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    verbose: bool = False
    plot_reliability: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure cache directory exists
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate training mode
        if self.training_mode not in ["simple", "improved"]:
            raise ValueError(f"training_mode must be 'simple' or 'improved', got: {self.training_mode}")
        
        # Validate ratios for improved mode
        if self.training_mode == "improved":
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got: {total_ratio}")
        
        # Validate signal percentiles
        if len(self.signal_percentiles) != 2:
            raise ValueError("signal_percentiles must be a tuple of exactly 2 values")
        if not (0 < self.signal_percentiles[0] < self.signal_percentiles[1] < 100):
            raise ValueError("signal_percentiles must be ordered and between 0-100")
    
    @property
    def model_cache_dir(self) -> Path:
        """Directory for saving trained models."""
        model_dir = self.cache_dir / "models"
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    @property
    def signals_cache_dir(self) -> Path:
        """Directory for saving trading signals."""
        signals_dir = self.cache_dir / "signals"
        signals_dir.mkdir(exist_ok=True)
        return signals_dir
    
    @property
    def data_cache_dir(self) -> Path:
        """Directory for caching price data."""
        data_dir = self.cache_dir / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        import dataclasses
        return {
            **dataclasses.asdict(self),
            'cache_dir': str(self.cache_dir),
            'model_cache_dir': str(self.model_cache_dir),
            'signals_cache_dir': str(self.signals_cache_dir),
            'data_cache_dir': str(self.data_cache_dir)
        }
    
    @classmethod
    def for_simple_training(cls, **kwargs) -> 'MLConfig':
        """Create config optimized for simple training mode."""
        defaults = {
            'training_mode': 'simple',
            'hidden_sizes': (128, 64, 32),
            'n_epochs': 50,
            'lr': 5e-5,
            'enable_regime_features': False,
            'threshold': 0.4
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod 
    def for_improved_training(cls, **kwargs) -> 'MLConfig':
        """Create config optimized for improved training mode."""
        defaults = {
            'training_mode': 'improved',
            'hidden_sizes': (64, 32, 16),  # Smaller model to prevent overfitting
            'n_epochs': 30,                # Fewer epochs
            'lr': 1e-4,                    # Higher learning rate
            'weight_decay': 0.01,          # More regularization
            'enable_regime_features': True,
            'signal_percentiles': (5, 95), # More conservative signals
            'dropout_rate': 0.5            # Higher dropout
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_crypto_asset(cls, symbol: str, **kwargs) -> 'MLConfig':
        """Create config optimized for specific cryptocurrency."""
        
        # Asset-specific optimizations
        crypto_defaults = {
            'BTC-USD': {
                'interval': '1h',
                'forecast_horizon_hours': 6,
                'vol_window_hours': 72,
            },
            'ETH-USD': {
                'interval': '1h', 
                'forecast_horizon_hours': 6,
                'vol_window_hours': 48,
            },
            'DOGE-USD': {
                'interval': '4h',
                'forecast_horizon_hours': 12,
                'vol_window_hours': 96,
            }
        }
        
        defaults = {'symbol': symbol}
        if symbol in crypto_defaults:
            defaults.update(crypto_defaults[symbol])
        defaults.update(kwargs)
        
        return cls(**defaults)


# Legacy compatibility - maintain original class names for backward compatibility
Config = MLConfig  # Alias for original Config class
ImprovedConfig = MLConfig  # Alias for original ImprovedConfig class 