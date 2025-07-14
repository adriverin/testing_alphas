"""
Trading Signal Generation
=========================

Centralized signal generation for ML forecasting models.
Consolidates signal generation logic from original files.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, Optional
import warnings

from .config import MLConfig


def generate_trading_signals(model: torch.nn.Module, dataset, config: MLConfig, 
                           mode: str = "auto") -> pd.Series:
    """
    Generate trading signals based on model predictions.
    
    Args:
        model: Trained ML model
        dataset: Dataset to generate signals for
        config: ML configuration
        mode: Signal generation mode ("simple", "improved", or "auto")
        
    Returns:
        pd.Series with trading signals (-1, 0, 1)
    """
    if mode == "auto":
        mode = config.training_mode
    
    print(f"📊 Generating trading signals using {mode} mode...")
    
    # Get model predictions
    probabilities = _get_model_probabilities(model, dataset, config)
    
    if mode == "simple":
        signals = _generate_simple_signals(probabilities, config)
    elif mode == "improved":
        signals = _generate_improved_signals(probabilities, config)
    else:
        raise ValueError(f"Unknown signal generation mode: {mode}")
    
    # Convert to pandas Series
    signals_series = pd.Series(signals, name='signal')
    
    # Print signal statistics
    signal_counts = signals_series.value_counts().sort_index()
    total_signals = len(signals_series)
    
    print(f"📊 Signal Distribution:")
    print(f"   Short (-1): {signal_counts.get(-1, 0):4d} ({signal_counts.get(-1, 0)/total_signals*100:.1f}%)")
    print(f"   Neutral(0): {signal_counts.get(0, 0):4d} ({signal_counts.get(0, 0)/total_signals*100:.1f}%)")
    print(f"   Long (+1):  {signal_counts.get(1, 0):4d} ({signal_counts.get(1, 0)/total_signals*100:.1f}%)")
    
    return signals_series


def _get_model_probabilities(model: torch.nn.Module, dataset, config: MLConfig) -> np.ndarray:
    """Get probability predictions from model."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    all_probabilities = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(config.device)
            
            # Get predictions based on model type
            if hasattr(model, 'predict_proba'):
                # For SimpleModel
                probs = model.predict_proba(X_batch)
            elif hasattr(model, 'get_logits'):
                # For MLPClassifier
                logits = model.get_logits(X_batch)
                probs = torch.softmax(logits, dim=1)
            else:
                # Default forward pass (should return probabilities)
                probs = model(X_batch)
            
            all_probabilities.append(probs.cpu().numpy())
    
    return np.vstack(all_probabilities)


def _generate_simple_signals(probabilities: np.ndarray, config: MLConfig) -> np.ndarray:
    """
    Generate signals using simple threshold-based approach.
    Based on original ml_forecast_prob_dist.py logic.
    """
    # Use percentile-based approach for signal generation
    bottom_scores = probabilities[:, 0] + probabilities[:, 1]  # Bottom 2 quantiles
    top_scores = probabilities[:, -2] + probabilities[:, -1]   # Top 2 quantiles
    
    # Calculate relative preference for extremes
    extreme_preference = top_scores - bottom_scores
    
    # Use percentile thresholds
    top_threshold = np.percentile(extreme_preference, 75)
    bottom_threshold = np.percentile(extreme_preference, 25)
    
    signals = np.zeros(len(probabilities), dtype=int)
    
    # Generate signals
    signals[extreme_preference > top_threshold] = 1   # Long signal
    signals[extreme_preference < bottom_threshold] = -1  # Short signal
    # Rest remain 0 (neutral)
    
    return signals


def _generate_improved_signals(probabilities: np.ndarray, config: MLConfig) -> np.ndarray:
    """
    Generate signals using improved approach with direction testing.
    Based on logic from ml_forecast_improved.py.
    """
    # Calculate extreme preference scores
    bottom_scores = probabilities[:, 0] + probabilities[:, 1]  # Bottom 2 quantiles
    top_scores = probabilities[:, -2] + probabilities[:, -1]   # Top 2 quantiles
    extreme_preference = top_scores - bottom_scores
    
    # Use configured percentiles for thresholds
    # top_threshold = np.percentile(extreme_preference, config.signal_percentiles[1])
    # bottom_threshold = np.percentile(extreme_preference, config.signal_percentiles[0])
    top_threshold = np.percentile(extreme_preference, 99)
    bottom_threshold = np.percentile(extreme_preference, 1)
    # print("="*1000)
    # print(f"Top threshold: {top_threshold}, Bottom threshold: {bottom_threshold}")
    
    # Generate signals in original direction
    signals_original = np.zeros(len(probabilities), dtype=int)
    signals_original[extreme_preference > top_threshold] = 1   # Momentum direction
    signals_original[extreme_preference < bottom_threshold] = -1  # Momentum direction
    
    # Generate signals in reversed direction (mean reversion)
    signals_reversed = np.zeros(len(probabilities), dtype=int)
    signals_reversed[extreme_preference > top_threshold] = -1  # Mean reversion
    signals_reversed[extreme_preference < bottom_threshold] = 1   # Mean reversion
    
    # For improved mode, we typically use the reversed direction based on
    # analysis showing better correlation with future returns
    # This can be configurable in the future
    return signals_original


def generate_signals_with_confidence(model: torch.nn.Module, dataset, config: MLConfig,
                                   confidence_threshold: float = 0.6) -> pd.DataFrame:
    """
    Generate signals with confidence scores.
    
    Args:
        model: Trained ML model
        dataset: Dataset for signal generation
        config: ML configuration
        confidence_threshold: Minimum confidence for signal generation
        
    Returns:
        DataFrame with signals and confidence scores
    """
    probabilities = _get_model_probabilities(model, dataset, config)
    
    # Calculate confidence as maximum probability
    confidence_scores = probabilities.max(axis=1)
    
    # Generate base signals
    if config.training_mode == "simple":
        base_signals = _generate_simple_signals(probabilities, config)
    else:
        base_signals = _generate_improved_signals(probabilities, config)
    
    # Apply confidence filter
    low_confidence_mask = confidence_scores < confidence_threshold
    filtered_signals = base_signals.copy()
    filtered_signals[low_confidence_mask] = 0  # Set to neutral for low confidence
    
    # Create result DataFrame
    result = pd.DataFrame({
        'signal': filtered_signals,
        'confidence': confidence_scores,
        'raw_signal': base_signals
    })
    
    # Statistics
    total = len(result)
    filtered_out = low_confidence_mask.sum()
    
    print(f"📊 Confidence Filtering Results:")
    print(f"   Threshold: {confidence_threshold:.2f}")
    print(f"   Filtered out: {filtered_out}/{total} ({filtered_out/total*100:.1f}%)")
    print(f"   Mean confidence: {confidence_scores.mean():.3f}")
    
    return result


def analyze_signal_quality(signals: pd.Series, returns: pd.Series, 
                          config: MLConfig) -> dict:
    """
    Analyze the quality of generated trading signals.
    
    Args:
        signals: Trading signals (-1, 0, 1)
        returns: Future returns aligned with signals
        config: ML configuration
        
    Returns:
        Dictionary with signal quality metrics
    """
    # Align signals and returns
    aligned_data = pd.DataFrame({
        'signal': signals,
        'return': returns
    }).dropna()
    
    if len(aligned_data) == 0:
        return {'error': 'No aligned data available'}
    
    results = {}
    
    # Basic statistics
    results['total_signals'] = len(aligned_data)
    results['signal_distribution'] = aligned_data['signal'].value_counts().to_dict()
    
    # Signal-return correlation
    correlation = aligned_data['signal'].corr(aligned_data['return'])
    results['signal_return_correlation'] = correlation
    
    # Performance by signal type
    for signal_type in [-1, 0, 1]:
        mask = aligned_data['signal'] == signal_type
        if mask.sum() > 0:
            signal_returns = aligned_data.loc[mask, 'return']
            results[f'signal_{signal_type}_stats'] = {
                'count': mask.sum(),
                'mean_return': signal_returns.mean(),
                'std_return': signal_returns.std(),
                'sharpe_ratio': signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0
            }
    
    # Directional accuracy
    long_signals = aligned_data['signal'] == 1
    short_signals = aligned_data['signal'] == -1
    
    if long_signals.sum() > 0:
        long_accuracy = (aligned_data.loc[long_signals, 'return'] > 0).mean()
        results['long_accuracy'] = long_accuracy
    
    if short_signals.sum() > 0:
        short_accuracy = (aligned_data.loc[short_signals, 'return'] < 0).mean()
        results['short_accuracy'] = short_accuracy
    
    # Overall signal effectiveness
    if (long_signals.sum() > 0) and (short_signals.sum() > 0):
        overall_accuracy = (
            (aligned_data.loc[long_signals, 'return'] > 0).sum() +
            (aligned_data.loc[short_signals, 'return'] < 0).sum()
        ) / (long_signals.sum() + short_signals.sum())
        results['overall_directional_accuracy'] = overall_accuracy
    
    return results


def save_signals(signals: Union[pd.Series, pd.DataFrame], config: MLConfig, 
                suffix: str = "") -> str:
    """
    Save trading signals to file.
    
    Args:
        signals: Trading signals to save
        config: ML configuration
        suffix: Optional suffix for filename
        
    Returns:
        Path to saved file
    """
    # Determine filename
    base_name = f"{config.symbol}_{config.training_mode}_signals"
    if suffix:
        base_name += f"_{suffix}"
    
    save_path = config.signals_cache_dir / f"{base_name}.parquet"
    
    # Save signals
    if isinstance(signals, pd.Series):
        signals_df = pd.DataFrame({'signal': signals})
    else:
        signals_df = signals
    
    signals_df.to_parquet(save_path)
    
    print(f"💾 Signals saved to: {save_path}")
    return str(save_path)


def load_signals(config: MLConfig, suffix: str = "") -> Optional[pd.DataFrame]:
    """
    Load previously saved trading signals.
    
    Args:
        config: ML configuration
        suffix: Optional suffix for filename
        
    Returns:
        DataFrame with signals or None if not found
    """
    base_name = f"{config.symbol}_{config.training_mode}_signals"
    if suffix:
        base_name += f"_{suffix}"
    
    load_path = config.signals_cache_dir / f"{base_name}.parquet"
    
    if load_path.exists():
        signals_df = pd.read_parquet(load_path)
        print(f"📁 Loaded signals from: {load_path}")
        return signals_df
    else:
        print(f"⚠️  No signals found at: {load_path}")
        return None


class SignalGenerator:
    """
    Class-based interface for signal generation with state management.
    """
    
    def __init__(self, model: torch.nn.Module, config: MLConfig):
        self.model = model
        self.config = config
        self.last_probabilities = None
        self.last_signals = None
    
    def generate(self, dataset, mode: str = "auto") -> pd.Series:
        """Generate trading signals and store state."""
        signals = generate_trading_signals(self.model, dataset, self.config, mode)
        self.last_signals = signals
        return signals
    
    def generate_with_confidence(self, dataset, confidence_threshold: float = 0.6) -> pd.DataFrame:
        """Generate signals with confidence filtering."""
        result = generate_signals_with_confidence(
            self.model, dataset, self.config, confidence_threshold
        )
        self.last_signals = result['signal']
        return result
    
    def analyze_quality(self, returns: pd.Series) -> dict:
        """Analyze quality of last generated signals."""
        if self.last_signals is None:
            raise ValueError("No signals generated yet. Call generate() first.")
        
        return analyze_signal_quality(self.last_signals, returns, self.config)
    
    def save(self, suffix: str = "") -> str:
        """Save last generated signals."""
        if self.last_signals is None:
            raise ValueError("No signals to save. Call generate() first.")
        
        return save_signals(self.last_signals, self.config, suffix) 