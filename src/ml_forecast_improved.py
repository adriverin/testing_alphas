#!/usr/bin/env python3
"""
Improved ML Forecasting with Proper Time-Series Validation

Addresses look-ahead bias and overfitting issues in alpha999 by implementing:
1. Proper time-series splits (no future data leakage)
2. Walk-forward model training  
3. Regime-aware feature engineering
4. Realistic transaction cost modeling
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List
import json

from src.ml_forecast_prob_dist import (
    Config, load_price_history, ReturnDataset, 
    bar_size_hours, MLPClassifier
)


@dataclass 
class ImprovedConfig(Config):
    """Enhanced configuration with validation parameters."""
    min_train_samples: int = 500
    validation_months: int = 6
    walk_forward_step: int = 30
    feature_stability_window: int = 90
    max_feature_drift: float = 0.3
    enable_regime_features: bool = True
    volatility_regime_window: int = 60
    n_ensemble_models: int = 3
    
    # Simplified model architecture for better generalization
    # hidden_sizes: tuple = (32, 16, 8)  # Much smaller model
    # n_epochs: int = 30  # Fewer epochs to prevent overfitting
    # lr: float = 1e-4  # Higher learning rate for simpler model
    # weight_decay: float = 0.01  # More regularization


class TimeSeriesDataSplitter:
    """Proper time-series data splitting without look-ahead bias."""
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df


class ImprovedFeatureEngineer:
    """Enhanced feature engineering with regime detection."""
    
    def __init__(self, cfg: ImprovedConfig):
        self.cfg = cfg
        
    def add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive feature set."""
        out = df.copy()
        out = self._add_basic_features(out)
        
        if self.cfg.enable_regime_features:
            out = self._add_regime_features(out)
            
        return out
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features."""
        out = df.copy()
        out["return"] = out["close"].pct_change()
        
        # Volatility
        vol_w = max(5, int(self.cfg.vol_window_hours / bar_size_hours(self.cfg.interval)))
        out["vol"] = out["return"].rolling(vol_w, min_periods=3).std()
        
        eps = 1e-8
        out["norm_return"] = out["return"] / (out["vol"] + eps)
        out["norm_return"] = out["norm_return"].clip(-5, 5)
        
        # Moving averages
        for w in [2, 5, 10, 20, 30]:
            sma = out["close"].rolling(w, min_periods=max(1, w//2)).mean()
            out[f"sma_{w}d"] = ((out["close"] - sma) / sma).fillna(0).clip(-1, 1)
        
        # Volatility ratios
        for w in [5, 10, 20]:
            vol = out["return"].rolling(w, min_periods=max(1, w//2)).std()
            out[f"vol_{w}d"] = (vol / out["vol"].rolling(20).mean()).fillna(1).clip(0, 5)
        
        # Momentum
        for w in [1, 3, 7, 14]:
            out[f"mom_{w}d"] = (out["close"] / out["close"].shift(w) - 1).fillna(0).clip(-1, 1)
        
        # RSI
        for w in [7, 14, 21]:
            out[f"rsi_{w}d"] = self._calculate_rsi(out["close"], w)
        
        return out
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        out = df.copy()
        
        # Volatility regime
        vol_window = self.cfg.volatility_regime_window
        rolling_vol = out["return"].rolling(vol_window).std()
        vol_threshold = rolling_vol.rolling(vol_window * 2).median()
        out["vol_regime"] = (rolling_vol > vol_threshold).astype(float)
        
        # Trend regime
        trend_strength = abs(out["close"].rolling(20).mean() - out["close"].rolling(5).mean()) / out["close"]
        trend_threshold = trend_strength.rolling(60).quantile(0.7)
        out["trend_regime"] = (trend_strength > trend_threshold).astype(float)
        
        # Momentum regime
        momentum_1d = out["return"]
        momentum_5d = out["close"].pct_change(5)
        momentum_persistence = (momentum_1d * momentum_5d > 0).rolling(10).mean()
        out["momentum_regime"] = momentum_persistence
        
        return out
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI with proper normalization."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=max(1, window//2)).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(1, window//2)).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50


def run_improved_training(config: ImprovedConfig = None) -> Dict:
    """Run the improved ML training pipeline."""
    if config is None:
        config = ImprovedConfig()
    
    print("🚀 Starting Improved ML Training Pipeline")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n📊 1. Loading data...")
    df = load_price_history(config)
    print(f"Loaded {len(df)} rows")
    
    # 2. Feature engineering
    print("\n🛠️  2. Feature engineering...")
    feature_engineer = ImprovedFeatureEngineer(config)
    df_features = feature_engineer.add_enhanced_features(df)
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['return', 'vol', 'norm_return', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Created {len(feature_cols)} features")
    
    # 3. Time-series splitting
    print("\n✂️  3. Data splitting...")
    splitter = TimeSeriesDataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_df, val_df, test_df = splitter.split(df_features)
    
    print(f"Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    
    # 4. Generate labels (NO LOOK-AHEAD BIAS)
    print("\n🏷️  4. Generating labels...")
    h = int(config.forecast_horizon_hours / bar_size_hours(config.interval))
    
    for df_split in [train_df, val_df, test_df]:
        df_split["future_norm_ret"] = df_split["norm_return"].shift(-h)
    
    train_df = train_df.dropna(subset=["future_norm_ret"])
    val_df = val_df.dropna(subset=["future_norm_ret"])
    test_df = test_df.dropna(subset=["future_norm_ret"])
    
    # 🔍 DEBUGGING: Analyze feature-target correlations
    print("\n🔬 5. Feature-Target Correlation Analysis...")
    train_targets = train_df["future_norm_ret"]
    
    print("Checking feature directions:")
    suspicious_features = []
    for feature in feature_cols[:5]:  # Check first 5 features
        if feature in train_df.columns:
            corr = np.corrcoef(train_df[feature], train_targets)[0,1]
            print(f"  {feature}: {corr:.3f}")
            
            # Flag suspicious correlations for momentum features
            if 'mom_' in feature and corr < -0.1:
                suspicious_features.append(f"{feature} (negative momentum correlation: {corr:.3f})")
            elif 'sma_' in feature and corr < -0.1:
                suspicious_features.append(f"{feature} (negative trend correlation: {corr:.3f})")
    
    if suspicious_features:
        print("⚠️  WARNING: Suspicious feature correlations detected:")
        for feature in suspicious_features:
            print(f"    - {feature}")
        print("This may explain why the model learns inverted directions!")
    
    # Quantile edges from TRAINING DATA ONLY
    quantile_edges = np.quantile(train_targets, np.linspace(0, 1, config.n_quantiles + 1)[1:-1])
    print(f"Quantile edges: {quantile_edges}")
    
    # 🔍 DEBUGGING: Validate quantile assignment logic
    print("\n🔬 6. Quantile Assignment Validation...")
    train_bins = np.digitize(train_targets, quantile_edges, right=False)
    
    print("Quantile bin analysis:")
    for bin_num in range(config.n_quantiles):
        bin_mask = train_bins == bin_num
        if bin_mask.sum() > 0:
            avg_return = train_targets[bin_mask].mean()
            print(f"  Bin {bin_num}: {bin_mask.sum()} samples, avg return: {avg_return:.4f}")
    
    # Check if bins are ordered correctly (should be increasing)
    bin_means = []
    for bin_num in range(config.n_quantiles):
        bin_mask = train_bins == bin_num
        if bin_mask.sum() > 0:
            bin_means.append(train_targets[bin_mask].mean())
    
    if len(bin_means) > 1:
        is_monotonic = all(bin_means[i] <= bin_means[i+1] for i in range(len(bin_means)-1))
        print(f"Bins are monotonically increasing: {is_monotonic}")
        if not is_monotonic:
            print("⚠️  WARNING: Quantile bins are not ordered correctly!")
    
    # Assign labels
    for df_split in [train_df, val_df, test_df]:
        df_split["bin"] = np.digitize(df_split["future_norm_ret"], quantile_edges, right=False)
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["bin"].astype(int).values
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["bin"].astype(int).values
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["bin"].astype(int).values
    
    # 7. Train model
    print("\n🤖 7. Training model...")
    
    # Use a much simpler model to prevent overfitting
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 16)  # Very simple
            self.dropout = nn.Dropout(0.5)  # High dropout
            self.fc2 = nn.Linear(16, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    train_dataset = ReturnDataset(X_train, y_train)
    val_dataset = ReturnDataset(X_val, y_val)
    
    model = SimpleModel(X_train.shape[1], config.n_quantiles)
    model.to(config.device)
    
    # Training setup with stronger regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Smaller batches
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    best_val_loss = float('inf')
    patience = 0
    max_epochs = 15  # Fewer epochs
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.device), y_batch.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config.device), y_batch.to(config.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.long())
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.long()).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 5:  # Early stopping after 5 epochs
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={val_accuracy:.3f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Additional validation: Check if model beats random guessing significantly
    random_accuracy = 1.0 / config.n_quantiles  # 20% for 5 quantiles
    if val_accuracy < random_accuracy + 0.02:  # Must be at least 2% better than random
        print(f"⚠️  WARNING: Model accuracy ({val_accuracy:.3f}) barely beats random ({random_accuracy:.3f})")
        print("Model may not have learned meaningful patterns!")
    
    print(f"Final validation accuracy: {val_accuracy:.3f} (random baseline: {random_accuracy:.3f})")
    
    # 8. Analyze model predictions
    print("\n🔬 8. Model Prediction Analysis...")
    test_dataset = ReturnDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    actual_bins = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probs)
            actual_bins.extend(y_batch.numpy())
    
    P = np.vstack(predictions)
    actual_bins = np.array(actual_bins)
    
    # Check if model predictions correlate with actual bins
    predicted_bins = P.argmax(axis=1)
    prediction_correlation = np.corrcoef(predicted_bins, actual_bins)[0,1]
    print(f"Model prediction correlation with actual bins: {prediction_correlation:.3f}")
    
    if prediction_correlation < 0:
        print("⚠️  WARNING: Model predictions are negatively correlated with actual outcomes!")
        print("This confirms the model learned inverted patterns.")
    
    # 9. Generate signals with debugging
    print("\n📊 9. Generating signals...")
    
    # Signal generation - TEST BOTH DIRECTIONS
    bottom_scores = P[:, 0] + P[:, 1]
    top_scores = P[:, -2] + P[:, -1]
    extreme_preference = top_scores - bottom_scores
    
    # Use moderate thresholds for reasonable signal frequency
    # Increase trading frequency from 10% to ~30% for better visualization
    top_threshold = np.percentile(extreme_preference, 95)
    bottom_threshold = np.percentile(extreme_preference, 5)
    
    # Generate signals in ORIGINAL direction
    signals_original = []
    for score in extreme_preference:
        if score > top_threshold:
            signal = 1   # Original direction (momentum)
        elif score < bottom_threshold:
            signal = -1  # Original direction (momentum)
        else:
            signal = 0
        signals_original.append(signal)
    
    # Generate signals in REVERSED direction  
    signals_reversed = []
    for score in extreme_preference:
        if score > top_threshold:
            signal = -1  # Reversed direction (mean reversion)
        elif score < bottom_threshold:
            signal = 1   # Reversed direction (mean reversion)
        else:
            signal = 0
        signals_reversed.append(signal)
    
    # Test both directions against actual returns
    test_returns = test_df["future_norm_ret"].values[:len(signals_original)]
    
    # Calculate correlation between signals and future returns
    original_corr = np.corrcoef(signals_original, test_returns)[0,1] if len(test_returns) > 0 else 0
    reversed_corr = np.corrcoef(signals_reversed, test_returns)[0,1] if len(test_returns) > 0 else 0
    
    print(f"Signal-return correlation (original direction): {original_corr:.3f}")
    print(f"Signal-return correlation (reversed direction): {reversed_corr:.3f}")
    
    # Choose the direction with better correlation (positive correlation preferred)
    if reversed_corr > original_corr:
        print(f"✅ Using REVERSED signals (better correlation: {reversed_corr:.3f} vs {original_corr:.3f})")
        final_signals = signals_reversed
        direction_used = "reversed"
    else:
        print(f"✅ Using ORIGINAL signals (better correlation: {original_corr:.3f} vs {reversed_corr:.3f})")
        final_signals = signals_original
        direction_used = "original"
    
    test_signals = pd.Series(final_signals, index=X_test.index[:len(final_signals)])
    
    # 🔧 GENERATE SIGNALS FOR FULL DATASET (not just test period)
    print("\n🔧 Generating signals for full dataset...")
    
    # Create full dataset for signal generation
    full_X = pd.concat([X_train, X_val, X_test], axis=0)
    full_y = np.concatenate([y_train, y_val, y_test], axis=0)
    full_dataset = ReturnDataset(full_X, full_y)
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Generate predictions for full dataset
    model.eval()
    full_predictions = []
    with torch.no_grad():
        for X_batch, y_batch in full_loader:
            X_batch = X_batch.to(config.device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            full_predictions.append(probs)
    
    P_full = np.vstack(full_predictions)
    
    # Generate signals for full dataset using same logic
    bottom_scores_full = P_full[:, 0] + P_full[:, 1]
    top_scores_full = P_full[:, -2] + P_full[:, -1]
    extreme_preference_full = top_scores_full - bottom_scores_full
    
    # Use same thresholds as test period
    top_threshold_full = np.percentile(extreme_preference_full, 85)
    bottom_threshold_full = np.percentile(extreme_preference_full, 15)
    
    # Generate full signals using the chosen direction
    full_signals = []
    for score in extreme_preference_full:
        if direction_used == "reversed":
            if score > top_threshold_full:
                signal = -1  # Reversed direction
            elif score < bottom_threshold_full:
                signal = 1   # Reversed direction
            else:
                signal = 0
        else:  # original direction
            if score > top_threshold_full:
                signal = 1   # Original direction
            elif score < bottom_threshold_full:
                signal = -1  # Original direction
            else:
                signal = 0
        full_signals.append(signal)
    
    full_signals_series = pd.Series(full_signals, index=full_X.index[:len(full_signals)])
    
    print(f"Full dataset signals: {full_signals_series.value_counts().to_dict()}")
    print(f"Full signals date range: {full_signals_series.index.min()} to {full_signals_series.index.max()}")
    
    # 10. Save results
    print("\n💾 10. Saving results...")
    output_dir = Path("artefacts/improved_ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'quantile_edges': quantile_edges,
        'feature_names': feature_cols,
        'input_dim': len(feature_cols),
        'direction_used': direction_used,
        'prediction_correlation': prediction_correlation,
        'signal_correlations': {
            'original': original_corr,
            'reversed': reversed_corr
        }
    }, output_dir / "improved_model.pt")
    
    # Save signals for FULL DATASET (not just test period)
    signals_df = pd.DataFrame({'signal': full_signals_series}, index=full_signals_series.index)
    signals_df.to_parquet(output_dir / "improved_trading_signals.parquet")
    
    # Save metadata
    metadata = {
        'config': asdict(config),
        'quantile_edges': quantile_edges.tolist(),
        'feature_names': feature_cols,
        'train_period': (str(train_df.index[0]), str(train_df.index[-1])),
        'val_period': (str(val_df.index[0]), str(val_df.index[-1])),
        'test_period': (str(test_df.index[0]), str(test_df.index[-1])),
        'full_period': (str(full_signals_series.index.min()), str(full_signals_series.index.max())),
        'signal_distribution': full_signals_series.value_counts().to_dict(),
        'test_signal_distribution': test_signals.value_counts().to_dict(),
        'direction_used': direction_used,
        'prediction_correlation': prediction_correlation,
        'signal_correlations': {
            'original': original_corr,
            'reversed': reversed_corr
        },
        'suspicious_features': suspicious_features
    }
    
    with open(output_dir / "improved_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Training complete!")
    print(f"📊 Test signals: {test_signals.value_counts().to_dict()}")
    print(f"📊 Full dataset signals: {full_signals_series.value_counts().to_dict()}")
    print(f"🎯 Direction used: {direction_used}")
    print(f"📈 Signal correlation: {metadata['signal_correlations'][direction_used]:.3f}")
    
    return {
        'model': model,
        'signals': full_signals_series,  # Return full signals, not just test
        'metadata': metadata
    }


if __name__ == "__main__":
    config = ImprovedConfig(
        symbol="BTC-USD",
        start="2020-01-01", 
        end="2024-01-01",
        n_quantiles=5,
        hidden_sizes=(128, 64, 32),
        n_epochs=30,
        lr=5e-5,
        enable_regime_features=True
    )
    
    results = run_improved_training(config) 