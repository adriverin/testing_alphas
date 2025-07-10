"""
Training Pipeline
=================

Centralized training functionality combining simple and improved approaches.
Consolidates training logic from original ml_forecast_prob_dist.py and ml_forecast_improved.py.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List, Optional
import json
import random
from pathlib import Path

from .config import MLConfig
from .data_loader import load_and_validate_data, bar_size_hours
from .feature_engineering import FeatureEngineer
from .models import create_model, get_model_info, ModelCheckpoint
from .evaluation import evaluate_model, ModelEvaluator


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ReturnDataset(Dataset):
    """
    Dataset class for normalized feature data.
    Enhanced version with better normalization handling.
    """
    
    def __init__(self, X: pd.DataFrame, y: np.ndarray, normalize: bool = True):
        """
        Initialize dataset with optional normalization.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        
        # Convert to numpy
        X_values = X.values.astype(np.float32)
        
        if normalize:
            # Calculate normalization stats
            self.mean_ = X_values.mean(axis=0)
            self.std_ = X_values.std(axis=0) + 1e-8  # Add epsilon to prevent division by zero
            
            # Handle constant features
            zero_std_mask = self.std_ < 1e-6
            if zero_std_mask.any():
                print(f"⚠️  Warning: {zero_std_mask.sum()} constant features detected, setting std to 1")
                self.std_[zero_std_mask] = 1.0
            
            # Normalize
            X_values = (X_values - self.mean_) / self.std_
            
            # Additional safety checks
            invalid_mask = np.isnan(X_values) | np.isinf(X_values)
            if invalid_mask.any():
                print(f"⚠️  Warning: {invalid_mask.sum()} invalid values after normalization, setting to 0")
                X_values[invalid_mask] = 0.0
            
            # Clip extreme values
            X_values = np.clip(X_values, -10, 10)
        
        self.X = torch.tensor(X_values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        print(f"📊 Dataset created: {self.X.shape}, target range: [{y.min()}, {y.max()}]")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def transform(self, X: pd.DataFrame) -> torch.Tensor:
        """Transform new data using stored normalization parameters."""
        if not self.normalize:
            return torch.tensor(X.values.astype(np.float32))
        
        X_values = X.values.astype(np.float32)
        X_values = (X_values - self.mean_) / self.std_
        X_values = np.clip(X_values, -10, 10)
        
        # Handle any invalid values
        invalid_mask = np.isnan(X_values) | np.isinf(X_values)
        if invalid_mask.any():
            X_values[invalid_mask] = 0.0
        
        return torch.tensor(X_values, dtype=torch.float32)


class TimeSeriesDataSplitter:
    """Time-series aware data splitting to prevent look-ahead bias."""
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df


def generate_labels(df: pd.DataFrame, config: MLConfig, 
                   split_data: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[int]]:
    """
    Generate quantile-based labels for the forecasting task.
    
    Args:
        df: DataFrame with features and 'norm_return' column
        config: ML configuration
        split_data: Whether to split data (simple mode) or not (improved mode)
        
    Returns:
        (features_df, labels, quantile_edges, split_idx or None)
    """
    # Calculate forecast horizon in bars
    horizon_bars = int(config.forecast_horizon_hours / bar_size_hours(config.interval))
    
    # Create future returns
    df = df.copy()
    df['future_norm_ret'] = df['norm_return'].shift(-horizon_bars)
    df = df.dropna(subset=['future_norm_ret'])
    
    if len(df) < 100:
        raise ValueError("Insufficient data after label generation")
    
    if split_data:
        # Simple mode: use train/test split for quantile calculation
        split_idx = int(len(df) * (1 - config.test_fraction))
        train_returns = df['future_norm_ret'].iloc[:split_idx]
    else:
        # Improved mode: use all data for quantile calculation 
        train_returns = df['future_norm_ret']
        split_idx = None
    
    # Calculate quantile edges from training data
    quantile_probs = np.linspace(0, 1, config.n_quantiles + 1)[1:-1]
    quantile_edges = np.quantile(train_returns, quantile_probs)
    
    # Assign bins
    labels = np.digitize(df['future_norm_ret'], quantile_edges, right=False)
    
    # Validate label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"📊 Label distribution: {dict(zip(unique_labels, counts))}")
    print(f"📊 Quantile edges: {quantile_edges}")
    
    # Remove future return and bin columns from features
    feature_df = df.drop(columns=['future_norm_ret'], errors='ignore')
    
    return feature_df, labels, quantile_edges, split_idx


def train_model(config: MLConfig) -> Dict:
    """
    Main training function that dispatches to appropriate training mode.
    
    Args:
        config: ML configuration
        
    Returns:
        Dictionary with training results
    """
    print(f"🚀 Starting {config.training_mode} training for {config.symbol}")
    print("=" * 60)
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    if config.training_mode == "simple":
        return run_simple_training(config)
    elif config.training_mode == "improved":
        return run_improved_training(config)
    else:
        raise ValueError(f"Unknown training mode: {config.training_mode}")


def run_simple_training(config: MLConfig) -> Dict:
    """
    Simple training approach similar to original ml_forecast_prob_dist.py.
    
    Args:
        config: ML configuration
        
    Returns:
        Dictionary with training results
    """
    print("📊 Simple Training Mode")
    
    # 1. Load and prepare data
    df = load_and_validate_data(config)
    
    # 2. Feature engineering
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    feature_names = engineer.get_feature_names()
    
    # 3. Generate labels with train/test split
    X, y, quantile_edges, split_idx = generate_labels(df_features, config, split_data=True)
    
    # 4. Split data
    X_train = X.iloc[:split_idx][feature_names]
    y_train = y[:split_idx]
    X_test = X.iloc[split_idx:][feature_names]
    y_test = y[split_idx:]
    
    print(f"📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # 5. Create datasets
    train_dataset = ReturnDataset(X_train, y_train, normalize=True)
    test_dataset = ReturnDataset(X_test, y_test, normalize=False)  # Use train normalization
    test_dataset.mean_ = train_dataset.mean_
    test_dataset.std_ = train_dataset.std_
    
    # 6. Create model
    model = create_model(len(feature_names), config, "mlp")
    
    # 7. Train
    trainer = ModelTrainer(config)
    training_history = trainer.train(model, train_dataset, test_dataset)
    
    # 8. Evaluate
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate(model, test_dataset, "Test")
    
    # 9. Generate signals for full dataset
    from .signal_generation import generate_trading_signals
    full_dataset = ReturnDataset(X[feature_names], y, normalize=False)
    full_dataset.mean_ = train_dataset.mean_
    full_dataset.std_ = train_dataset.std_
    
    signals = generate_trading_signals(model, full_dataset, config, mode="simple")
    signals.index = X.index[:len(signals)]
    
    # 10. Save results
    results = _save_training_results(
        model, config, quantile_edges, feature_names, signals,
        training_history, evaluation_results, "simple"
    )
    
    return results


def run_improved_training(config: MLConfig) -> Dict:
    """
    Improved training approach with time-series validation.
    
    Args:
        config: ML configuration
        
    Returns:
        Dictionary with training results
    """
    print("📊 Improved Training Mode")
    
    # 1. Load and prepare data
    df = load_and_validate_data(config)
    
    # 2. Feature engineering
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    feature_names = engineer.get_feature_names()
    
    # 3. Time-series splitting
    splitter = TimeSeriesDataSplitter(config.train_ratio, config.val_ratio, config.test_ratio)
    train_df, val_df, test_df = splitter.split(df_features)
    
    print(f"📊 Train: {len(train_df)} ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"📊 Val: {len(val_df)} ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"📊 Test: {len(test_df)} ({test_df.index[0]} to {test_df.index[-1]})")
    
    # 4. Generate labels for each split separately (NO LOOK-AHEAD BIAS)
    horizon_bars = int(config.forecast_horizon_hours / bar_size_hours(config.interval))
    
    for df_split in [train_df, val_df, test_df]:
        df_split['future_norm_ret'] = df_split['norm_return'].shift(-horizon_bars)
    
    # Remove rows with NaN future returns
    train_df = train_df.dropna(subset=['future_norm_ret'])
    val_df = val_df.dropna(subset=['future_norm_ret'])  
    test_df = test_df.dropna(subset=['future_norm_ret'])
    
    # Calculate quantile edges from TRAINING DATA ONLY
    train_returns = train_df['future_norm_ret']
    quantile_probs = np.linspace(0, 1, config.n_quantiles + 1)[1:-1]
    quantile_edges = np.quantile(train_returns, quantile_probs)
    
    # Assign labels using training quantiles
    train_df['bin'] = np.digitize(train_df['future_norm_ret'], quantile_edges, right=False)
    val_df['bin'] = np.digitize(val_df['future_norm_ret'], quantile_edges, right=False)
    test_df['bin'] = np.digitize(test_df['future_norm_ret'], quantile_edges, right=False)
    
    # Prepare datasets
    X_train = train_df[feature_names]
    y_train = train_df['bin'].values
    X_val = val_df[feature_names]
    y_val = val_df['bin'].values
    X_test = test_df[feature_names]
    y_test = test_df['bin'].values
    
    # 5. Create datasets
    train_dataset = ReturnDataset(X_train, y_train, normalize=True)
    val_dataset = ReturnDataset(X_val, y_val, normalize=False)
    test_dataset = ReturnDataset(X_test, y_test, normalize=False)
    
    # Use training normalization for validation and test
    val_dataset.mean_ = train_dataset.mean_
    val_dataset.std_ = train_dataset.std_
    test_dataset.mean_ = train_dataset.mean_
    test_dataset.std_ = train_dataset.std_
    
    # 6. Create model (simpler for improved mode)
    model = create_model(len(feature_names), config, "simple")
    
    # 7. Train with validation
    trainer = ModelTrainer(config)
    training_history = trainer.train_with_validation(model, train_dataset, val_dataset)
    
    # 8. Evaluate
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate(model, test_dataset, "Test")
    
    # 9. Generate signals for full dataset
    from .signal_generation import generate_trading_signals
    
    # Combine all data for signal generation
    full_X = pd.concat([X_train, X_val, X_test], axis=0)
    full_y = np.concatenate([y_train, y_val, y_test], axis=0)
    full_dataset = ReturnDataset(full_X, full_y, normalize=False)
    full_dataset.mean_ = train_dataset.mean_
    full_dataset.std_ = train_dataset.std_
    
    signals = generate_trading_signals(model, full_dataset, config, mode="improved")
    signals.index = full_X.index[:len(signals)]
    
    # 10. Save results
    results = _save_training_results(
        model, config, quantile_edges, feature_names, signals,
        training_history, evaluation_results, "improved"
    )
    
    return results


class ModelTrainer:
    """Handles model training with different strategies."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.checkpointer = ModelCheckpoint(config)
    
    def train(self, model: nn.Module, train_dataset: Dataset, 
              val_dataset: Optional[Dataset] = None) -> Dict:
        """Train model with optional validation."""
        
        if val_dataset is not None:
            return self.train_with_validation(model, train_dataset, val_dataset)
        else:
            return self.train_simple(model, train_dataset)
    
    def train_simple(self, model: nn.Module, train_dataset: Dataset) -> Dict:
        """Simple training without validation split."""
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        return self.train_with_validation(model, train_subset, val_subset)
    
    def train_with_validation(self, model: nn.Module, train_dataset: Dataset, 
                             val_dataset: Dataset) -> Dict:
        """Train model with validation monitoring."""
        
        # Setup data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_accuracy = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['learning_rate'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self.checkpointer.save(model, epoch, val_loss, val_accuracy)
                
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
            
            # Progress logging
            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | Acc: {val_accuracy:.3f} | "
                      f"LR: {current_lr:.6f}")
        
        print(f"✅ Training complete. Best val loss: {best_val_loss:.4f}")
        return history
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)
            
            # Check for invalid inputs
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                print("⚠️  Warning: Invalid input detected, skipping batch")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'get_logits'):
                # For MLPClassifier, get logits directly
                outputs = model.get_logits(X_batch)
            else:
                # For SimpleModel, forward returns logits
                outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            
            # Check for training divergence
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                print(f"⚠️  Training divergence detected: loss = {loss.item()}")
                raise RuntimeError("Training diverged")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)
                
                # Forward pass
                if hasattr(model, 'get_logits'):
                    outputs = model.get_logits(X_batch)
                else:
                    outputs = model(X_batch)
                
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy


def _save_training_results(model: nn.Module, config: MLConfig, quantile_edges: np.ndarray,
                          feature_names: List[str], signals: pd.Series,
                          training_history: Dict, evaluation_results: Dict,
                          mode: str) -> Dict:
    """Save training results and return summary."""
    
    output_dir = config.model_cache_dir
    
    # Save model
    model_path = output_dir / f"{config.symbol}_{mode}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'quantile_edges': quantile_edges.tolist(),
        'feature_names': feature_names,
        'input_dim': len(feature_names),
        'training_history': training_history,
        'evaluation_results': evaluation_results
    }, model_path)
    
    # Save signals
    signals_path = config.signals_cache_dir / f"{config.symbol}_{mode}_signals.parquet"
    signals_df = pd.DataFrame({'signal': signals}, index=signals.index)
    signals_df.to_parquet(signals_path)
    
    # Save metadata
    metadata = {
        'config': config.to_dict(),
        'model_info': get_model_info(model),
        'quantile_edges': quantile_edges.tolist(),
        'feature_names': feature_names,
        'signal_distribution': signals.value_counts().to_dict(),
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'mode': mode
    }
    
    metadata_path = output_dir / f"{config.symbol}_{mode}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"💾 Results saved:")
    print(f"   Model: {model_path}")
    print(f"   Signals: {signals_path}")
    print(f"   Metadata: {metadata_path}")
    
    return {
        'model': model,
        'signals': signals,
        'metadata': metadata,
        'quantile_edges': quantile_edges,
        'feature_names': feature_names
    } 