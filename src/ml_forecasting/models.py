"""
Neural Network Models
=====================

Centralized model architectures for ML forecasting.
Combines and enhances MLPClassifier and SimpleModel from original files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .config import MLConfig


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for classification with configurable architecture.
    """
    
    def __init__(self, input_dim: int, config: MLConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = config.n_quantiles
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_sizes:
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            self._init_weights(linear)
            layers.append(linear)
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout for regularization
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        output_layer = nn.Linear(prev_dim, self.output_dim)
        self._init_weights(output_layer)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
        
        # Store layer information for analysis
        self.layer_dims = [input_dim] + list(config.hidden_sizes) + [self.output_dim]
    
    def _init_weights(self, layer: nn.Linear):
        """Initialize weights using Xavier normal initialization."""
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with softmax output for probability distribution.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability distribution over quantiles (batch_size, n_quantiles)
        """
        logits = self.network(x)
        return F.softmax(logits, dim=1)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits without softmax (useful for training)."""
        return self.network(x)
    
    def get_layer_outputs(self, x: torch.Tensor) -> list:
        """Get outputs from each layer for analysis."""
        outputs = []
        current_x = x
        
        for layer in self.network:
            current_x = layer(current_x)
            if isinstance(layer, (nn.Linear, nn.ReLU)):
                outputs.append(current_x.clone())
        
        return outputs


class SimpleModel(nn.Module):
    """
    Simplified model architecture designed to prevent overfitting.
    Enhanced version of the SimpleModel from improved training.
    """
    
    def __init__(self, input_dim: int, config: MLConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = config.n_quantiles
        
        # Very simple architecture
        hidden_dim = min(32, max(8, input_dim // 4))  # Adaptive hidden size
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate if config.dropout_rate > 0 else 0.5)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (no softmax for training)."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved robustness.
    """
    
    def __init__(self, input_dim: int, config: MLConfig):
        super().__init__()
        self.config = config
        self.n_models = config.n_ensemble_models
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            self._create_model(input_dim, config, i) 
            for i in range(self.n_models)
        ])
    
    def _create_model(self, input_dim: int, config: MLConfig, model_idx: int):
        """Create individual model for ensemble with slight variations."""
        if config.training_mode == "simple":
            # Use SimpleModel for simple mode
            return SimpleModel(input_dim, config)
        else:
            # Use MLPClassifier for improved mode
            return MLPClassifier(input_dim, config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging predictions from all models."""
        predictions = []
        
        for model in self.models:
            if isinstance(model, SimpleModel):
                pred = F.softmax(model(x), dim=1)
            else:
                pred = model(x)  # MLPClassifier already returns softmax
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    
    def get_individual_predictions(self, x: torch.Tensor) -> list:
        """Get predictions from each individual model."""
        predictions = []
        
        for model in self.models:
            if isinstance(model, SimpleModel):
                pred = F.softmax(model(x), dim=1)
            else:
                pred = model(x)
            predictions.append(pred)
        
        return predictions


def create_model(input_dim: int, config: MLConfig, model_type: str = "auto") -> nn.Module:
    """
    Factory function to create appropriate model based on configuration.
    
    Args:
        input_dim: Number of input features
        config: ML configuration
        model_type: Type of model to create ("auto", "mlp", "simple", "ensemble")
        
    Returns:
        Initialized neural network model
    """
    if model_type == "auto":
        # Auto-select based on training mode
        if config.training_mode == "simple":
            model_type = "simple"
        else:
            model_type = "mlp"
    
    if model_type == "mlp":
        model = MLPClassifier(input_dim, config)
    elif model_type == "simple":
        model = SimpleModel(input_dim, config)
    elif model_type == "ensemble":
        model = EnsembleModel(input_dim, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device
    model = model.to(config.device)
    
    # Set random seed for reproducible initialization
    torch.manual_seed(config.random_seed)
    
    print(f"📱 Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: Neural network model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_type': type(model).__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': next(model.parameters()).device,
    }
    
    # Add layer information if available
    if hasattr(model, 'layer_dims'):
        info['layer_dimensions'] = model.layer_dims
    
    if hasattr(model, 'n_models'):
        info['ensemble_size'] = model.n_models
    
    return info


class ModelCheckpoint:
    """
    Utility class for saving and loading model checkpoints.
    """
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.checkpoint_dir = config.model_cache_dir
    
    def save(self, model: nn.Module, epoch: int, loss: float, 
             accuracy: float, metadata: dict = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Training epoch
            loss: Validation loss
            accuracy: Validation accuracy
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            self.checkpoint_dir / 
            f"{self.config.symbol}_{self.config.training_mode}_epoch_{epoch:03d}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': self.config.to_dict(),
            'model_info': get_model_info(model),
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    def load(self, checkpoint_path: str, model: nn.Module) -> dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'accuracy': checkpoint['accuracy'],
            'metadata': checkpoint.get('metadata', {})
        }
    
    def find_best_checkpoint(self, metric: str = "accuracy") -> Optional[str]:
        """
        Find the best checkpoint based on specified metric.
        
        Args:
            metric: Metric to optimize ("accuracy" or "loss")
            
        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        pattern = f"{self.config.symbol}_{self.config.training_mode}_epoch_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('-inf') if metric == "accuracy" else float('inf')
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                value = checkpoint[metric]
                
                if (metric == "accuracy" and value > best_value) or \
                   (metric == "loss" and value < best_value):
                    best_value = value
                    best_checkpoint = str(checkpoint_path)
            except Exception as e:
                print(f"⚠️  Error reading checkpoint {checkpoint_path}: {e}")
        
        return best_checkpoint 