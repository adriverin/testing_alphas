#!/usr/bin/env python3
"""
Fix Alpha999 by generating and saving probability files

This script generates the probability files that Alpha999 needs for proper backtesting.
It loads the trained models and generates probabilities for the full dataset.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import ML forecasting modules
from src.ml_forecasting.data_loader import load_and_validate_data, bar_size_hours
from src.ml_forecasting.feature_engineering import engineer_features
from src.ml_forecasting.training import ReturnDataset, generate_labels
from src.ml_forecasting.signal_generation import _get_model_probabilities
from src.ml_forecasting.config import MLConfig


def generate_probabilities_for_asset(asset_symbol, start_date="2023-05-05", end_date="2025-01-01"):
    """Generate and save probabilities for a single asset"""
    
    print(f"📊 Generating probabilities for {asset_symbol}...")
    
    # Check if model exists
    model_path = Path(f"artefacts/models/{asset_symbol}_improved_model.pt")
    if not model_path.exists():
        print(f"❌ Model not found for {asset_symbol}: {model_path}")
        return False
    
    try:
        # Load configuration
        config = MLConfig()
        config.symbol = asset_symbol
        config.start = start_date
        config.end = end_date
        config.device = torch.device('cpu')  # Use CPU for consistency
        
        # Load data
        print(f"   Loading data for {asset_symbol}...")
        raw_data = load_and_validate_data(config)
        
        # Engineer features
        print(f"   Engineering features...")
        features_df, feature_names = engineer_features(raw_data, config)
        
        # Generate labels/targets
        print(f"   Generating labels...")
        X, y, quantile_edges, _ = generate_labels(features_df, config, split_data=False)
        
        # Extract feature columns (exclude non-feature columns)
        feature_cols = [col for col in X.columns if col.startswith(('sma_', 'volatility_', 'momentum_', 'rsi_', 'regime_'))]
        if not feature_cols:
            # Fallback to all columns except known non-feature columns
            exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 'norm_return']
            feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        print(f"   Selected feature columns: {len(feature_cols)}")
        
        # Clean data
        if len(feature_cols) == 0:
            print(f"   ❌ No feature columns found")
            return False
            
        clean_X = X[feature_cols].dropna()
        clean_y = y[:len(clean_X)]  # Match lengths after dropna
        
        print(f"   Feature matrix shape: {clean_X.shape}")
        print(f"   Target shape: {clean_y.shape}")
        
        # Create dataset
        print(f"   Creating dataset...")
        dataset = ReturnDataset(clean_X, clean_y, normalize=False)  # No normalization for inference
        
        # Load trained model
        print(f"   Loading trained model...")
        model_data = torch.load(model_path, map_location=config.device, weights_only=False)
        
        # Check if it's a checkpoint dict or full model
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            # It's a training checkpoint with metadata
            from src.ml_forecasting.models import create_model
            
            # Use saved input_dim if available, otherwise infer from data
            input_dim = model_data.get('input_dim', clean_X.shape[1])
            model = create_model(input_dim, config)
            model.load_state_dict(model_data['model_state_dict'])
        elif isinstance(model_data, dict):
            # It's just a state_dict
            from src.ml_forecasting.models import create_model
            input_dim = clean_X.shape[1]
            model = create_model(input_dim, config)
            model.load_state_dict(model_data)
        else:
            # If it's a full model object
            model = model_data
            
        model.eval()
        
        # Generate probabilities
        print(f"   Generating probabilities...")
        probabilities = _get_model_probabilities(model, dataset, config)
        
        # Create probability DataFrame with proper index
        prob_df = pd.DataFrame(
            probabilities,
            index=clean_X.index,
            columns=[f'prob_quantile_{i}' for i in range(probabilities.shape[1])]
        )
        
        # Save probabilities
        output_path = Path(f"artefacts/signals/{asset_symbol}_improved_probabilities.parquet")
        prob_df.to_parquet(output_path)
        
        print(f"✅ Probabilities saved: {output_path}")
        print(f"   Shape: {prob_df.shape}")
        print(f"   Date range: {prob_df.index.min()} to {prob_df.index.max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating probabilities for {asset_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to generate probabilities for all assets"""
    
    print("🚀 Generating ML Probabilities for Alpha999")
    print("=" * 50)
    
    # Get assets from your current configuration
    assets = ['DOGE-USD', 'PEPE-USD', 'SHIB-USD', 'FLOKI-USD']
    
    success_count = 0
    
    for asset in assets:
        if generate_probabilities_for_asset(asset):
            success_count += 1
        print()
    
    print(f"✅ Successfully generated probabilities for {success_count}/{len(assets)} assets")
    
    if success_count > 0:
        print("\n🎉 Alpha999 should now work properly!")
        print("💡 Try running: python generate_dashboard_data.py --alpha alpha999")
    else:
        print("\n❌ No probabilities were generated. Check the errors above.")


if __name__ == "__main__":
    main() 