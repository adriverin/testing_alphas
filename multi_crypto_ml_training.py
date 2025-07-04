# multi_crypto_ml_training.py
from src.ml_forecast_improved import run_improved_training, ImprovedConfig
import pandas as pd
from pathlib import Path

def train_multi_crypto_models(assets: list, base_config: ImprovedConfig):
    """Train separate ML models for each cryptocurrency."""
    
    print(f"🚀 Training ML models for {len(assets)} cryptocurrencies")
    all_signals = {}
    
    for asset in assets:
        print(f"\n📊 Training model for {asset}...")
        
        # Create asset-specific config
        asset_config = ImprovedConfig(
            symbol=asset,
            start=base_config.start,
            end=base_config.end,
            n_quantiles=base_config.n_quantiles,
            hidden_sizes=base_config.hidden_sizes,
            n_epochs=base_config.n_epochs,
            lr=base_config.lr,
            enable_regime_features=base_config.enable_regime_features
        )
        
        # Train asset-specific model
        results = run_improved_training(asset_config)
        
        # Store signals with asset identifier
        asset_signals = results['signals']
        asset_signals.name = asset
        all_signals[asset] = asset_signals
        
        print(f"✅ {asset} model complete: {asset_signals.value_counts().to_dict()}")
    
    # Combine all signals into MultiIndex DataFrame
    signals_df = pd.DataFrame(all_signals)
    
    # Save combined signals
    output_dir = Path("artefacts/improved_ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_dir / "multi_crypto_signals.parquet")
    
    print(f"💾 Saved combined signals for {len(assets)} assets")
    return signals_df

# Usage
if __name__ == "__main__":
    crypto_assets = ['BTC-USD', 'ETH-USD']
    
    base_config = ImprovedConfig(
        start="2020-01-01",
        end="2023-10-31", 
        n_quantiles=5,
        hidden_sizes=(64, 32, 16),
        n_epochs=30,
        lr=1e-4,
        enable_regime_features=True
    )
    
    train_multi_crypto_models(crypto_assets, base_config)