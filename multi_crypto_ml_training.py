"""
Multi-Cryptocurrency ML Training Script (Updated)
=================================================

Updated to use the new centralized ml_forecasting module.

This script now leverages the enhanced ml_forecasting module which provides:
- Unified configuration management
- Better error handling and parallel training support
- Enhanced signal analysis and model evaluation
- Backward compatibility with the original interface

For the legacy interface, see MIGRATION_GUIDE.md
"""

# Import from the new centralized module
from src.ml_forecasting import MLConfig, train_multi_crypto_models as train_models_new
import pandas as pd
from pathlib import Path

def train_multi_crypto_models(assets: list, base_config):
    """
    Legacy wrapper function for backward compatibility.
    
    This function maintains the original interface while using the new
    centralized ml_forecasting module under the hood.
    """
    
    print("🔄 Using legacy interface with new centralized module")
    print("ℹ️  Consider migrating to the new interface - see MIGRATION_GUIDE.md")
    
    # Convert legacy config to new MLConfig if needed
    if hasattr(base_config, 'enable_regime_features'):
        # Convert ImprovedConfig to MLConfig
        new_config = MLConfig.for_improved_training(
            start=base_config.start,
            end=base_config.end,
            interval=getattr(base_config, 'interval', '4h'),
            forecast_horizon_hours=getattr(base_config, 'forecast_horizon_hours', 6),
            vol_window_hours=getattr(base_config, 'vol_window_hours', 60),
            n_quantiles=base_config.n_quantiles,
            hidden_sizes=base_config.hidden_sizes,
            n_epochs=base_config.n_epochs,
            lr=base_config.lr,
            enable_regime_features=base_config.enable_regime_features
        )
    else:
        # Assume it's already an MLConfig
        new_config = base_config
    
    # Use the new centralized function
    results = train_models_new(
        assets=assets,
        base_config=new_config,
        parallel=False  # Keep sequential for backward compatibility
    )
    
    # Extract signals_df for backward compatibility
    signals_df = results['signals_df']
    
    # Save to legacy location for compatibility
    output_dir = Path("artefacts/improved_ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_dir / "multi_crypto_signals.parquet")
    
    print(f"💾 Saved to legacy location: {output_dir / 'multi_crypto_signals.parquet'}")
    print(f"📊 New enhanced results also available in: artefacts/multi_asset/")
    
    return signals_df

# Usage
if __name__ == "__main__":
    print("🚀 Multi-Crypto ML Training Script")
    print("=" * 50)
    
    # # Example 1: Using legacy interface (for backward compatibility)
    # print("\n📊 Example 1: Legacy Interface")
    # crypto_assets_legacy = ['DOGE-USD']

    # # Create config using the new MLConfig but compatible parameters
    # base_config_legacy = MLConfig.for_improved_training(
    #     start="2020-01-01",
    #     end="2024-01-15",  # Shorter period for demo
    #     interval="4h",
    #     forecast_horizon_hours=6,
    #     vol_window_hours=60,
    #     n_quantiles=5,
    #     hidden_sizes=(64, 32, 16),
    #     n_epochs=10,  # Reduced for demo
    #     lr=1e-4,
    #     enable_regime_features=True
    # )
    
    # # Use legacy wrapper
    # signals_df_legacy = train_multi_crypto_models(crypto_assets_legacy, base_config_legacy)
    # print(f"✅ Legacy interface complete: {signals_df_legacy.shape}")



    crypto_assets_new = ['ETH-USD']
    
    base_config_new = MLConfig.for_improved_training(
        start="2018-01-01", 
        end="2024-12-31",  
        interval="1h",
        forecast_horizon_hours=6,
        vol_window_hours=24,
        n_quantiles=7,
        hidden_sizes=(64, 32, 16),
        n_epochs=100,  
        lr=5e-5,
        enable_regime_features=True,
        verbose=False,
        sma_windows=(5, 10, 20, 30, 40, 50),
        volatility_windows=(3, 5, 10, 20, 30),
        momentum_windows=(3, 7, 14, 21, 30, 40),
        rsi_windows=(3, 7, 14, 21, 30, 40, 50),
        signal_percentiles=(2, 98)
        # signal_percentiles=(3, 97)
    )

    # FOLLOWING PREDICTS 36% OF QUINTILES??? STILL LEAD TO BAD RETURNS SOMEHOW
    # crypto_assets_new = ['PEPE-USD']
    
    # base_config_new = MLConfig.for_improved_training(
    #     start="2023-06-01", 
    #     end="2024-12-31",  
    #     interval="1h",
    #     forecast_horizon_hours=2,
    #     vol_window_hours=24,
    #     n_quantiles=5,
    #     hidden_sizes=(32, 16, 8),
    #     n_epochs=100,  # Reduced for demo
    #     lr=5e-5,
    #     enable_regime_features=True,
    #     verbose=False
    # )
    
    # Use new centralized interface directly
    results_new = train_models_new(
        assets=crypto_assets_new,
        base_config=base_config_new,
        parallel=False,  # Enable parallel training
        max_workers=1
    )
    
    print(f"✅ New interface complete:")
    print(f"   Signals shape: {results_new['signals_df'].shape}")
    print(f"   Successful assets: {results_new['summary']['successful_assets']}")
    print(f"   Training time: {results_new['summary']['total_training_time']:.1f}s")
    
    # print("\n🎉 Multi-crypto training demonstration complete!")
    # print("📋 Check MIGRATION_GUIDE.md for full migration instructions")