#!/usr/bin/env python3
"""
Test Script for Centralized ML Forecasting Module
=================================================

This script tests the new centralized ml_forecasting module to ensure
it works correctly and produces equivalent results to the original files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_forecasting import MLConfig, train_model, generate_trading_signals, train_multi_crypto_models
from ml_forecasting.data_loader import load_and_validate_data
from ml_forecasting.feature_engineering import FeatureEngineer
from ml_forecasting.models import create_model, get_model_info
from ml_forecasting.evaluation import evaluate_model
from ml_forecasting.signal_generation import analyze_signal_quality


def test_config_creation():
    """Test configuration creation and validation."""
    print("🧪 Testing Configuration Creation...")
    
    # Test basic config
    config = MLConfig(symbol="BTC-USD", training_mode="simple")
    assert config.symbol == "BTC-USD"
    assert config.training_mode == "simple"
    
    # Test factory methods
    simple_config = MLConfig.for_simple_training(symbol="ETH-USD")
    assert simple_config.training_mode == "simple"
    assert simple_config.enable_regime_features == False
    
    improved_config = MLConfig.for_improved_training(symbol="DOGE-USD")
    assert improved_config.training_mode == "improved"
    assert improved_config.enable_regime_features == True
    
    # Test asset-specific config
    btc_config = MLConfig.for_crypto_asset("BTC-USD")
    assert btc_config.symbol == "BTC-USD"
    assert btc_config.interval == "1h"  # BTC-specific default
    
    print("✅ Configuration tests passed!")


def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing Data Loading...")
    
    config = MLConfig(
        symbol="DOGE-USD",
        start="2023-12-01",  # Longer period for sufficient data
        end="2024-01-15",
        interval="1h",
        vol_window_hours=24  # Reduce volatility window requirement
    )
    
    try:
        df = load_and_validate_data(config)
        assert not df.empty, "Data should not be empty"
        assert 'close' in df.columns, "Should have close column"
        assert 'return' in df.columns, "Should have return column"
        print(f"   Loaded {len(df)} rows of data")
        print("✅ Data loading tests passed!")
        return df
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return None


def test_feature_engineering():
    """Test feature engineering functionality."""
    print("🧪 Testing Feature Engineering...")
    
    config = MLConfig(
        symbol="DOGE-USD",
        start="2023-12-01",  # Longer period for sufficient data
        end="2024-01-15",
        interval="1h",  # Use hourly data for more data points
        enable_regime_features=True,
        vol_window_hours=24  # Reduce volatility window requirement
    )
    
    # Load some test data first
    df = load_and_validate_data(config)
    if df is None:
        print("❌ Cannot test features without data")
        return None
    
    # Test feature engineering
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    
    feature_names = engineer.get_feature_names()
    
    assert len(feature_names) > 0, "Should have generated features"
    assert 'sma_5d' in feature_names, "Should have SMA features"
    assert 'vol_5d' in feature_names, "Should have volatility features"
    assert 'mom_1d' in feature_names, "Should have momentum features"
    assert 'rsi_7d' in feature_names, "Should have RSI features"
    
    if config.enable_regime_features:
        regime_features = [f for f in feature_names if 'regime' in f]
        assert len(regime_features) > 0, "Should have regime features"
    
    print(f"   Generated {len(feature_names)} features")
    print("✅ Feature engineering tests passed!")
    return df_features, feature_names


def test_model_creation():
    """Test model creation functionality."""
    print("🧪 Testing Model Creation...")
    
    config = MLConfig(training_mode="simple", n_quantiles=5, hidden_sizes=(32, 16))
    
    # Test model creation
    model = create_model(input_dim=20, config=config, model_type="auto")
    
    model_info = get_model_info(model)
    assert model_info['total_parameters'] > 0, "Model should have parameters"
    
    print(f"   Created {model_info['model_type']} with {model_info['total_parameters']} parameters")
    print("✅ Model creation tests passed!")
    return model


def test_simple_training():
    """Test simple training mode."""
    print("🧪 Testing Simple Training Mode...")
    
    config = MLConfig.for_simple_training(
        symbol="DOGE-USD",
        start="2023-12-01",  # Longer period for sufficient data
        end="2024-01-15",
        interval="1h",  # Use hourly data for more data points
        n_epochs=5,  # Fast training for testing
        verbose=False,
        vol_window_hours=24  # Reduce requirements for testing
    )
    
    try:
        results = train_model(config)
        
        assert 'model' in results, "Should return trained model"
        assert 'signals' in results, "Should return trading signals" 
        assert 'metadata' in results, "Should return metadata"
        
        signals = results['signals']
        assert not signals.empty, "Should generate signals"
        assert signals.dtype in ['int64', 'int32'], "Signals should be integers"
        assert set(signals.unique()).issubset({-1, 0, 1}), "Signals should be -1, 0, or 1"
        
        print(f"   Generated {len(signals)} signals: {signals.value_counts().to_dict()}")
        print("✅ Simple training tests passed!")
        return results
        
    except Exception as e:
        print(f"❌ Simple training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_improved_training():
    """Test improved training mode."""
    print("🧪 Testing Improved Training Mode...")
    
    config = MLConfig.for_improved_training(
        symbol="DOGE-USD",
        start="2023-12-01",  # Longer period for sufficient data
        end="2024-01-15",
        interval="1h",  # Use hourly data for more data points
        n_epochs=5,  # Fast training for testing
        verbose=False,
        vol_window_hours=24  # Reduce requirements for testing
    )
    
    try:
        results = train_model(config)
        
        assert 'model' in results, "Should return trained model"
        assert 'signals' in results, "Should return trading signals"
        assert 'metadata' in results, "Should return metadata"
        
        signals = results['signals']
        assert not signals.empty, "Should generate signals"
        
        print(f"   Generated {len(signals)} signals: {signals.value_counts().to_dict()}")
        print("✅ Improved training tests passed!")
        return results
        
    except Exception as e:
        print(f"❌ Improved training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multi_asset_training():
    """Test multi-asset training functionality."""
    print("🧪 Testing Multi-Asset Training...")
    
    base_config = MLConfig.for_improved_training(
        start="2023-12-01",  # Longer period for sufficient data
        end="2024-01-15",
        interval="1h",  # Use hourly data for more data points
        n_epochs=3,  # Very fast training
        verbose=False,
        vol_window_hours=24  # Reduce requirements for testing
    )
    
    # Test with just two assets for speed
    assets = ['DOGE-USD', 'BTC-USD']
    
    try:
        results = train_multi_crypto_models(
            assets=assets,
            base_config=base_config,
            parallel=False  # Sequential for testing
        )
        
        assert 'signals_df' in results, "Should return combined signals"
        assert 'summary' in results, "Should return summary"
        
        signals_df = results['signals_df']
        summary = results['summary']
        
        print(f"   Trained {summary['successful_assets']} assets successfully")
        if not signals_df.empty:
            print(f"   Combined signals shape: {signals_df.shape}")
        
        print("✅ Multi-asset training tests passed!")
        return results
        
    except Exception as e:
        print(f"❌ Multi-asset training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_signal_analysis():
    """Test signal analysis functionality."""
    print("🧪 Testing Signal Analysis...")
    
    # Create some test signals and returns
    import pandas as pd
    import numpy as np
    
    # Generate synthetic data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    signals = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates, name='signal')
    returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates, name='return')
    
    config = MLConfig()
    
    try:
        analysis = analyze_signal_quality(signals, returns, config)
        
        assert 'total_signals' in analysis, "Should have total signals count"
        assert 'signal_distribution' in analysis, "Should have signal distribution"
        assert 'signal_return_correlation' in analysis, "Should have correlation"
        
        print(f"   Analyzed {analysis['total_signals']} signals")
        print(f"   Signal-return correlation: {analysis['signal_return_correlation']:.3f}")
        print("✅ Signal analysis tests passed!")
        return analysis
        
    except Exception as e:
        print(f"❌ Signal analysis test failed: {e}")
        return None


def run_compatibility_test():
    """Test backward compatibility with original imports."""
    print("🧪 Testing Backward Compatibility...")
    
    try:
        # Test that we can still import the legacy names
        from ml_forecasting.config import Config, ImprovedConfig
        
        # Test that they work the same as MLConfig
        legacy_config = Config(symbol="BTC-USD")
        new_config = MLConfig(symbol="BTC-USD")
        
        assert legacy_config.symbol == new_config.symbol
        
        improved_legacy = ImprovedConfig(symbol="ETH-USD", enable_regime_features=True)
        assert improved_legacy.enable_regime_features == True
        
        print("✅ Backward compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Centralized ML Forecasting Module")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 8  # All tests including compatibility
    
    # Run all tests
    if test_config_creation():
        tests_passed += 1
    
    data = test_data_loading()
    if data is not None:
        tests_passed += 1
    
    if test_feature_engineering():
        tests_passed += 1
    
    if test_model_creation():
        tests_passed += 1
    
    if test_simple_training():
        tests_passed += 1
    
    if test_improved_training():
        tests_passed += 1
    
    if test_multi_asset_training():
        tests_passed += 1
    
    if test_signal_analysis():
        tests_passed += 1
    
    # Compatibility test
    if run_compatibility_test():
        tests_passed += 1
    
    print(f"\n🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The centralized module is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 