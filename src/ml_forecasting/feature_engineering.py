"""
Feature Engineering
===================

Centralized feature engineering for ML forecasting.
Combines and enhances functionality from original files.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

from .config import MLConfig
from .data_loader import bar_size_hours


class FeatureEngineer:
    """
    Unified feature engineering class combining basic and regime features.
    """
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_names: List[str] = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df: DataFrame with 'close' and 'return' columns
            
        Returns:
            DataFrame with engineered features
        """
        print("🛠️  Starting feature engineering...")
        
        # Start with basic features
        df_features = self._add_basic_features(df.copy())
        
        # Add regime features if enabled
        if self.config.enable_regime_features:
            df_features = self._add_regime_features(df_features)
        
        # Add volatility and normalization
        df_features = self._add_volatility_features(df_features)
        
        # Final cleanup
        df_features = self._cleanup_features(df_features)
        
        # Store feature names for later use
        self.feature_names = [col for col in df_features.columns 
                             if col not in ['return', 'vol', 'norm_return', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Feature engineering complete: {len(self.feature_names)} features")
        return df_features
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        print("   Adding basic technical features...")
        
        # Simple Moving Averages
        for window in self.config.sma_windows:
            sma = df['close'].rolling(window, min_periods=max(1, window//2)).mean()
            df[f'sma_{window}d'] = ((df['close'] - sma) / sma).fillna(0).clip(-1, 1)
        
        # Volatility ratios
        for window in self.config.volatility_windows:
            vol = df['return'].rolling(window, min_periods=max(1, window//2)).std()
            rolling_vol_mean = df['return'].rolling(20, min_periods=10).std()
            df[f'vol_{window}d'] = (vol / rolling_vol_mean).fillna(1).clip(0, 5)
        
        # Momentum features
        for window in self.config.momentum_windows:
            momentum = (df['close'] / df['close'].shift(window) - 1).fillna(0).clip(-1, 1)
            df[f'mom_{window}d'] = momentum
        
        # RSI features
        for window in self.config.rsi_windows:
            df[f'rsi_{window}d'] = self._calculate_rsi(df['close'], window)
        
        # Price position within recent range
        for window in [10, 20, 50]:
            rolling_min = df['close'].rolling(window, min_periods=max(1, window//2)).min()
            rolling_max = df['close'].rolling(window, min_periods=max(1, window//2)).max()
            price_position = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            df[f'price_pos_{window}d'] = price_position.fillna(0.5).clip(0, 1)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        print("   Adding regime detection features...")
        
        # Volatility regime
        vol_window = self.config.volatility_regime_window
        rolling_vol = df['return'].rolling(vol_window, min_periods=max(1, vol_window//2)).std()
        vol_threshold = rolling_vol.rolling(vol_window * 2, min_periods=vol_window).median()
        df['vol_regime'] = (rolling_vol > vol_threshold).astype(float).fillna(0)
        
        # Trend regime (based on moving average slopes)
        for ma_window in [5, 20]:
            ma = df['close'].rolling(ma_window).mean()
            ma_slope = (ma / ma.shift(ma_window//2) - 1).fillna(0)
            trend_strength = abs(ma_slope)
            trend_threshold = trend_strength.rolling(60, min_periods=30).quantile(0.7)
            df[f'trend_regime_{ma_window}d'] = (trend_strength > trend_threshold).astype(float).fillna(0)
        
        # Momentum regime (momentum persistence)
        momentum_1d = df['return']
        momentum_5d = df['close'].pct_change(5)
        momentum_persistence = (momentum_1d * momentum_5d > 0).rolling(10, min_periods=5).mean()
        df['momentum_regime'] = momentum_persistence.fillna(0.5)
        
        # Volume regime (if volume data available)
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20, min_periods=10).mean()
            vol_threshold = avg_volume.rolling(60, min_periods=30).quantile(0.7)
            df['volume_regime'] = (df['volume'] > vol_threshold).astype(float).fillna(0)
        
        # Volatility clustering
        vol_persistence = (df['return'].abs() > df['return'].abs().rolling(20).mean()).rolling(5).mean()
        df['vol_clustering'] = vol_persistence.fillna(0.2)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility estimation and normalized returns."""
        print("   Adding volatility features...")
        
        # Calculate volatility window in bars
        vol_window_bars = max(5, int(self.config.vol_window_hours / bar_size_hours(self.config.interval)))
        
        # Rolling volatility
        df['vol'] = df['return'].rolling(vol_window_bars, min_periods=3).std()
        
        # Normalized returns
        eps = 1e-8
        df['norm_return'] = df['return'] / (df['vol'] + eps)
        df['norm_return'] = df['norm_return'].clip(-5, 5).fillna(0)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['vol'].rolling(20, min_periods=10).std().fillna(0)
        
        # Realized volatility (different windows)
        for window in [5, 10, 20]:
            realized_vol = df['return'].rolling(window).std()
            df[f'realized_vol_{window}d'] = realized_vol.fillna(0)
        
        # Volatility ratio (short vs long term)
        short_vol = df['return'].rolling(5).std()
        long_vol = df['return'].rolling(20).std()
        df['vol_ratio'] = (short_vol / (long_vol + eps)).fillna(1).clip(0, 5)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI with proper normalization."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=max(1, window//2)).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(1, window//2)).mean()
        
        eps = 1e-8
        rs = gain / (loss + eps)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to [-1, 1] range
        return ((rsi - 50) / 50).fillna(0).clip(-1, 1)
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of engineered features."""
        print("   Cleaning up features...")
        
        # Get feature columns (exclude basic price/return columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['return', 'vol', 'norm_return', 'open', 'high', 'low', 'close', 'volume']]
        
        initial_len = len(df)
        
        # Handle infinite values
        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            if 'regime' in col or 'clustering' in col:
                df[col] = df[col].fillna(0)  # Regime features default to 0
            elif 'vol' in col:
                df[col] = df[col].fillna(df[col].median())  # Volatility features use median
            elif 'price_pos' in col:
                df[col] = df[col].fillna(0.5)  # Price position defaults to middle
            else:
                df[col] = df[col].fillna(0)  # Most features default to 0
        
        # Remove rows where essential features are still NaN
        essential_cols = ['return', 'vol', 'norm_return']
        df = df.dropna(subset=essential_cols)
        
        final_len = len(df)
        
        if final_len < initial_len:
            print(f"   Removed {initial_len - final_len} rows with missing essential data ({(initial_len - final_len)/initial_len*100:.1f}%)")
        
        # Final validation
        for col in feature_cols:
            if df[col].isna().any():
                print(f"⚠️  Warning: {col} still has NaN values, filling with 0")
                df[col] = df[col].fillna(0)
            
            if np.isinf(df[col]).any():
                print(f"⚠️  Warning: {col} still has infinite values, clipping")
                df[col] = df[col].clip(-1e6, 1e6)
        
        print(f"   Final feature matrix shape: {df[feature_cols].shape}")
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance_groups(self) -> dict:
        """Group features by type for analysis."""
        groups = {
            'sma': [f for f in self.feature_names if f.startswith('sma_')],
            'volatility': [f for f in self.feature_names if 'vol' in f and not 'regime' in f],
            'momentum': [f for f in self.feature_names if f.startswith('mom_')],
            'rsi': [f for f in self.feature_names if f.startswith('rsi_')],
            'price_position': [f for f in self.feature_names if f.startswith('price_pos_')],
            'regime': [f for f in self.feature_names if 'regime' in f or 'clustering' in f],
        }
        return groups


def engineer_features(df: pd.DataFrame, config: MLConfig) -> tuple[pd.DataFrame, List[str]]:
    """
    Convenience function for feature engineering.
    
    Args:
        df: Input DataFrame with price data
        config: ML configuration
        
    Returns:
        Tuple of (engineered DataFrame, list of feature names)
    """
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    feature_names = engineer.get_feature_names()
    
    return df_features, feature_names


# Backward compatibility functions
def add_features(df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
    """Legacy function for backward compatibility with original code."""
    engineer = FeatureEngineer(config)
    return engineer.engineer_features(df) 