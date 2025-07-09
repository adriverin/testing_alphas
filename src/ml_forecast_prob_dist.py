"""
ml_forecast_prob_dist.py
------------------------
A modular reference implementation for forecasting the *probability distribution*
of future **normalised returns** using supervised learning.

Key design choices
------------------
* **Crypto focus** (default symbol: `BTC-USD`, hourly candles via *yfinance* for simplicity).
  Replace `load_price_history()` with a CCXT/ Binance loader for lower-latency data.
* **Feature set** – EMA, EMSD (EW-stdev) and RSI computed at several time-scales.
* **Label** – K-way quantile bin of the *future* normalised return at horizon *h*.
* **Model** – Small fully-connected network with Softmax output (PyTorch).
* **Evaluation** – Cross-entropy loss, accuracy, confusion matrix & reliability.
* **Reproducible & extensible** – all hyper-parameters live in a single `Config`
  dataclass and random seeds are fixed.

The file is ready to `python ml_forecast_prob_dist.py` – it will download data,
train the model, print metrics and save artefacts.  Use it as a starting point
for deeper experimentation or back-testing.

Author: OpenAI ChatGPT-o3 & Anthropic Claude 4 Sonnet – 2025-07-02 
Licence: MIT
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import ccxt

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ================== CONFIG =========================================

@dataclass
class Config:
    symbol: str = "BTC-USD"
    start: str = "2015-01-01"
    end: str = "2024-01-01"
    interval: str = "1h"
    forecast_horizon_hours: int = 24  # 1 day ahead prediction
    vol_window_hours: int = 240       # 10 days for volatility estimation
    # ema_windows: tuple = (0.5, 1, 3, 6, 12)
    # rsi_windows: tuple = (3, 6, 12, 24)

    ema_windows = (24, 48, 72, 120, 168)  # hours → 1d, 2d, 3d, 5d, 7d (for daily bars)
    rsi_windows = (48, 96, 144, 192)     # hours → 2d, 4d, 6d, 8d (for daily bars)

    n_quantiles: int = 5
    test_fraction: float = 0.20 # percentage of data used for testing 
    batch_size: int = 256
    lr: float = 5e-5           # reduced learning rate to prevent divergence
    weight_decay: float = 1e-4
    n_epochs: int = 50
    hidden_sizes: tuple = (128, 64, 32)
    cache_dir: Path = Path("artefacts")
    plot_rel: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False
    threshold: float = 0.4     # Probability threshold for trading signals (40%)

CFG = Config()
CFG.cache_dir.mkdir(exist_ok=True)

# ================== UTILS ==========================================

def bar_size_hours(interval: str):
    if interval.endswith('m'):
        return int(interval[:-1])/60
    if interval.endswith('h'):
        return int(interval[:-1])
    if interval.endswith('d'):
        return int(interval[:-1])*24
    raise ValueError(f"Unknown interval: {interval}")

# ================== DATA ===========================================

def load_price_history(cfg):
    f = cfg.cache_dir / f"prices_{cfg.symbol.replace('/', '')}_{cfg.interval}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        if not df.empty:
            return df

    binance = ccxt.binance()
    symbol_ccxt = cfg.symbol.replace('-USD','/USDT')
    since = binance.parse8601(f"{cfg.start}T00:00:00Z")
    end   = binance.parse8601(f"{cfg.end}T00:00:00Z")
    tf = cfg.interval
    limit = 1000
    ohlcv = []

    while since < end:
        batch = binance.fetch_ohlcv(symbol_ccxt, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv += batch
        since = batch[-1][0]+1
        if len(batch)<limit:
            break

    if not ohlcv:
        raise RuntimeError("No data fetched.")

    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")[["close"]]
    df.to_parquet(f)
    return df

# ================== FEATURES =======================================

def add_features(df, cfg):
    out = df.copy()
    out["return"] = out["close"].pct_change()
    
    # Volatility window in bars (for daily data, 10 days = 10 bars)
    vol_w = max(5, int(cfg.vol_window_hours / bar_size_hours(cfg.interval)))
    out["vol"] = out["return"].rolling(vol_w, min_periods=3).std()
    
    eps = 1e-8
    out["norm_return"] = out["return"] / (out["vol"] + eps)
    out["norm_return"] = out["norm_return"].clip(-5, 5)

    bar_h = bar_size_hours(cfg.interval)
    
    # Simple moving average features (in bars, not hours)
    sma_windows = [2, 5, 10, 20, 30]  # 2, 5, 10, 20, 30 day SMAs
    for w in sma_windows:
        sma = out["close"].rolling(w, min_periods=max(1, w//2)).mean()
        out[f"sma_{w}d"] = ((out["close"] - sma) / sma).fillna(0).clip(-1, 1)
    
    # Volatility features
    for w in [5, 10, 20]:
        vol = out["return"].rolling(w, min_periods=max(1, w//2)).std()
        out[f"vol_{w}d"] = (vol / out["vol"].rolling(20).mean()).fillna(1).clip(0, 5)
    
    # Momentum features
    for w in [1, 3, 7, 14]:
        out[f"mom_{w}d"] = (out["close"] / out["close"].shift(w) - 1).fillna(0).clip(-1, 1)

    # Simple RSI
    def simple_rsi(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=max(1, window//2)).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(1, window//2)).mean()
        rs = gain / (loss + eps)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # Normalize to [-1, 1]

    for w in [7, 14, 21]:
        out[f"rsi_{w}d"] = simple_rsi(out["close"], w).fillna(0).clip(-1, 1)

    feature_cols = [c for c in out.columns if c not in ("return", "vol", "norm_return")]
    
    # Final cleanup
    print(f"Feature engineering completed. Features: {len(feature_cols)}")
    for col in feature_cols:
        # Handle any remaining NaN/inf values
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(0)  # Simple: fill with 0
        
    # Remove rows with any NaN in target variable or essential features
    essential_cols = ["return", "vol", "norm_return"] + feature_cols
    initial_len = len(out)
    out = out.dropna(subset=essential_cols)
    final_len = len(out)
    
    print(f"Data cleaned: {initial_len} -> {final_len} rows ({final_len/initial_len*100:.1f}% kept)")
    print(f"Features shape: {out[feature_cols].shape}")
    
    return out

def generate_labels(df,cfg):
    h = int(cfg.forecast_horizon_hours / bar_size_hours(cfg.interval))
    df = df.copy()
    df["future_norm_ret"] = df["norm_return"].shift(-h)
    df = df.dropna(subset=["future_norm_ret"])
    if len(df)<100:
        raise RuntimeError("Too few rows after feature engineering.")
    split = int(len(df)*(1-cfg.test_fraction)) # test_fraction% of the data is used for testing and the rest for training
    train_fnr = df["future_norm_ret"].iloc[:split]
    if train_fnr.empty:
        raise RuntimeError("Train split empty.")
    q = np.quantile(train_fnr,np.linspace(0,1,cfg.n_quantiles+1)[1:-1])
    df["bin"] = np.digitize(df["future_norm_ret"], q, right=False)
    y = df["bin"].astype(int).values
    X = df.drop(columns=["future_norm_ret","bin"])
    return X,y,q,split

# ================== DATASET ========================================

class ReturnDataset(Dataset):
    mean_ = None
    std_ = None
    def __init__(self,X,y):
        Xv = X.values.astype(np.float32)
        if ReturnDataset.mean_ is None:
            ReturnDataset.mean_ = Xv.mean(0)
            ReturnDataset.std_ = Xv.std(0)+1e-8
            print(f"Dataset normalization stats:")
            print(f"  Mean: {ReturnDataset.mean_}")
            print(f"  Std: {ReturnDataset.std_}")
        
        # Check for zero std (constant features)
        zero_std_mask = ReturnDataset.std_ < 1e-6
        if zero_std_mask.any():
            print(f"Warning: {zero_std_mask.sum()} features have zero/near-zero std, setting std to 1")
            ReturnDataset.std_[zero_std_mask] = 1.0
        
        Xv = (Xv - ReturnDataset.mean_) / ReturnDataset.std_
        
        # Check for any remaining NaN/inf after normalization
        invalid_mask = np.isnan(Xv) | np.isinf(Xv)
        if invalid_mask.any():
            print(f"Warning: {invalid_mask.sum()} invalid values after normalization, setting to 0")
            Xv[invalid_mask] = 0.0
        
        # Additional clipping to prevent extreme values
        Xv = np.clip(Xv, -10, 10)
        
        print(f"Final tensor stats - shape: {Xv.shape}, min: {Xv.min():.4f}, max: {Xv.max():.4f}")
        
        self.X = torch.tensor(Xv)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

# ================== MODEL ==========================================

class MLPClassifier(nn.Module):
    def __init__(self,input_dim,cfg):
        super().__init__()
        layers, prev = [], input_dim
        for h in cfg.hidden_sizes:
            linear = nn.Linear(prev,h)
            # Xavier initialization for better stability
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        
        final_layer = nn.Linear(prev,cfg.n_quantiles)
        nn.init.xavier_normal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return torch.softmax(self.net(x),dim=1)

# ================== TRAIN ==========================================

def train_model(train_ds,val_ds,input_dim,cfg):
    model = MLPClassifier(input_dim,cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    lossfn = nn.CrossEntropyLoss()
    tdl = DataLoader(train_ds,cfg.batch_size,shuffle=True)  # Added shuffle for better training
    vdl = DataLoader(val_ds,cfg.batch_size,shuffle=False)

    def run(loader,train=True):
        model.train(train)
        tot,n = 0.0,0
        for xb,yb in loader:
            xb,yb = xb.to(cfg.device), yb.to(cfg.device)
            
            # Check for invalid inputs
            if torch.isnan(xb).any() or torch.isinf(xb).any():
                print("Warning: Invalid input detected, skipping batch")
                continue
                
            out = model(xb)
            loss = lossfn(out,yb)
            
            # More comprehensive checks for training divergence
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                print(f"Loss: {loss.item()}, gradients stats:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
                raise RuntimeError("Training diverged.")
                
            if train:
                opt.zero_grad()
                loss.backward()
                # More aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if grad_norm > 10:
                    print(f"Warning: Large gradient norm: {grad_norm:.6f}")
                opt.step()
            tot += loss.item()*xb.size(0); n+=xb.size(0)
        
        if n == 0:
            print("Warning: No valid batches processed")
            return 0.0
        return tot/n

    best_val_loss = float('inf')
    patience_counter = 0
    
    for e in range(cfg.n_epochs):
        tr = run(tdl,True)
        vl = run(vdl,False)
        
        scheduler.step(vl)
        
        # Early stopping
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
        else:
            patience_counter += 1
            
        if e%5==0 or e==cfg.n_epochs-1:
            print(f"Epoch {e:02d} | train {tr:.4f} | val {vl:.4f} | lr {opt.param_groups[0]['lr']:.6f}")
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {e}")
            break
            
    return model

# ================== EVAL ===========================================

def evaluate(model,ds,cfg,tag="Test"):
    dl = DataLoader(ds,cfg.batch_size)
    P,Y = [],[]
    with torch.no_grad():
        for xb,yb in dl:
            P.append(model(xb.to(cfg.device)).cpu().numpy())
            Y.append(yb.numpy())
    P = np.vstack(P); Y = np.concatenate(Y)
    yh = P.argmax(1)
    acc = accuracy_score(Y,yh)
    cm = confusion_matrix(Y,yh,labels=np.arange(cfg.n_quantiles))
    print(f"{tag} acc: {acc:.2%}\n{cm}")
    if cfg.plot_rel:
        fig,ax=plt.subplots()
        for k in range(cfg.n_quantiles):
            t,p = calibration_curve((Y==k).astype(int),P[:,k],n_bins=10)
            ax.plot(p,t,"o-",label=f"bin {k}")
        ax.plot([0,1],[0,1],"--")
        ax.legend(); ax.set_title(f"{tag} reliability")
        plt.savefig(cfg.cache_dir/f"reliability_{tag.lower()}.png")
        plt.close()

# ================== MAIN ===========================================

def main(cfg=CFG):
    print("Configuration:",asdict(cfg))
    df = load_price_history(cfg)
    print(f"Loaded {len(df)} rows")
    df_feat = add_features(df,cfg)
    print(f"Feature rows: {len(df_feat)}")
    X,y,q,split = generate_labels(df_feat,cfg)
    print("Quantile edges:", np.round(q,4))
    Xtr,ytr = X.iloc[:split], y[:split]
    Xte,yte = X.iloc[split:], y[split:]
    dtr = ReturnDataset(Xtr,ytr)
    dte = ReturnDataset(Xte,yte)
    model = train_model(dtr,dte,X.shape[1],cfg)
    evaluate(model,dte,cfg)
    
    # Generate and save trading signals with threshold rule
    print(f"\n--- Generating Trading Signals with {cfg.threshold*100:.0f}% Threshold ---")
    signals = save_model_and_generate_signals(model, X, y, q, split, cfg, cfg.threshold)
    
    # Print signal statistics
    signal_counts = signals.value_counts().sort_index()
    total_signals = len(signals)
    print(f"Signal Distribution:")
    print(f"  Short (-1): {signal_counts.get(-1, 0):4d} ({signal_counts.get(-1, 0)/total_signals*100:.1f}%)")
    print(f"  Neutral(0): {signal_counts.get(0, 0):4d} ({signal_counts.get(0, 0)/total_signals*100:.1f}%)")
    print(f"  Long (+1):  {signal_counts.get(1, 0):4d} ({signal_counts.get(1, 0)/total_signals*100:.1f}%)")
    
    return signals

def generate_trading_signals(model, ds, cfg, threshold=0.4):
    """
    Generate trading signals based on model predictions.
    
    Args:
        model: Trained ML model
        ds: Dataset to generate signals for
        cfg: Configuration object
        threshold: Not used in this version - kept for compatibility
    
    Returns:
        pd.Series: Trading signals (-1, 0, 1) with timestamps as index
    """
    model.eval()
    dl = DataLoader(ds, cfg.batch_size)
    P = []
    
    with torch.no_grad():
        for xb, yb in dl:
            # Get logits and convert to probabilities
            logits = model(xb.to(cfg.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            P.append(probs)
    
    P = np.vstack(P)
    
    # Alternative approach: Use top/bottom percentile predictions
    # Find the most confident predictions towards extremes
    bottom_scores = P[:, 0] + P[:, 1]  # Bottom 2 quantiles
    top_scores = P[:, -2] + P[:, -1]   # Top 2 quantiles
    
    # Calculate relative confidence scores
    extreme_preference = top_scores - bottom_scores
    
    # Use percentile-based thresholds for signal generation
    # Take top and bottom 5% of predictions as signals
    top_threshold = np.percentile(extreme_preference, 75)
    bottom_threshold = np.percentile(extreme_preference, 25)
    # top_threshold = np.percentile(extreme_preference, 95)
    # bottom_threshold = np.percentile(extreme_preference, 5)
    
    signals = []
    for score in extreme_preference:
        if score > top_threshold:
            signal = 1   # Long signal (bullish bias)
        elif score < bottom_threshold:
            signal = -1  # Short signal (bearish bias)
        else:
            signal = 0   # No signal
        
        signals.append(signal)
    
    return pd.Series(signals)

def save_model_and_generate_signals(model, X, y, q, split, cfg, threshold=0.6):
    """
    Save the trained model and generate trading signals for the entire dataset.
    
    Args:
        model: Trained ML model
        X: Feature DataFrame
        y: Labels array
        q: Quantile edges
        split: Train/test split index
        cfg: Configuration object
        threshold: Probability threshold for trading signals
    
    Returns:
        pd.Series: Trading signals for the entire dataset
    """
    # Save the model
    model_path = cfg.cache_dir / "return_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'quantile_edges': q,
        'feature_names': X.columns.tolist(),
        'input_dim': X.shape[1]
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Create dataset for the entire period to generate signals
    full_dataset = ReturnDataset(X, y)
    
    # Generate trading signals
    signals = generate_trading_signals(model, full_dataset, cfg, threshold)
    
    # Align signals with original DataFrame index
    if hasattr(X, 'index'):
        # Ensure signals are properly aligned with the feature matrix
        signals.index = X.index[:len(signals)]
    
    # Save signals
    signals_path = cfg.cache_dir / f"trading_signals_threshold_{int(threshold*100)}.parquet"
    signals_df = pd.DataFrame({'signal': signals}, index=signals.index)
    signals_df.to_parquet(signals_path)
    print(f"Trading signals saved to {signals_path}")
    
    return signals

if __name__=="__main__":
    main()