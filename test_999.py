# Custom configuration example
from src.ml_forecast_prob_dist import Config, main

# High-frequency setup
config_hourly = Config(
    symbol="BTC-USD",
    interval="1h",
    forecast_horizon_hours=4,
    n_epochs=100,
    hidden_sizes=(256, 128, 64)
)

# Conservative long-term setup  
config_daily = Config(
    symbol="BTC-USD", 
    interval="1d",
    forecast_horizon_hours=24,  # 1 week
    vol_window_hours=120,        # 30 days
    n_quantiles=5,               # Simpler classification
    n_epochs=100,
    hidden_sizes=(256, 128, 64),
)

# Run with custom config
signals = main(config_daily)