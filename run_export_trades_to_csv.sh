#!bin/bash

# Export alpha999 trades to Excel
python export_trades_to_csv/export_trades.py alpha999 # --stop-loss -5.0 

# Export all alphas to CSV
#python export_trades_to_csv/export_trades.py --all-alphas --format csv

# Open the latest trades file
python export_trades_to_csv/view_trades.py alpha999 
