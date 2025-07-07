#!bin/bash

rm *parquet
rm artefacts/*parquet
#rm artefacts/improved_ml/*

#python src/ml_forecast_prob_dist.py
#python test_999.py

#python run_simple_validation.py

python multi_crypto_ml_training.py
python main.py $1 $2 $3 # 1: type of run; 2:--stop-loss; 3: stop loss pct 

open reports/interval_reports/alpha998_interval_report.pdf


