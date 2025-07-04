#!bin/bash

rm *parquet
rm artefacts/*parquet
#rm artefacts/improved_ml/*

#python src/ml_forecast_prob_dist.py
#python test_999.py

#python run_simple_validation.py

python multi_crypto_ml_training.py
python main.py $1

open reports/interval_reports/alpha998_interval_report.pdf


