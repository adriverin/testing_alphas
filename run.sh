#!bin/bash

rm *parquet
rm artefacts/*parquet

python src/ml_forecast_prob_dist.py
python main.py $1

open reports/interval_reports/alpha999_interval_report.pdf


