# Forecast Dataset Audit

- Rows: 100,824
- Forecast NaNs: {'forecast_load': 0, 'forecast_gen_wind_on': 0, 'forecast_gen_wind_off': 0, 'forecast_gen_solar': 0, 'forecast_gen_wind_pv': 0, 'forecast_gen_total': 0, 'forecast_gen_other': 0, 'forecast_residual_load': 0}
- Source counts: {'own': 36842, 'smard': 63574, 'actual': 408, 'missing': 0}
- First own forecast timestamp: `2022-01-15T01:00:00`
- Own 2022+ residual identity max abs error: 0
- Holdout source counts: {'own': 54, 'smard': 2106, 'actual': 0, 'missing': 0}
- Dataset schema banned-token hits: {}

## Source Counts By Year

```text
col_0   own  smard  actual  missing
row_0                              
2014      0     24       0        0
2015      0   8760       0        0
2016      0   8784       0        0
2017      0   8760       0        0
2018      0   8352     408        0
2019      0   8760       0        0
2020      0   8784       0        0
2021      0   8760       0        0
2022   8423    337       0        0
2023   8760      0       0        0
2024   8784      0       0        0
2025   8760      0       0        0
2026   2115   2253       0        0
```

## prog_residual Comparison

```json
{
  "rows_compared_2022_plus": 39432,
  "mean_abs_diff": 2918.376907826674,
  "max_abs_diff": 37435.739527690996,
  "rows_different_gt_1mw": 36833
}
```
