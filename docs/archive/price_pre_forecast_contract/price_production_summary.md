# price/production

- Registry key: `price_production`
- Runs: 5
- Exported: 2026-07-13T18:59:07+00:00
- Contract disposition: pre-forecast-contract, leaky/non-comparable

## Metrics

- `metrics.mae` best: 10.346749; median: 11.148329
- `metrics.rmse` best: 15.547212; median: 16.572071

## Top 5 Runs

```text
                          run_id experiment_id   status                       start_time                         end_time  metrics.mae  metrics.rmse  metrics.pi_coverage tags.stage                                        tags.feature_version tags.feature_contract tags.model_class tags.archived                         tags.archive_reason
6d5c0af3364d46419e58560e4df937a8             7 FINISHED 2026-05-27 14:19:26.734000+00:00 2026-05-27 14:19:26.850000+00:00    10.346749     15.566135                  NaN production                                                        slim            prog_leaky   WeightEnsemble          true pre-forecast-contract; leaky/non-comparable
63128d07caf0403e91d1ded6319782b1             7 FINISHED 2026-05-27 13:12:39.437000+00:00 2026-05-27 13:12:39.555000+00:00    10.447853     15.547212                  NaN production                                                        slim            prog_leaky   WeightEnsemble          true pre-forecast-contract; leaky/non-comparable
e64d499dd99742b48208ff712ef52d82             7 FINISHED 2026-06-24 09:27:59.994000+00:00 2026-06-24 09:28:00.217000+00:00    11.148329     18.094039             0.900463 production fs_rfecv_optimum+fs_shap_top247+fs_shap_top66+fs_shap_top90            prog_leaky   WeightEnsemble          true pre-forecast-contract; leaky/non-comparable
2fbd74ececed4dde93e7bd763a97400b             7 FINISHED 2026-05-29 02:04:59.401000+00:00 2026-05-29 02:04:59.632000+00:00    11.239245     17.999458             0.900463 production                                                         max            prog_leaky   WeightEnsemble          true pre-forecast-contract; leaky/non-comparable
fb93b9e5680e4472971b024dbe4cfd03             7 FINISHED 2026-05-27 12:16:58.876000+00:00 2026-05-27 12:16:59.083000+00:00    12.290521     16.572071                  NaN production                                                        slim            prog_leaky   WeightEnsemble          true pre-forecast-contract; leaky/non-comparable
```
