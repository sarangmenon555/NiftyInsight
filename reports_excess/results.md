# Results

## Model quality

| split | prediction | rows | base_rate | auc | average_precision | precision_top_1pct |
|---|---|---|---|---|---|---|
| valid | pred_dt | 165530 | 0.2608 | 0.5181 | 0.2736 | 0.3033 |
| valid | pred_rf | 165530 | 0.2608 | 0.5551 | 0.2983 | 0.3674 |
| valid | pred_xgb | 165530 | 0.2608 | 0.5475 | 0.2930 | 0.3704 |
| valid | pred_ens | 165530 | 0.2608 | 0.5544 | 0.2979 | 0.3668 |
| test | pred_dt | 471437 | 0.2743 | 0.5296 | 0.2945 | 0.3080 |
| test | pred_rf | 471437 | 0.2743 | 0.5538 | 0.3133 | 0.3908 |
| test | pred_xgb | 471437 | 0.2743 | 0.5504 | 0.3116 | 0.4048 |
| test | pred_ens | 471437 | 0.2743 | 0.5544 | 0.3141 | 0.4031 |

## Vector simulation (approximate)

| prediction | strategy | trades | precision | avg_net_return_per_trade | approx_cagr | approx_sharpe | approx_max_drawdown |
|---|---|---|---|---|---|---|---|
| pred_dt | sim1 | 61392 | 0.3255 | 0.0014 | -0.0647 | -1.1683 | -0.3401 |
| pred_dt | sim2 | 3690 | 0.3252 | 0.0002 | 0.0006 | -0.4561 | -0.3089 |
| pred_dt | sim3 | 2820 | 0.3209 | -0.0008 | -0.0377 | -1.0037 | -0.3494 |
| pred_rf | sim1 | 84999 | 0.3290 | 0.0047 | -0.0166 | -0.7184 | -0.2899 |
| pred_rf | sim2 | 3690 | 0.3889 | 0.0049 | 0.2625 | 1.0299 | -0.3515 |
| pred_rf | sim3 | 2820 | 0.3957 | 0.0056 | 0.2280 | 0.9978 | -0.2290 |
| pred_xgb | sim1 | 78457 | 0.3291 | 0.0069 | -0.0043 | -0.6021 | -0.2832 |
| pred_xgb | sim2 | 3690 | 0.3626 | 0.0003 | -0.0014 | -0.3137 | -0.4033 |
| pred_xgb | sim3 | 2820 | 0.3688 | 0.0006 | 0.0128 | -0.3152 | -0.2864 |
| pred_ens | sim1 | 85892 | 0.3301 | 0.0058 | -0.0082 | -0.6428 | -0.2839 |
| pred_ens | sim2 | 3690 | 0.3824 | 0.0034 | 0.1705 | 0.5945 | -0.3565 |
| pred_ens | sim3 | 2820 | 0.3918 | 0.0043 | 0.1711 | 0.6821 | -0.2204 |

## Exact simulation versus benchmarks

| key | total_return | cagr | volatility | sharpe | sortino | max_drawdown | calmar | trades | win_rate | avg_exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| pred_dt__sim1_equal_weight | -0.0835 | -0.0298 | 0.2085 | -0.3489 | -0.4483 | -0.4149 | -0.0717 | 1479.0000 | 0.4638 | 0.7915 |
| pred_dt__sim2_stop_weighted | -0.4826 | -0.2041 | 0.2033 | -1.3176 | -1.7622 | -0.6215 | -0.3285 | 1398.0000 | 0.4435 | 0.7768 |
| pred_dt__sim3_regime_concentrated | -0.2374 | -0.0896 | 0.1886 | -0.7381 | -0.9292 | -0.4028 | -0.2225 | 597.0000 | 0.4539 | 0.6098 |
| pred_rf__sim1_equal_weight | -0.0587 | -0.0208 | 0.3145 | -0.1145 | -0.1609 | -0.4258 | -0.0487 | 1479.0000 | 0.4936 | 0.7904 |
| pred_rf__sim2_stop_weighted | -0.1617 | -0.0593 | 0.3048 | -0.2566 | -0.3529 | -0.5812 | -0.1020 | 1486.0000 | 0.4711 | 0.7491 |
| pred_rf__sim3_regime_concentrated | 0.1820 | 0.0596 | 0.2897 | 0.1173 | 0.1473 | -0.4678 | 0.1275 | 622.0000 | 0.4759 | 0.5868 |
| pred_xgb__sim1_equal_weight | -0.3290 | -0.1291 | 0.3167 | -0.4743 | -0.6568 | -0.5018 | -0.2574 | 1479.0000 | 0.4503 | 0.7895 |
| pred_xgb__sim2_stop_weighted | -0.5287 | -0.2295 | 0.3108 | -0.8746 | -1.2162 | -0.6478 | -0.3542 | 1513.0000 | 0.4263 | 0.7509 |
| pred_xgb__sim3_regime_concentrated | -0.3725 | -0.1491 | 0.3069 | -0.5721 | -0.6638 | -0.5568 | -0.2678 | 610.0000 | 0.4475 | 0.5799 |
| pred_ens__sim1_equal_weight | -0.2220 | -0.0833 | 0.3174 | -0.3141 | -0.4348 | -0.5413 | -0.1539 | 1479.0000 | 0.4767 | 0.7900 |
| pred_ens__sim2_stop_weighted | -0.1829 | -0.0676 | 0.3106 | -0.2739 | -0.3719 | -0.5610 | -0.1205 | 1506.0000 | 0.4602 | 0.7523 |
| pred_ens__sim3_regime_concentrated | -0.3178 | -0.1241 | 0.2917 | -0.5206 | -0.6132 | -0.5729 | -0.2167 | 617.0000 | 0.4603 | 0.5841 |
| Nifty 50 | 0.1867 | 0.0611 | 0.1337 | 0.0155 | 0.0213 | -0.1577 | 0.3876 |  |  |  |
| Nifty 500 | 0.2373 | 0.0766 | 0.1433 | 0.1229 | 0.1541 | -0.1884 | 0.4063 |  |  |  |
| Equal weight universe | 0.3105 | 0.0983 | 0.1936 | 0.2360 | 0.2918 | -0.2425 | 0.4051 |  |  |  |

## Error analysis

# Findings for pred_ens at threshold 0.3005

Signals on the test period: 85892. Precision 0.330. Average forward return 0.0088.
By year: best lift 1.23 at 2025, worst lift 1.17 at 2026.
By regime: best lift 1.25 at healthy, worst lift 1.08 at weak.
By volatility_bucket: best lift 1.17 at q1_low, worst lift 1.12 at q4_high.
By liquidity_bucket: best lift 1.25 at q1_low, worst lift 1.16 at q4_high.
By sector (50 or more signals): best lift 1.42 at Construction, worst lift 0.76 at Diversified.

Features that differ most between false positives and true positives:
- cs_rank_vol_21: standardized gap -0.08
- intraday_range: standardized gap -0.07
- atr_pct_14: standardized gap -0.07
- vol_63: standardized gap -0.05
- dist_52w_low: standardized gap -0.05
- ret_126: standardized gap -0.05
- vol_21: standardized gap -0.04
- vix: standardized gap 0.04
