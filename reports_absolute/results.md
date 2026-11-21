# Results

## Model quality

| split | prediction | rows | base_rate | auc | average_precision | precision_top_1pct |
|---|---|---|---|---|---|---|
| valid | pred_dt | 165530 | 0.2444 | 0.4900 | 0.2363 | 0.3746 |
| valid | pred_rf | 165530 | 0.2444 | 0.5798 | 0.3044 | 0.4224 |
| valid | pred_xgb | 165530 | 0.2444 | 0.5775 | 0.3024 | 0.3668 |
| valid | pred_ens | 165530 | 0.2444 | 0.5801 | 0.3047 | 0.4224 |
| test | pred_dt | 471437 | 0.2372 | 0.4693 | 0.2214 | 0.2490 |
| test | pred_rf | 471437 | 0.2372 | 0.6029 | 0.3126 | 0.4728 |
| test | pred_xgb | 471437 | 0.2372 | 0.5771 | 0.2977 | 0.4285 |
| test | pred_ens | 471437 | 0.2372 | 0.5935 | 0.3082 | 0.4697 |

## Vector simulation (approximate)

| prediction | strategy | trades | precision | avg_net_return_per_trade | approx_cagr | approx_sharpe | approx_max_drawdown |
|---|---|---|---|---|---|---|---|
| pred_dt | sim1 | 86838 | 0.1968 | -0.0104 | -0.1458 | -4.3887 | -0.3740 |
| pred_dt | sim2 | 1060 | 0.1764 | -0.0111 | -0.1522 | -4.2732 | -0.3817 |
| pred_dt | sim3 | 850 | 0.1612 | -0.0102 | -0.1147 | -4.5023 | -0.2968 |
| pred_rf | sim1 | 124988 | 0.3191 | 0.0035 | 0.0202 | -0.3084 | -0.2985 |
| pred_rf | sim2 | 2818 | 0.3297 | 0.0010 | 0.0628 | 0.0460 | -0.3573 |
| pred_rf | sim3 | 2002 | 0.3242 | 0.0011 | 0.0706 | 0.0798 | -0.2682 |
| pred_xgb | sim1 | 101848 | 0.3100 | 0.0026 | 0.0269 | -0.3412 | -0.2095 |
| pred_xgb | sim2 | 1856 | 0.3324 | 0.0033 | 0.1212 | 0.4594 | -0.2343 |
| pred_xgb | sim3 | 1222 | 0.3061 | 0.0038 | 0.0876 | 0.2451 | -0.1127 |
| pred_ens | sim1 | 113488 | 0.3171 | 0.0032 | -0.0102 | -0.5885 | -0.2984 |
| pred_ens | sim2 | 2296 | 0.3441 | 0.0023 | 0.0460 | -0.0647 | -0.2621 |
| pred_ens | sim3 | 1522 | 0.3357 | 0.0033 | 0.0429 | -0.1443 | -0.2216 |

## Exact simulation versus benchmarks

| key | total_return | cagr | volatility | sharpe | sortino | max_drawdown | calmar | trades | win_rate | avg_exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| pred_dt__sim1_equal_weight | -0.5260 | -0.2279 | 0.1151 | -2.7097 | -2.5503 | -0.5367 | -0.4247 | 516.0000 | 0.3527 | 0.2771 |
| pred_dt__sim2_stop_weighted | -0.5724 | -0.2551 | 0.1316 | -2.6186 | -2.2891 | -0.5788 | -0.4406 | 418.0000 | 0.3517 | 0.2867 |
| pred_dt__sim3_regime_concentrated | -0.4565 | -0.1904 | 0.1209 | -2.1881 | -1.8215 | -0.4565 | -0.4172 | 190.0000 | 0.3579 | 0.2376 |
| pred_rf__sim1_equal_weight | -0.1333 | -0.0484 | 0.2799 | -0.2649 | -0.3316 | -0.4608 | -0.1050 | 1220.0000 | 0.4615 | 0.6523 |
| pred_rf__sim2_stop_weighted | -0.3432 | -0.1355 | 0.2586 | -0.6734 | -0.8973 | -0.4920 | -0.2755 | 1255.0000 | 0.4359 | 0.6135 |
| pred_rf__sim3_regime_concentrated | -0.4619 | -0.1933 | 0.2420 | -1.0165 | -1.2173 | -0.5748 | -0.3363 | 518.0000 | 0.4305 | 0.4824 |
| pred_xgb__sim1_equal_weight | 0.2486 | 0.0800 | 0.2263 | 0.1597 | 0.1838 | -0.2734 | 0.2926 | 856.0000 | 0.4848 | 0.4586 |
| pred_xgb__sim2_stop_weighted | 0.0681 | 0.0231 | 0.2218 | -0.0806 | -0.0901 | -0.2574 | 0.0896 | 879.0000 | 0.4744 | 0.4399 |
| pred_xgb__sim3_regime_concentrated | 0.1221 | 0.0407 | 0.1792 | -0.0545 | -0.0586 | -0.2954 | 0.1379 | 338.0000 | 0.4645 | 0.3324 |
| pred_ens__sim1_equal_weight | -0.1654 | -0.0607 | 0.2660 | -0.3415 | -0.4235 | -0.4031 | -0.1506 | 1033.0000 | 0.4569 | 0.5529 |
| pred_ens__sim2_stop_weighted | -0.1902 | -0.0705 | 0.2463 | -0.4311 | -0.5380 | -0.3898 | -0.1809 | 1058.0000 | 0.4376 | 0.5137 |
| pred_ens__sim3_regime_concentrated | 0.0156 | 0.0054 | 0.2070 | -0.1851 | -0.2280 | -0.3190 | 0.0169 | 411.0000 | 0.4599 | 0.3899 |
| Nifty 50 | 0.1867 | 0.0611 | 0.1337 | 0.0155 | 0.0213 | -0.1577 | 0.3876 |  |  |  |
| Nifty 500 | 0.2373 | 0.0766 | 0.1433 | 0.1229 | 0.1541 | -0.1884 | 0.4063 |  |  |  |
| Equal weight universe | 0.3105 | 0.0983 | 0.1936 | 0.2360 | 0.2918 | -0.2425 | 0.4051 |  |  |  |

## Error analysis

# Findings for pred_ens at threshold 0.2760

Signals on the test period: 113488. Precision 0.317. Average forward return 0.0062.
By year: best lift 1.44 at 2025, worst lift 1.27 at 2024.
By regime: best lift 1.29 at healthy, worst lift 1.17 at weak.
By volatility_bucket: best lift 1.66 at q1_low, worst lift 1.13 at q4_high.
By liquidity_bucket: best lift 1.35 at q4_high, worst lift 1.31 at q2.
By sector (50 or more signals): best lift 1.66 at Fast Moving Consumer Goods, worst lift 1.06 at Diversified.

Features that differ most between false positives and true positives:
- ew_vol_21: standardized gap -0.35
- intraday_range: standardized gap -0.20
- ew_ret_5: standardized gap 0.20
- vix: standardized gap -0.19
- breadth_sma50: standardized gap 0.18
- vix_pctile_252: standardized gap -0.18
- ew_dist_sma_200: standardized gap 0.18
- nifty50_pe: standardized gap 0.17
