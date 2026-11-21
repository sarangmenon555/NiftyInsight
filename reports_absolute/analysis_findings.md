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
