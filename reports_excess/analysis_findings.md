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
