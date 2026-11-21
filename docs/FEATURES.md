# Feature dictionary

## Numeric features

| Feature | Description |
|---|---|
| ret_1 | One day return on the split-adjusted close. |
| ret_2 | Two day return. |
| ret_3 | Three day return. |
| ret_5 | One trading week return. |
| ret_10 | Two trading week return. |
| ret_21 | One month momentum (21 trading days). |
| ret_63 | Three month momentum. |
| ret_126 | Six month momentum. |
| gap | Opening gap versus the previous adjusted close. |
| intraday_range | High minus low divided by close, a daily volatility proxy. |
| body | Close minus open divided by open, the candle body. |
| close_location | Position of the close inside the day range, 0 is the low and 1 is the high. |
| vol_5 | Standard deviation of daily returns over 5 days. |
| vol_21 | Standard deviation of daily returns over 21 days. |
| vol_63 | Standard deviation of daily returns over 63 days. |
| vol_ratio_5_63 | Short term over long term volatility, flags volatility expansion. |
| atr_pct_14 | Average true range over 14 days as a share of price. |
| rsi_14 | Relative strength index with a 14 day Wilder smoothing. |
| macd_hist | MACD histogram (12, 26, 9) scaled by price. |
| dist_sma_20 | Distance of price from its 20 day simple moving average. |
| dist_sma_50 | Distance of price from its 50 day simple moving average. |
| dist_sma_200 | Distance of price from its 200 day simple moving average. |
| dist_52w_high | Distance of price from its 52 week high, always zero or negative. |
| dist_52w_low | Distance of price from its 52 week low, always zero or positive. |
| bb_pctb | Bollinger percent B over 20 days, how far price sits inside the band. |
| max_ret_21 | Largest single day gain in the last 21 days, a lottery-like feature. |
| min_ret_21 | Largest single day loss in the last 21 days. |
| skew_63 | Skewness of daily returns over 63 days. |
| log_turnover | Log of the traded value in rupees for the day. |
| turnover_ratio_5_21 | 5 day average traded value over the 21 day average. |
| volume_ratio_5_21 | 5 day average share volume over the 21 day average. |
| deliv_per | Share of traded quantity that was marked for delivery, taken from NSE bhavcopy. |
| deliv_ratio_21 | Delivery percentage divided by its own 21 day average. |
| deliv_trend_5_21 | 5 day average delivery percentage minus the 21 day average. |
| log_trades | Log of the number of trades executed in the day. |
| avg_trade_value | Log of average rupees per trade, separates retail and institutional activity. |
| log_amihud_21 | Log scaled Amihud illiquidity: mean absolute return per rupee traded over 21 days. |
| beta_63 | Rolling 63 day beta against the equal weighted universe return. |
| ew_ret_1 | Equal weighted universe return for the day. |
| ew_ret_5 | Equal weighted universe return over 5 days. |
| ew_ret_21 | Equal weighted universe return over 21 days. |
| ew_dist_sma_200 | Distance of the equal weighted universe index from its 200 day average. |
| ew_vol_21 | Annualised 21 day volatility of the equal weighted universe. |
| breadth_sma50 | Share of universe stocks trading above their 50 day average. |
| breadth_adv | Share of universe stocks that closed up on the day. |
| nifty50_ret_1 | Nifty 50 one day return, from NSE index closing files. |
| nifty50_ret_5 | Nifty 50 five day return. |
| nifty50_ret_21 | Nifty 50 21 day return. |
| nifty50_dist_sma_200 | Distance of the Nifty 50 from its 200 day average. |
| vix | India VIX closing level. |
| vix_chg_5 | Five day percentage change of India VIX. |
| vix_pctile_252 | Percentile rank of India VIX within the last year. |
| nifty50_pe | Nifty 50 trailing price to earnings ratio. |
| nifty50_pb | Nifty 50 price to book ratio. |
| nifty50_dy | Nifty 50 dividend yield. |
| midcap_rel_21 | Nifty Midcap 100 21 day return minus Nifty 50 21 day return, a risk appetite proxy. |
| cs_rank_ret_21 | Cross sectional percentile of the 21 day return on that date. |
| cs_rank_ret_63 | Cross sectional percentile of the 63 day return on that date. |
| cs_rank_vol_21 | Cross sectional percentile of 21 day volatility on that date. |
| cs_rank_deliv_per | Cross sectional percentile of delivery percentage on that date. |
| cs_rank_turnover_ratio | Cross sectional percentile of the 5 to 21 day traded value ratio. |
| rel_ret_21_sector | 21 day return minus the average 21 day return of the stock's sector. |
| rel_strength_21 | 21 day return minus the equal weighted universe 21 day return. |
| vol_adj_mom_63 | 63 day return divided by its volatility scaled for the horizon. |
| mom_deliv | Interaction of 21 day momentum and the delivery ratio. |

## Dummy and binned features

| Feature | Description |
|---|---|
| dow_* | Day of week dummies, Monday to Friday. |
| month_* | Calendar month dummies. |
| sector_* | Sector dummies from the NSE Nifty 500 industry list, one per sector. |
| bin_ret21_* | Five bins of 21 day momentum: crash, down, flat, up, surge. |
| bin_vol21_* | Four bins of 21 day volatility: low, mid, high, extreme. |
| bin_rsi_* | Five RSI zones from oversold to overbought. |
| bin_hi52_* | Four bins of distance from the 52 week high. |
| bin_vix_* | Four India VIX regimes: calm, normal, elevated, stress. |
| bin_deliv_* | Three bins of the delivery ratio: low, normal, high. |
| flag_above_sma200 | Price above the 200 day average. |
| flag_near_high52 | Price within 5 percent of the 52 week high. |
| flag_near_low52 | Price within 10 percent of the 52 week low. |
| flag_gap_up | Opening gap above 2 percent. |
| flag_gap_down | Opening gap below minus 2 percent. |
| flag_volume_spike | Traded value over twice its 21 day average. |
| flag_high_delivery | Delivery ratio above 1.5. |
| flag_oversold | RSI below 30. |
| flag_regime_ok | Market regime filter: universe above its 200 day average and VIX at or below 25. |

## Targets

| Target | Description |
|---|---|
| ret_fwd | Forward return from the next day open to the close H trading days later, split adjusted. |
| target_reg | Equals ret_fwd by default. With ALGO_TARGET_MODE=excess it is ret_fwd minus the average ret_fwd of the universe on the same date. |
| target_cls | 1 when target_reg is at or above the configured threshold (absolute or excess), otherwise 0. |
