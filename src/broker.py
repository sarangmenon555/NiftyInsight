import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config


def build_orders(signals, capital, max_positions, weight_by_proba=False):
    picks = signals.head(max_positions).copy()
    if len(picks) == 0:
        return pd.DataFrame(columns=["symbol", "side", "quantity", "reference_price", "exchange", "product", "order_type", "stop_loss_price"])
    slot = capital / max_positions
    weights = np.ones(len(picks))
    if weight_by_proba and "proba" in picks.columns:
        threshold = picks["proba"].min()
        weights = np.clip(picks["proba"].to_numpy() / threshold, 1.0, 1.5)
    picks["budget"] = slot * weights
    picks["quantity"] = np.floor(picks["budget"] / picks["price"]).astype(int)
    picks = picks[picks["quantity"] > 0]
    return pd.DataFrame(
        {
            "symbol": picks["symbol"].to_numpy(),
            "side": "BUY",
            "quantity": picks["quantity"].to_numpy(),
            "reference_price": picks["price"].to_numpy(),
            "exchange": "NSE",
            "product": "CNC",
            "order_type": "MARKET",
            "stop_loss_price": picks["stop_loss_price"].to_numpy(),
        }
    )


class KiteBroker:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.client = None
        if not dry_run:
            if os.environ.get("ENABLE_LIVE_TRADING") != "1":
                raise RuntimeError("Live trading is disabled. Set ENABLE_LIVE_TRADING=1 to allow real orders.")
            from kiteconnect import KiteConnect

            self.client = KiteConnect(api_key=os.environ["KITE_API_KEY"])
            self.client.set_access_token(os.environ["KITE_ACCESS_TOKEN"])

    def place(self, orders):
        log = []
        for row in orders.itertuples(index=False):
            record = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": row.symbol,
                "side": row.side,
                "quantity": int(row.quantity),
                "reference_price": float(row.reference_price),
                "mode": "dry_run" if self.dry_run else "live",
                "order_id": "",
                "status": "not_sent",
            }
            if not self.dry_run:
                try:
                    order_id = self.client.place_order(
                        variety=self.client.VARIETY_REGULAR,
                        exchange=self.client.EXCHANGE_NSE,
                        tradingsymbol=row.symbol,
                        transaction_type=self.client.TRANSACTION_TYPE_BUY,
                        quantity=int(row.quantity),
                        product=self.client.PRODUCT_CNC,
                        order_type=self.client.ORDER_TYPE_MARKET,
                    )
                    record["order_id"] = str(order_id)
                    record["status"] = "sent"
                except Exception as error:
                    record["status"] = "error: " + type(error).__name__
            else:
                record["status"] = "dry_run_only"
            log.append(record)
        frame = pd.DataFrame(log)
        if frame.empty:
            return frame
        path = config.REPORT_DIR / "orders_log.csv"
        header = not path.exists()
        frame.to_csv(path, mode="a", header=header, index=False)
        return frame
