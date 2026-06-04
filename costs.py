"""
costs.py — shared Indian-equity transaction cost model
======================================================
Single source of truth for per-trade charges, used by BOTH the live/paper path
(paper_engine.py) and the backtest (research/backtester.py). Lives at the repo
root so the live runtime never has to import anything from research/.

Delivery (CNC) charges: brokerage ₹0 (Zerodha delivery), exchange txn charge,
SEBI charge, STT (both sides for delivery), stamp duty (buy side), a flat DP
charge per sell, and 18% GST on the brokerage+exchange+SEBI components. All rates
come from config.py.
"""
import config as cfg


def txn_cost(value, side):
    brokerage = 0.0                                   # Zerodha CNC delivery = ₹0
    exch      = value * cfg.EXCHANGE_CHARGE
    sebi      = value * cfg.SEBI_CHARGE
    stt       = value * (cfg.STT_BUY if side == "buy" else cfg.STT_SELL)   # both sides for delivery
    stamp     = value * cfg.STAMP_DUTY if side == "buy"  else 0.0
    dp        = 15.93                  if side == "sell" else 0.0          # flat DP charge per sell
    gst       = (brokerage + exch + sebi) * getattr(cfg, "GST_RATE", 0.0) # 18% GST on charges
    return brokerage + exch + sebi + stt + stamp + dp + gst
