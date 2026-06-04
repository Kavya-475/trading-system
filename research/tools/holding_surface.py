"""
tools/holding_surface.py — holding-period return surface (every entry→exit year)
=================================================================================
Runs the CURRENT config (config.py) over EVERY (start_year → end_year) window in
[START_YEAR, END_YEAR]: gaps of 1, 2, … years. For 2007→2026 that's 19+18+…+1 =
190 backtests. Drives backtester.run_backtest — no logic reimplemented.

Output: a CAGR matrix (rows = entry year, cols = exit year) + a summary
(median/min/max CAGR, % of windows positive, worst drawdown). Full per-window
metrics saved to surface_results.csv.

⚠️  Survivorship-free universe membership ends ~2020-07-31; windows EXITING after
that carry stale-universe bias (optimistic). Read post-2020 columns with that in
mind — the pre-2020 block is the clean one.

Run on the VM (all cores):
    nohup python tools/holding_surface.py > surface.log 2>&1 &
    python tools/holding_surface.py --start-year 2007 --end-year 2026 --workers 8
"""
import argparse
import contextlib
import os
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, HERE)                              # config, data caches, outputs (root)
sys.path.insert(0, os.path.join(HERE, "research"))    # backtester.py (research/)
OUT = os.path.join(HERE, "surface_results.csv")


def _init():
    import backtester as bt
    orig, cache = bt.load_data, {}
    def cached():
        if "d" not in cache:
            cache["d"] = orig()
        return cache["d"]
    bt.load_data = cached


def evaluate(se):
    import config as cfg, backtester as bt
    s, e = se
    cfg.START_DATE, cfg.END_DATE = "%d-01-01" % s, "%d-01-01" % e
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            m = bt.run_backtest(save=False)
        return {"start": s, "end": e, "years": e - s, "cagr": round(float(m["cagr"]), 6),
                "maxdd": round(float(m["maxdd"]), 6), "sharpe": round(float(m["sharpe"]), 6),
                "edge_cagr": round(float(m["edge_cagr"]), 6)}
    except Exception as ex:                                          # noqa: BLE001
        return {"start": s, "end": e, "years": e - s, "cagr": None, "err": str(ex)[:120]}


def report():
    df = pd.read_csv(OUT)
    df = df.dropna(subset=["cagr"])
    import config as cfg
    print("\n" + "=" * 96)
    print("  HOLDING-PERIOD CAGR SURFACE  (rows = entry year, cols = exit year)   %d windows" % len(df))
    print("  config: weights %.2f/%.2f/%.2f/%.2f/%.2f · TOP_N %d · cut %d · tiered · regime-%s"
          % (cfg.W_MOM_12M, cfg.W_MOM_6M, cfg.W_MOM_3M, cfg.W_VOL, cfg.W_DIST_DMA,
             cfg.TOP_N, cfg.EXIT_RANK_CUTOFF, "OFF" if getattr(cfg, "BACKTEST_FORCE_RISK_ON", False) else "ON"))
    print("  ⚠ exit-year columns after 2020 use the stale post-2020 universe (optimistic)")
    print("=" * 96)
    mat = df.pivot(index="start", columns="end", values="cagr") * 100
    with pd.option_context("display.width", 260, "display.max_columns", 60,
                           "display.float_format", lambda x: f"{x:5.0f}" if pd.notna(x) else "    ·"):
        print(mat.to_string(na_rep="    ·"))

    # Summaries — full set and the clean pre-2020-exit subset.
    for label, sub in (("ALL windows", df), ("clean (exit ≤ 2020)", df[df["end"] <= 2020])):
        if sub.empty:
            continue
        cg = sub["cagr"]
        print("\n  %s — %d windows" % (label, len(sub)))
        print("    CAGR   median %5.1f%%   min %5.1f%% (%s)   max %5.1f%% (%s)   | positive %.0f%%"
              % (cg.median() * 100, cg.min() * 100,
                 "%d-%d" % (sub.loc[cg.idxmin(), "start"], sub.loc[cg.idxmin(), "end"]),
                 cg.max() * 100, "%d-%d" % (sub.loc[cg.idxmax(), "start"], sub.loc[cg.idxmax(), "end"]),
                 100 * (cg > 0).mean()))
        print("    maxDD  median %5.1f%%   worst %5.1f%%   |  beat-universe-edge in %.0f%% of windows"
              % (sub["maxdd"].median() * 100, sub["maxdd"].min() * 100, 100 * (sub["edge_cagr"] > 0).mean()))
    print("\n  Full per-window metrics: %s" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2007)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(); return

    wins = [(s, e) for s in range(args.start_year, args.end_year)
            for e in range(s + 1, args.end_year + 1)]
    print("Holding-period surface: %d windows (%d→%d) · %d workers"
          % (len(wins), args.start_year, args.end_year, args.workers))
    t0, done, rows = time.time(), 0, []
    with mp.Pool(args.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(evaluate, wins, chunksize=1):
            rows.append(r); done += 1
            if done % 20 == 0 or done == len(wins):
                el = time.time() - t0
                print("  %d/%d  elapsed %.1fm  eta %.1fm" % (done, len(wins), el / 60, el / done * (len(wins) - done) / 60), flush=True)
    pd.DataFrame(rows).sort_values(["start", "end"]).to_csv(OUT, index=False)
    report()


if __name__ == "__main__":
    main()
