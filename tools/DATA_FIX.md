# Data fixes for a bias-free backtest

The backtester's *machinery* is realistic (costs, STCG/LTCG with the 2024 rate
change settled annually, next-day-open fills with missed gap-ups, slippage,
idle-cash interest, point-in-time universe). Two **data** gaps still bias results.
Both need files that can only be downloaded manually from NSE / niftyindices.

---

## Gap 1 — Universe history is stale after 2020-07-31  (HIGH impact)

`universe_history.csv` and `IndexInclExcl.xls` both stop at **2020-07-31**, but
the backtest runs to **2026**. So for ~6 years of the window, every day trades
against the *2020* constituent list. The survivorship fix is effectively off
exactly where the live strategy operates.

**Fix — add the missing constituent snapshots:**

1. Go to **niftyindices.com → Resources → Index Constituents**
   (or NSE → Indices → NIFTY LARGEMIDCAP 250 → Historical / Factsheet).
2. Download the **NIFTY LargeMidcap 250** constituent list for each semi-annual
   rebalance from **Sep-2020 to the latest** (NSE rebalances end-March & end-Sept):
   `2020-09`, `2021-03`, `2021-09`, `2022-03`, `2022-09`, `2023-03`, `2023-09`,
   `2024-03`, `2024-09`, `2025-03`, `2025-09`, plus the **current** list. (~12 files)
3. Save each into `nse_snapshots/` named by its effective date, e.g.
   `nse_snapshots/2024-09-30.csv`. The file must have a Symbol column and ideally
   an Industry/Sector column (the parser auto-detects common header variants).
4. Re-run:
   ```bash
   python build_universe_history.py   # rewrites universe_history.csv
   python data_manager.py             # only if new_tickers.txt is non-empty
   python backtester.py
   ```
   The STALE warning in the backtest output should disappear.

> The newer `IndexInclExcl.xls` (inclusion/exclusion events through the present),
> if niftyindices publishes an updated one, is an alternative source — but it
> lists full company names, not NSE symbols, so it needs a name→symbol mapping
> step. The dated-snapshot route above is simpler and is what the existing
> `build_universe_history.py` already ingests.

---

## Gap 2 — 47 of 420 historical tickers (11%) have no price data  (MEDIUM impact)

These are delisted / acquired / renamed names (AMTEKAUTO, BHUSHAN, ESSAR, IVRCL,
HOTELEELA, …). yfinance does not serve delisted Indian equities, so they silently
vanish from scoring — a residual survivorship bias that flatters the long
(2007–2020) part of the backtest. The backtester now **prints this gap** at
startup so it's measured, not hidden.

**Options, best to worst:**

1. **Paid vendor with delisted data** — the only true fix. For NSE: a
   point-in-time database (e.g. a bhavcopy-based archive, or a commercial feed)
   that retains delisted symbols. Load it into the same cache CSV format.
2. **Reconstruct from NSE bhavcopy archives** — NSE publishes daily bhavcopy
   files going back years; they include securities that later delisted. Building
   a delisted-aware cache from these is the DIY version of option 1.
3. **Accept & document** — if the goal is the live (2023→) strategy, the
   delisted names are almost all pre-2020 and matter little to recent results.
   Keep the startup warning and treat pre-2020 CAGR as an upper bound.

---

## What was already changed in code (no action needed)

- `config.py`: `BACKTEST_FORCE_RISK_ON=False` (regime filter now exercised in
  backtests, decoupled from the live `FORCE_RISK_ON`); `START_DATE=2007-10-01`
  (window now spans 2008/2018/2020/2022 so drawdowns are real);
  `SLIPPAGE_BPS`, `CASH_YIELD`, `ANNUAL_TAX`, LTCG exemption knobs added.
- `backtester.py`: per-side slippage on fills; daily idle-cash interest;
  annual (per-financial-year) capital-gains tax with the ₹1L/1.25L exemption and
  short-term-loss offset; a fair equal-weight same-universe benchmark; plus the
  survivorship-gap and staleness warnings.
