---
name: feedback-execution-bugs
description: Bugs found and fixed in execution.py and data_manager.py in May 2026
metadata:
  type: feedback
---

get_portfolio_value PAPER_MODE double-counted capital (returned CAPITAL + stock_value). Fixed to return just CAPITAL.
**Why:** On any rebalance with existing holdings, inflated portfolio value caused 2× over-allocation per position.
**How to apply:** In PAPER_MODE always use CAPITAL as the deployment base; don't add current stock value to it.

data_manager._update_index_cache crashed with "If using all scalar values, you must pass an index" when only 1 trading day was fetched (yf.download returns 1 row, .squeeze() returns a float scalar, not a Series). Fixed by passing `index=r500.index` to the DataFrame constructor.
**Why:** The regime cache was silently skipping every daily update, potentially causing stale regime data.
**How to apply:** Whenever creating a DataFrame from .squeeze() calls, always pass an explicit index.
