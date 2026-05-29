import subprocess
import re
from itertools import product

TOP_N_VALUES      = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MAX_SECTOR_VALUES = [2, 3, 4, 5, 6]
EXIT_RANK_VALUES  = [15, 20, 25, 30, 35, 40]

results = []
total   = len(TOP_N_VALUES) * len(MAX_SECTOR_VALUES) * len(EXIT_RANK_VALUES)
count   = 0

for n, s, e in product(TOP_N_VALUES, MAX_SECTOR_VALUES, EXIT_RANK_VALUES):
    count += 1
    with open("config.py", "r") as f:
        cfg = f.read()
    cfg = re.sub(r'TOP_N\s*=\s*\d+',            f'TOP_N            = {n}', cfg)
    cfg = re.sub(r'MAX_PER_SECTOR\s*=\s*\d+',   f'MAX_PER_SECTOR   = {s}', cfg)
    cfg = re.sub(r'EXIT_RANK_CUTOFF\s*=\s*\d+', f'EXIT_RANK_CUTOFF = {e}', cfg)
    cfg = re.sub(r'START_DATE\s*=\s*"[\d-]+"',  'START_DATE = "2008-01-01"', cfg)
    cfg = re.sub(r'END_DATE\s*=\s*"[\d-]+"',    'END_DATE   = "2015-12-31"', cfg)
    with open("config.py", "w") as f:
        f.write(cfg)
    result = subprocess.run(['python3', 'backtester.py'], capture_output=True, text=True)
    out = result.stdout + result.stderr
    def ex(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None
    cagr    = ex(r'CAGR\s+:\s+([\d.]+)%')
    sharpe  = ex(r'Sharpe\s+:\s+([\d.]+)')
    sortino = ex(r'Sortino\s+:\s+([\d.]+)')
    maxdd   = ex(r'Max Drawdown\s+:\s+(-[\d.]+)%')
    if sharpe:
        results.append({'n':n,'s':s,'e':e,'cagr':cagr,'sharpe':sharpe,'sortino':sortino,'maxdd':maxdd})
        print(f"[{count}/{total}] N={n} S={s} E={e} | CAGR={cagr:.1f}% Sharpe={sharpe:.2f} DD={maxdd:.1f}%")

with open("config.py", "r") as f:
    cfg = f.read()
cfg = re.sub(r'TOP_N\s*=\s*\d+',            'TOP_N            = 12', cfg)
cfg = re.sub(r'MAX_PER_SECTOR\s*=\s*\d+',   'MAX_PER_SECTOR   = 3',  cfg)
cfg = re.sub(r'EXIT_RANK_CUTOFF\s*=\s*\d+', 'EXIT_RANK_CUTOFF = 25', cfg)
cfg = re.sub(r'START_DATE\s*=\s*"[\d-]+"',  'START_DATE = "2018-01-01"', cfg)
cfg = re.sub(r'END_DATE\s*=\s*"[\d-]+"',    'END_DATE   = "2026-05-22"', cfg)
with open("config.py", "w") as f:
    f.write(cfg)

results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\n" + "="*75)
print("TOP 10 — TRAIN PERIOD (2008-2015)")
print("="*75)
print(f"{'Rank':>4} {'N':>4} {'S':>4} {'E':>5} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
print("-"*75)
for i, r in enumerate(results[:10], 1):
    print(f"{i:>4} {r['n']:>4} {r['s']:>4} {r['e']:>5} {r['cagr']:>7.1f}% {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['maxdd']:>7.1f}%")
print("="*75)
print(f"Total combinations tested: {total}")
