"""MVSK vs Mean-Variance on real ETF data.

Compares tail-risk-aware (MVSK) allocation against classical mean-variance
using a diversified ETF universe. Shows how higher moments shift weights
away from crash-prone, fat-tailed assets.

Requires: pip install yfinance pandas
"""

import numpy as np

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    raise SystemExit("pip install yfinance pandas")

from yand_mvsk import EfficientMVSK

# --- Data ---
tickers = ["SPY", "QQQ", "TLT", "GLD", "IEF", "HYG", "EFA", "VNQ"]
print(f"Downloading {', '.join(tickers)} ...")
prices = yf.download(tickers, start="2019-01-01", end="2024-12-31", progress=False)["Close"]
prices = prices.dropna()
print(f"  {len(prices)} trading days, {prices.shape[1]} assets\n")

# --- MVSK optimization ---
gamma = 6.0
ef_mvsk = EfficientMVSK.from_prices(prices, gamma=gamma)
w_mvsk = ef_mvsk.optimize()

# --- Mean-variance (drop skewness & kurtosis) ---
from yand_mvsk import crra_coefficients
c_mv = crra_coefficients(gamma)
c_mv[2] = 0.0  # zero out skewness
c_mv[3] = 0.0  # zero out kurtosis
ef_mv = EfficientMVSK.from_prices(prices, gamma=gamma, c=c_mv)
w_mv = ef_mv.optimize()

# --- Weight comparison ---
print("=" * 62)
print(f"{'Ticker':>8}  {'MVSK':>10}  {'MV':>10}  {'Δ':>10}")
print("-" * 62)
cw_mvsk = ef_mvsk.clean_weights(cutoff=0.001)
cw_mv = ef_mv.clean_weights(cutoff=0.001)
for t in tickers:
    m, v = cw_mvsk.get(t, 0), cw_mv.get(t, 0)
    delta = m - v
    bar = "+" * int(abs(delta) * 100) if delta > 0.005 else "-" * int(abs(delta) * 100) if delta < -0.005 else ""
    print(f"{t:>8}  {m:>9.1%}  {v:>9.1%}  {delta:>+9.1%}  {bar}")
print("=" * 62)

# --- Performance stats ---
print("\nMVSK-optimal portfolio:")
s_mvsk = ef_mvsk.portfolio_performance(verbose=True)

print("\nMV-optimal portfolio:")
s_mv = ef_mv.portfolio_performance(verbose=True)

# --- Per-asset moment profile ---
returns = prices.pct_change().dropna()
print("\n" + "=" * 62)
print(f"{'Ticker':>8}  {'Ann.Ret':>8}  {'Vol':>8}  {'Skew':>8}  {'ExKurt':>8}")
print("-" * 62)
for t in tickers:
    r = returns[t]
    mu = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    c = r - r.mean()
    m2 = np.mean(c**2)
    skew = np.mean(c**3) / (m2**1.5)
    kurt = np.mean(c**4) / (m2**2) - 3
    print(f"{t:>8}  {mu:>7.1%}  {vol:>7.1%}  {skew:>+7.2f}  {kurt:>+7.2f}")
print("=" * 62)

print("\nKey insight: MVSK shifts weight toward assets with better skewness/kurtosis profiles,")
print("reducing exposure to crash-prone, fat-tailed positions that MV ignores.")
