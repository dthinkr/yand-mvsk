"""Minimal YAND-MVSK example: from returns to optimized portfolio in 4 lines."""

import numpy as np
from yand_mvsk import EfficientMVSK

# 50 assets, 2 years of daily returns
rng = np.random.default_rng(42)
R = rng.standard_normal((504, 50)) * 0.02 + 0.0003

ef = EfficientMVSK(R, gamma=6, tickers=[f"A{i:02d}" for i in range(50)])
weights = ef.optimize()
cleaned = ef.clean_weights()

print(f"Converged: {ef.result.converged} in {ef.result.n_iter} iterations")
print(f"Non-zero weights: {sum(1 for v in cleaned.values() if v > 0)}")
print(f"\nTop 5 allocations:")
for ticker, w in sorted(cleaned.items(), key=lambda x: -x[1])[:5]:
    print(f"  {ticker}: {w:.1%}")

print()
ef.portfolio_performance(verbose=True)
