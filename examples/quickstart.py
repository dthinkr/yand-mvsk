"""Minimal YAND-MVSK example: 3 lines to an optimal portfolio."""

import numpy as np
from yand_mvsk import yand_mvsk_solve, crra_coefficients

# 50 assets, 2 years of daily returns
rng = np.random.default_rng(42)
R = rng.standard_normal((504, 50)) * 0.02 + 0.0003

# Solve with CRRA risk aversion gamma=6
result = yand_mvsk_solve(R, crra_coefficients(gamma=6))

print(f"Converged: {result.converged} in {result.n_iter} iterations")
print(f"KKT residual: {result.kkt_residual:.2e}")
print(f"Non-zero weights (>1%): {np.sum(result.x > 0.01)}")
print(f"Top 5 weights: {np.sort(result.x)[::-1][:5].round(3)}")
