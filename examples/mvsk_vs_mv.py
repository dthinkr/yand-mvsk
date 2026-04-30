"""Compare MVSK-optimal vs mean-variance-optimal portfolios."""

import numpy as np
from yand_mvsk import yand_mvsk_solve, crra_coefficients, MVSKOracle

rng = np.random.default_rng(77)
n, T = 30, 504

# Generate returns with mild skewness
mu = rng.normal(0.0005, 0.001, n)
vol = rng.uniform(0.01, 0.04, n)
R = mu[np.newaxis, :] + rng.standard_normal((T, n)) * vol[np.newaxis, :]

gamma = 6.0
c_mvsk = crra_coefficients(gamma)
c_mv = np.array([c_mvsk[0], c_mvsk[1], 0.0, 0.0])  # drop skewness & kurtosis

res_mvsk = yand_mvsk_solve(R, c_mvsk)
res_mv = yand_mvsk_solve(R, c_mv)

# Evaluate both under the full MVSK objective
oracle = MVSKOracle(R, c_mvsk)
f_mvsk = oracle.value(res_mvsk.x)
f_mv = oracle.value(res_mv.x)

print(f"MVSK objective (lower is better):")
print(f"  MVSK-optimal: {f_mvsk:.8e}")
print(f"  MV-optimal:   {f_mv:.8e}")
print(f"  Gap: {f_mv - f_mvsk:.4e}")

port_mvsk = R @ res_mvsk.x
port_mv = R @ res_mv.x

print(f"\nPortfolio statistics:")
print(f"  {'':12} {'MVSK':>10} {'MV':>10}")
print(f"  {'Return':12} {port_mvsk.mean()*252*100:>9.2f}% {port_mv.mean()*252*100:>9.2f}%")
print(f"  {'Volatility':12} {port_mvsk.std()*np.sqrt(252)*100:>9.2f}% {port_mv.std()*np.sqrt(252)*100:>9.2f}%")
