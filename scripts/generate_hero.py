"""Generate the hero graphic for the README."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

from yand_mvsk import yand_mvsk_solve, crra_coefficients, MVSKOracle

def generate_returns(n, T, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0005, 0.001, n)
    vol = rng.uniform(0.01, 0.04, n)
    return mu[np.newaxis, :] + rng.standard_normal((T, n)) * vol[np.newaxis, :]

gamma = 6.0
c = crra_coefficients(gamma)

configs = [
    (20,  252, '#2563eb', 'n = 20'),
    (100, 252, '#7c3aed', 'n = 100'),
    (400, 252, '#db2777', 'n = 400'),
    (800, 252, '#ea580c', 'n = 800'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'wspace': 0.35})

for n, T, color, label in configs:
    R = generate_returns(n, T, seed=42+n)
    res = yand_mvsk_solve(R, c, max_iter=50, verbose=False)

    oracle = MVSKOracle(R, c)
    f_star = res.f_val
    gaps = [abs(h - f_star) + 1e-18 for h in res.history]

    iters = list(range(len(gaps)))
    ax1.semilogy(iters, gaps, '-o', color=color, label=label,
                 markersize=4, linewidth=1.8, alpha=0.9)

    x_ew = np.ones(n) / n
    f_ew = oracle.value(x_ew)
    improvement = (f_ew - f_star) / abs(f_ew) * 100
    ax2.bar(label, improvement, color=color, alpha=0.85, width=0.6)

ax1.set_xlabel('Iteration')
ax1.set_ylabel('|f(x) - f*|')
ax1.set_title('Convergence', fontweight='bold', fontsize=13)
ax1.legend(frameon=False, fontsize=10)
ax1.set_ylim(bottom=1e-18)

ax2.set_ylabel('Improvement over equal-weight (%)')
ax2.set_title('MVSK vs Equal-Weight', fontweight='bold', fontsize=13)

fig.suptitle('YAND-MVSK: Fast Higher-Moment Portfolio Optimization',
             fontsize=15, fontweight='bold', y=1.02)

plt.savefig('docs/hero.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('docs/hero.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved docs/hero.png and docs/hero.svg')
