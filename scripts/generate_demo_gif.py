"""Generate README demo GIF: MVSK optimization converging in real time."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from yand_mvsk import yand_mvsk_solve, crra_coefficients, MVSKOracle

# --- Setup ---
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "IEF", "HYG", "EFA", "VNQ"]
n = len(TICKERS)
T = 504

rng = np.random.default_rng(42)
#                      SPY    QQQ    TLT    GLD    IEF    HYG    EFA    VNQ
mu  = np.array([      0.10,  0.14,  0.03,  0.06,  0.02,  0.05,  0.07,  0.04]) / 252
vol = np.array([      0.18,  0.24,  0.10,  0.14,  0.06,  0.12,  0.19,  0.20]) / np.sqrt(252)
# Strong moment differences: QQQ/VNQ crash-prone, TLT/GLD safe-haven
skew_bias = np.array([-0.8, -1.5,   0.4,   0.6,   0.2,  -0.6,  -1.0,  -1.2])
kurt_bias = np.array([ 1.0,  3.0,   0.2,   0.3,   0.1,   1.5,   2.0,   2.5])

R = np.empty((T, n))
for i in range(n):
    z = rng.standard_normal(T)
    # Add skewness via chi-squared mixing
    chi = rng.chisquare(4, T)
    z += skew_bias[i] * (chi - 4) / (2 * np.sqrt(2))
    # Add kurtosis via t-distribution mixing
    t_df = max(5, 20 - kurt_bias[i] * 3)
    z *= np.sqrt(t_df / rng.chisquare(t_df, T)) if kurt_bias[i] > 0.5 else 1.0
    R[:, i] = mu[i] + vol[i] * z

gamma = 6.0
c_mvsk = crra_coefficients(gamma)
c_mv = c_mvsk.copy()
c_mv[2] = 0.0
c_mv[3] = 0.0

# Collect per-iteration weights for MVSK
oracle_mvsk = MVSKOracle(R, c_mvsk)
snapshots = [np.ones(n) / n]
x = np.ones(n) / n
for max_it in range(1, 25):
    res = yand_mvsk_solve(R, c_mvsk, x0=np.ones(n) / n, max_iter=max_it)
    snapshots.append(res.x.copy())
    if res.converged:
        break

import time
t0 = time.perf_counter()
res_mvsk = yand_mvsk_solve(R, c_mvsk)
solve_ms = (time.perf_counter() - t0) * 1000
res_mv = yand_mvsk_solve(R, c_mv)

# Per-asset moments
centered = R - R.mean(axis=0)
m2 = np.mean(centered ** 2, axis=0)
asset_skew = np.mean(centered ** 3, axis=0) / (m2 ** 1.5)
asset_kurt = np.mean(centered ** 4, axis=0) / (m2 ** 2) - 3.0

# Portfolio stats helper
def port_stats(w):
    pr = R @ w
    mu_a = pr.mean() * 252
    vol_a = pr.std() * np.sqrt(252)
    c = pr - pr.mean()
    v2 = np.mean(c ** 2)
    sk = np.mean(c ** 3) / (v2 ** 1.5) if v2 > 1e-30 else 0
    ku = np.mean(c ** 4) / (v2 ** 2) - 3.0 if v2 > 1e-30 else 0
    return mu_a, vol_a, sk, ku

# Objective history
f_star = res_mvsk.f_val
obj_history = []
for snap in snapshots:
    obj_history.append(oracle_mvsk.value(snap))

total_frames = len(snapshots) + 20  # hold last frame

# --- Figure ---
fig = plt.figure(figsize=(13, 5), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.8, 1.0], wspace=0.35,
                      left=0.06, right=0.96, top=0.82, bottom=0.15)

ax_w = fig.add_subplot(gs[0])
ax_obj = fig.add_subplot(gs[1])
ax_stats = fig.add_subplot(gs[2])

BLUE = "#2563eb"
ORANGE = "#ea580c"
GRAY = "#94a3b8"

# Static: MV weights as reference
x_pos = np.arange(n)
bar_width = 0.35

def update(frame):
    idx = min(frame, len(snapshots) - 1)
    w_cur = snapshots[idx]

    # --- Left: Weight bars ---
    ax_w.clear()
    ax_w.bar(x_pos - bar_width / 2, w_cur * 100, bar_width,
             color=BLUE, alpha=0.9, label="MVSK")
    ax_w.bar(x_pos + bar_width / 2, res_mv.x * 100, bar_width,
             color=ORANGE, alpha=0.7, label="MV")
    ax_w.set_xticks(x_pos)
    ax_w.set_xticklabels(TICKERS, fontsize=9)
    ax_w.set_ylabel("Weight (%)", fontsize=10)
    ax_w.set_title("Portfolio Weights", fontweight="bold", fontsize=12)
    ax_w.legend(frameon=False, fontsize=9, loc="upper right")
    ax_w.set_ylim(0, max(res_mvsk.x.max(), res_mv.x.max()) * 105)
    ax_w.spines["top"].set_visible(False)
    ax_w.spines["right"].set_visible(False)

    # --- Middle: Objective convergence ---
    ax_obj.clear()
    show_idx = min(idx + 1, len(obj_history))
    gaps = [abs(v - f_star) + 1e-16 for v in obj_history[:show_idx]]
    ax_obj.semilogy(range(show_idx), gaps, "-o", color=BLUE,
                    markersize=5, linewidth=2)
    ax_obj.set_xlim(-0.5, len(obj_history) + 0.5)
    ax_obj.set_ylim(1e-16, max(gaps) * 5)
    ax_obj.set_xlabel("Iteration", fontsize=10)
    ax_obj.set_ylabel("|f(x) - f*|", fontsize=10)
    ax_obj.set_title("Convergence", fontweight="bold", fontsize=12)
    ax_obj.spines["top"].set_visible(False)
    ax_obj.spines["right"].set_visible(False)

    # --- Right: Stats table ---
    ax_stats.clear()
    ax_stats.axis("off")
    ax_stats.set_title("Portfolio Statistics", fontweight="bold", fontsize=12)

    mu_m, vol_m, sk_m, ku_m = port_stats(w_cur)
    mu_v, vol_v, sk_v, ku_v = port_stats(res_mv.x)
    sharpe_m = mu_m / vol_m if vol_m > 1e-10 else 0
    sharpe_v = mu_v / vol_v if vol_v > 1e-10 else 0

    rows = [
        ("Return",   f"{mu_m:>7.1%}", f"{mu_v:>7.1%}"),
        ("Vol",      f"{vol_m:>7.1%}", f"{vol_v:>7.1%}"),
        ("Sharpe",   f"{sharpe_m:>7.2f}", f"{sharpe_v:>7.2f}"),
        ("Skew",     f"{sk_m:>+7.3f}", f"{sk_v:>+7.3f}"),
        ("ExKurt",   f"{ku_m:>+7.3f}", f"{ku_v:>+7.3f}"),
    ]

    # Header
    ax_stats.text(0.05, 0.90, "", fontsize=10, fontfamily="monospace",
                  transform=ax_stats.transAxes)
    ax_stats.text(0.42, 0.90, "MVSK", fontsize=10, fontweight="bold",
                  color=BLUE, fontfamily="monospace", transform=ax_stats.transAxes)
    ax_stats.text(0.72, 0.90, "MV", fontsize=10, fontweight="bold",
                  color=ORANGE, fontfamily="monospace", transform=ax_stats.transAxes)

    for i, (label, vm, vv) in enumerate(rows):
        y = 0.74 - i * 0.16
        ax_stats.text(0.05, y, label, fontsize=10, fontfamily="monospace",
                      transform=ax_stats.transAxes)
        ax_stats.text(0.38, y, vm, fontsize=10, color=BLUE,
                      fontfamily="monospace", transform=ax_stats.transAxes)
        ax_stats.text(0.68, y, vv, fontsize=10, color=ORANGE,
                      fontfamily="monospace", transform=ax_stats.transAxes)

        # Highlight better value
        try:
            fm, fv = float(vm.strip().rstrip('%')), float(vv.strip().rstrip('%'))
            if label in ("Return", "Sharpe", "Skew"):
                winner_x = 0.38 if fm >= fv else 0.68
            else:
                winner_x = 0.38 if fm <= fv else 0.68
            winner_c = BLUE if winner_x == 0.38 else ORANGE
        except ValueError:
            pass

    if idx < len(snapshots) - 1:
        iter_label = f"iter {idx}"
    else:
        iter_label = f"converged in {len(snapshots) - 1} iters / {solve_ms:.0f} ms"
    fig.suptitle(f"YAND-MVSK  —  {iter_label}",
                 fontsize=14, fontweight="bold")

anim = FuncAnimation(fig, update, frames=total_frames, interval=400)
anim.save("docs/demo.gif", writer=PillowWriter(fps=3), dpi=150)
plt.close()

print(f"Saved docs/demo.gif ({total_frames} frames)")

import os
size_kb = os.path.getsize("docs/demo.gif") / 1024
print(f"File size: {size_kb:.0f} KB")
