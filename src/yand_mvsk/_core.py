"""
YAND-MVSK: Yau's Affine-Normal Descent for Mean-Variance-Skewness-Kurtosis
Portfolio Optimization.

Reference: arXiv:2604.25378 (Niu, Yau et al., 2026).

Stores only (mu, A, c) — no explicit coskewness/cokurtosis tensors.
All derivative actions computed via matrix-vector products with A and A^T.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, solve
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MVSKResult:
    """Result of YAND-MVSK optimization.

    Attributes
    ----------
    x : portfolio weights on the simplex.
    f_val : objective value at x.
    kkt_residual : first-order stationarity measure.
    n_iter : number of iterations used.
    converged : whether KKT residual reached the tolerance.
    history : per-iteration objective values.
    """
    x: np.ndarray
    f_val: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)


class MVSKOracle:
    """Exact sample oracle for the MVSK objective (Proposition 1, eqs 2-5).

    Parameters
    ----------
    R : (T, n) return matrix — T observations of n assets.
    c : (4,) preference vector [c1, c2, c3, c4].
    """

    def __init__(self, R: np.ndarray, c: np.ndarray):
        self.T, self.n = R.shape
        self.mu = R.mean(axis=0)
        self.A = R - self.mu[np.newaxis, :]
        self.c = c.astype(float)
        self._z: Optional[np.ndarray] = None
        self._x_id: Optional[int] = None

    def _ensure_z(self, x: np.ndarray):
        xid = id(x)
        if self._x_id != xid:
            self._z = self.A @ x
            self._x_id = xid

    def value_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute f(x) and nabla f(x) in one pass."""
        self._z = self.A @ x
        self._x_id = id(x)
        z = self._z
        c1, c2, c3, c4 = self.c
        T = self.T
        z2 = z * z
        z3 = z * z2

        f = (-c1 * (self.mu @ x)
             + (c2 / T) * np.sum(z2)
             - (c3 / T) * np.sum(z3)
             + (c4 / T) * np.sum(z2 * z2))

        g = (-c1 * self.mu
             + (2 * c2 / T) * (self.A.T @ z)
             - (3 * c3 / T) * (self.A.T @ z2)
             + (4 * c4 / T) * (self.A.T @ z3))

        return f, g

    def value(self, x: np.ndarray) -> float:
        """Compute f(x) only."""
        z = self.A @ x
        c1, c2, c3, c4 = self.c
        T = self.T
        z2 = z * z
        return (-c1 * (self.mu @ x)
                + (c2 / T) * np.sum(z2)
                - (c3 / T) * np.sum(z * z2)
                + (c4 / T) * np.sum(z2 * z2))

    def hessian_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product nabla^2 f(x) v."""
        self._ensure_z(x)
        z = self._z
        c1, c2, c3, c4 = self.c
        T = self.T
        Av = self.A @ v
        return ((2 * c2 / T) * (self.A.T @ Av)
                - (6 * c3 / T) * (self.A.T @ (z * Av))
                + (12 * c4 / T) * (self.A.T @ (z * z * Av)))

    def quartic_coefficients(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Quartic line-search coefficients A0..A4 (Proposition 22)."""
        self._ensure_z(x)
        z = self._z
        w = self.A @ d
        c1, c2, c3, c4 = self.c

        z2, z3 = z * z, z * z * z
        w2, w3 = w * w, w * w * w

        s11 = np.dot(z, w) / self.T
        s21 = np.dot(z2, w) / self.T
        s31 = np.dot(z3, w) / self.T
        s02 = np.dot(w, w) / self.T
        s12 = np.dot(z, w2) / self.T
        s22 = np.dot(z2, w2) / self.T
        s03 = np.dot(w, w2) / self.T
        s13 = np.dot(z, w3) / self.T
        s04 = np.dot(w, w3) / self.T

        A0 = self.value(x)
        A1 = -c1 * (self.mu @ d) + 2*c2*s11 - 3*c3*s21 + 4*c4*s31
        A2 = c2*s02 - 3*c3*s12 + 6*c4*s22
        A3 = -c3*s03 + 4*c4*s13
        A4 = c4*s04

        return np.array([A0, A1, A2, A3, A4])


# ---------------------------------------------------------------------------
#  Simplex tangent-space reduction
# ---------------------------------------------------------------------------

class SimplexReduction:
    def __init__(self, n: int, xref: Optional[np.ndarray] = None):
        self.n = n
        self.xref = xref if xref is not None else np.ones(n) / n
        self.U = self._build_tangent_basis(n)

    @staticmethod
    def _build_tangent_basis(n: int) -> np.ndarray:
        e = np.ones(n) / np.sqrt(n)
        e1 = np.zeros(n)
        e1[0] = 1.0
        v = e + e1
        v_norm_sq = np.dot(v, v)
        U = np.eye(n)[:, 1:] - 2.0 * np.outer(v, v[1:]) / v_norm_sq
        return U


def _build_householder_frame(nu: np.ndarray) -> np.ndarray:
    m = len(nu)
    if m <= 1:
        return np.zeros((m, 0))
    e1 = np.zeros(m)
    e1[0] = 1.0
    s = 1.0 if nu[0] >= 0 else -1.0
    v = nu + s * e1
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-30:
        v = nu - s * e1
        v_norm_sq = np.dot(v, v)
    Q = np.eye(m)[:, 1:] - 2.0 * np.outer(v, v[1:]) / v_norm_sq
    return Q


# ---------------------------------------------------------------------------
#  Quartic exact line search (Proposition 22)
# ---------------------------------------------------------------------------

def _quartic_line_search(coeffs: np.ndarray, alpha_max: float) -> float:
    A0, A1, A2, A3, A4 = coeffs

    if alpha_max <= 0:
        return 0.0

    def phi(a):
        return A0 + a * (A1 + a * (A2 + a * (A3 + a * A4)))

    candidates = [0.0, alpha_max]

    if abs(A4) > 1e-30:
        roots = np.roots([4*A4, 3*A3, 2*A2, A1])
        for r in roots:
            if np.isreal(r):
                r_real = float(np.real(r))
                if 1e-15 < r_real < alpha_max - 1e-15:
                    candidates.append(r_real)
    elif abs(A3) > 1e-30:
        disc = 4*A2*A2 - 12*A3*A1
        if disc >= 0:
            sq = np.sqrt(disc)
            for r in [(-2*A2 + sq) / (6*A3), (-2*A2 - sq) / (6*A3)]:
                if 1e-15 < r < alpha_max - 1e-15:
                    candidates.append(r)
    elif abs(A2) > 1e-30:
        r = -A1 / (2*A2)
        if 1e-15 < r < alpha_max - 1e-15:
            candidates.append(r)

    return min(candidates, key=phi)


# ---------------------------------------------------------------------------
#  Feasibility and projection
# ---------------------------------------------------------------------------

def _feasibility_cap(x: np.ndarray, d: np.ndarray, tau: float) -> float:
    neg = d < -1e-15
    if not np.any(neg):
        return 1e10
    return max(float(np.min((x[neg] - tau) / (-d[neg]))), 0.0)


def _project_simplex(x: np.ndarray, tau: float = 0.0) -> np.ndarray:
    n = len(x)
    budget = 1.0 - n * tau
    if budget <= 1e-15:
        return np.ones(n) / n
    z = x - tau
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    rho = 0
    for j in range(n):
        if z_sorted[j] > (cumsum[j] - budget) / (j + 1):
            rho = j + 1
    if rho == 0:
        return np.ones(n) / n
    theta = (cumsum[rho - 1] - budget) / rho
    return np.maximum(z - theta, 0.0) + tau


def _kkt_residual(x: np.ndarray, g: np.ndarray, tau: float) -> float:
    return norm(x - _project_simplex(x - g, tau))


# ---------------------------------------------------------------------------
#  PCG tangent solve
# ---------------------------------------------------------------------------

def _pcg_solve(
    oracle: MVSKOracle, xk: np.ndarray, U: np.ndarray, Q: np.ndarray,
    rhs: np.ndarray, lam: float, tol: float, maxit: int,
) -> np.ndarray:
    m = Q.shape[1]
    u = np.zeros(m)
    r = rhs.copy()
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(min(maxit, m)):
        UQp = U @ (Q @ p)
        Hp = Q.T @ (U.T @ oracle.hessian_vec(xk, UQp)) + lam * p
        pHp = np.dot(p, Hp)
        if pHp <= 1e-30:
            break
        alpha = rsold / pHp
        u += alpha * p
        r -= alpha * Hp
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return u


# ---------------------------------------------------------------------------
#  Step helpers
# ---------------------------------------------------------------------------

def _tangent_gradient(gk: np.ndarray, n: int) -> np.ndarray:
    d = -(gk - (gk.sum() / n) * np.ones(n))
    dn = norm(d)
    return d / dn if dn > 1e-30 else d


def _try_step(
    xk: np.ndarray, dk: np.ndarray, fk: float,
    oracle: MVSKOracle, tau: float,
) -> Optional[np.ndarray]:
    alpha_max = _feasibility_cap(xk, dk, tau)
    if alpha_max <= 1e-15:
        return None
    coeffs = oracle.quartic_coefficients(xk, dk)
    alpha_k = _quartic_line_search(coeffs, alpha_max)
    x_trial = xk + alpha_k * dk
    if np.any(x_trial < tau) or abs(x_trial.sum() - 1.0) > 1e-10:
        x_trial = _project_simplex(x_trial, tau)
    if oracle.value(x_trial) < fk:
        return x_trial
    return None


def _try_step_armijo(
    xk: np.ndarray, dk: np.ndarray, gk: np.ndarray, fk: float,
    oracle: MVSKOracle, tau: float,
    sigma: float = 1e-4, beta: float = 0.5,
) -> Optional[np.ndarray]:
    alpha_max = _feasibility_cap(xk, dk, tau)
    if alpha_max <= 1e-15:
        return None
    descent = gk @ dk
    if descent >= 0:
        return None
    alpha = min(1.0, alpha_max)
    for _ in range(30):
        x_trial = xk + alpha * dk
        if np.any(x_trial < tau) or abs(x_trial.sum() - 1.0) > 1e-10:
            x_trial = _project_simplex(x_trial, tau)
        if oracle.value(x_trial) <= fk + sigma * alpha * descent:
            return x_trial
        alpha *= beta
    return None


def _projected_step(
    xk: np.ndarray, dk: np.ndarray, gk: np.ndarray, fk: float,
    oracle: MVSKOracle, tau: float, eta_list: list[float],
) -> Optional[np.ndarray]:
    best_x: Optional[np.ndarray] = None
    best_f = fk

    for eta in eta_list:
        x_bar = _project_simplex(xk + eta * dk, tau)
        d_bar = x_bar - xk
        if norm(d_bar) < 1e-15 or gk @ d_bar >= 0:
            continue
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            x_try = xk + alpha * d_bar
            f_try = oracle.value(x_try)
            if f_try < best_f:
                best_x = x_try.copy()
                best_f = f_try

    return best_x


def _gradient_step(
    xk: np.ndarray, gk: np.ndarray, fk: float,
    oracle: MVSKOracle, n: int, tau: float,
) -> Optional[np.ndarray]:
    x_proj = _project_simplex(xk - gk, tau)
    dk_pg = x_proj - xk
    dk_pg_norm = norm(dk_pg)
    if dk_pg_norm < 1e-15:
        return None
    alpha = 1.0
    for _ in range(25):
        x_try = xk + alpha * dk_pg
        if np.any(x_try < tau) or abs(x_try.sum() - 1.0) > 1e-10:
            x_try = _project_simplex(x_try, tau)
        if oracle.value(x_try) < fk:
            return x_try
        alpha *= 0.5
    return None


# ---------------------------------------------------------------------------
#  Log-determinant correction
# ---------------------------------------------------------------------------

def _logdet_correction(
    oracle: MVSKOracle, xk: np.ndarray, U: np.ndarray, Q: np.ndarray,
    H_T_reg: np.ndarray, m: int,
) -> np.ndarray:
    c1, c2, c3, c4 = oracle.c
    T = oracle.T

    oracle._ensure_z(xk)
    z = oracle._z

    UQ = U @ Q
    P = oracle.A @ UQ

    w = (-6.0 * c3 / T) + (24.0 * c4 / T) * z

    try:
        H_inv = np.linalg.inv(H_T_reg)
    except np.linalg.LinAlgError:
        return np.zeros(m)

    V = P @ H_inv
    q = np.sum(P * V, axis=1)

    return P.T @ (w * q)


def _logdet_correction_pcg(
    oracle: MVSKOracle, xk: np.ndarray, U: np.ndarray, Q: np.ndarray,
    lam: float, m: int, krylov_tol: float, krylov_maxit: int,
    n_probes: int = 5,
) -> np.ndarray:
    c1, c2, c3, c4 = oracle.c
    T = oracle.T
    oracle._ensure_z(xk)
    z = oracle._z

    UQ = U @ Q
    P = oracle.A @ UQ
    w = (-6.0 * c3 / T) + (24.0 * c4 / T) * z

    rng = np.random.default_rng(int(abs(xk[0] * 1e8)) % (2**31))
    q_approx = np.zeros(T)
    for _ in range(n_probes):
        r = rng.choice([-1.0, 1.0], size=m)
        v = _pcg_solve(oracle, xk, U, Q, r, lam, krylov_tol, krylov_maxit)
        q_approx += (P @ r) * (P @ v)
    q_approx /= n_probes

    return P.T @ (w * q_approx)


# ---------------------------------------------------------------------------
#  YAND direction with face continuation
# ---------------------------------------------------------------------------

def _yand_direction_on_face(
    xk: np.ndarray, gk: np.ndarray, free: np.ndarray,
    oracle: MVSKOracle, U_full: np.ndarray, lam: float,
    use_pcg: bool, n: int, krylov_tol: float, krylov_maxit: int,
) -> np.ndarray:
    n_free = int(np.sum(free))

    if n_free == n:
        U = U_full
    else:
        U = SimplexReduction._build_tangent_basis(n_free)
        idx_free = np.where(free)[0]
        U_ambient = np.zeros((n, n_free - 1))
        U_ambient[idx_free, :] = U
        U = U_ambient

    g_bar = U.T @ gk
    g_norm = norm(g_bar)

    if g_norm <= 1e-15:
        return _tangent_gradient(gk, n)

    nu = g_bar / g_norm
    Q = _build_householder_frame(nu)

    if Q.shape[1] == 0:
        dk_y = -nu
    else:
        Unu = U @ nu
        h = Q.T @ (U.T @ oracle.hessian_vec(xk, Unu))
        m = Q.shape[1]

        if use_pcg and m > 500:
            a = _logdet_correction_pcg(
                oracle, xk, U, Q, lam, m,
                krylov_tol, krylov_maxit,
            )
            rhs = h - (g_norm / (m + 1)) * a
            u_sol = _pcg_solve(oracle, xk, U, Q, rhs, lam,
                               krylov_tol, krylov_maxit)
        else:
            H_T = np.empty((m, m))
            for j in range(m):
                Uqj = U @ Q[:, j]
                H_T[:, j] = Q.T @ (U.T @ oracle.hessian_vec(xk, Uqj))
            H_T_reg = H_T + lam * np.eye(m)

            a = _logdet_correction(oracle, xk, U, Q, H_T_reg, m)
            rhs = h - (g_norm / (m + 1)) * a

            try:
                u_sol = solve(H_T_reg, rhs)
            except np.linalg.LinAlgError:
                u_sol = np.linalg.lstsq(H_T_reg, rhs, rcond=None)[0]

        dk_y = Q @ u_sol - nu

    dk = U @ dk_y

    if gk @ dk >= 0:
        dk = _tangent_gradient(gk, n)

    return dk


# ---------------------------------------------------------------------------
#  CRRA utilities
# ---------------------------------------------------------------------------

def crra_coefficients(gamma: float) -> np.ndarray:
    """CRRA preference coefficients for risk aversion parameter gamma.

    Returns c = (1, gamma/2, gamma(gamma+1)/6, gamma(gamma+1)(gamma+2)/24).
    """
    return np.array([
        1.0,
        gamma / 2.0,
        gamma * (gamma + 1) / 6.0,
        gamma * (gamma + 1) * (gamma + 2) / 24.0,
    ])


def check_convexity(c: np.ndarray) -> bool:
    """Check the sufficient convexity condition: c4 > 0 and 8*c2*c4 > 3*c3^2."""
    _, c2, c3, c4 = c
    return c4 > 0 and 8 * c2 * c4 > 3 * c3**2


# ---------------------------------------------------------------------------
#  Main solver
# ---------------------------------------------------------------------------

def yand_mvsk_solve(
    R: np.ndarray,
    c: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tau: float = 1e-8,
    tol: float = 1e-6,
    max_iter: int = 300,
    lam: float = 1e-4,
    use_pcg: bool = False,
    krylov_tol: float = 1e-3,
    krylov_maxit: int = 15,
    line_search: str = 'quartic',
    verbose: bool = False,
) -> MVSKResult:
    """Solve the MVSK portfolio optimization problem.

    Minimizes  f(x) = -c1*m1 + c2*m2 - c3*m3 + c4*m4
    over the simplex {x >= tau, sum(x) = 1}.

    Parameters
    ----------
    R : (T, n) return matrix.
    c : (4,) preference vector [c1, c2, c3, c4].
    x0 : initial portfolio (default: equal-weight).
    tau : lower bound on each weight.
    tol : KKT convergence tolerance.
    max_iter : iteration budget.
    lam : Tikhonov regularization for the tangent Hessian.
    use_pcg : use conjugate gradients instead of direct solve.
    krylov_tol, krylov_maxit : CG solver parameters.
    line_search : 'quartic' (exact) or 'armijo' (backtracking).
    verbose : print per-iteration diagnostics.

    Returns
    -------
    MVSKResult with optimal weights, objective, KKT residual, and history.
    """
    _, n = R.shape

    oracle = MVSKOracle(R, c)

    if x0 is None:
        x0 = np.ones(n) / n
    xk = np.clip(x0, tau, None)
    xk /= xk.sum()

    reduction = SimplexReduction(n, xref=xk.copy())
    U = reduction.U

    history: list[float] = []
    stall_count = 0
    eta_list = [0.05, 0.045, 0.02, 0.1, 0.2]

    for k in range(max_iter):
        fk, gk = oracle.value_gradient(xk)
        history.append(fk)
        kkt = _kkt_residual(xk, gk, tau)

        if verbose and k % 10 == 0:
            print(f"  iter {k:4d}  f={fk:+.8e}  KKT={kkt:.4e}")

        if kkt <= tol:
            if verbose:
                print(f"  converged at iter {k}, KKT={kkt:.4e}")
            return MVSKResult(xk, fk, kkt, k, True, history)

        boundary_thr = max(tau * 100, 1e-6)
        free = xk > boundary_thr
        boundary_but_wants_in = (~free) & (gk < gk[free].min() if np.any(free) else np.zeros(n, dtype=bool))
        free = free | boundary_but_wants_in
        n_free = int(np.sum(free))

        if n_free >= 2:
            dk = _yand_direction_on_face(
                xk, gk, free, oracle, U, lam, use_pcg, n,
                krylov_tol, krylov_maxit,
            )
        else:
            dk = _tangent_gradient(gk, n)

        candidates = []

        if line_search == 'armijo':
            x_ls = _try_step_armijo(xk, dk, gk, fk, oracle, tau)
        else:
            x_ls = _try_step(xk, dk, fk, oracle, tau)
        if x_ls is not None:
            candidates.append((oracle.value(x_ls), x_ls))

        x_proj = _projected_step(xk, dk, gk, fk, oracle, tau, eta_list)
        if x_proj is not None:
            candidates.append((oracle.value(x_proj), x_proj))

        x_grad = _gradient_step(xk, gk, fk, oracle, n, tau)
        if x_grad is not None:
            candidates.append((oracle.value(x_grad), x_grad))

        if candidates:
            x_new = min(candidates, key=lambda t: t[0])[1]
        else:
            x_new = None

        if x_new is not None:
            xk = x_new
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 5:
                if verbose:
                    print(f"  stalled at iter {k}, KKT={kkt:.4e}")
                return MVSKResult(xk, fk, kkt, k, kkt <= tol, history)

    fk, gk = oracle.value_gradient(xk)
    kkt = _kkt_residual(xk, gk, tau)
    history.append(fk)
    if verbose:
        print(f"  max_iter reached, KKT={kkt:.4e}")
    return MVSKResult(xk, fk, kkt, max_iter, kkt <= tol, history)
