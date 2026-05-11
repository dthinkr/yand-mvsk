from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import numpy as np

from yand_mvsk._core import (
    MVSKResult,
    yand_mvsk_solve,
    crra_coefficients,
    check_convexity,
)


class EfficientMVSK:
    """MVSK portfolio optimizer with a pandas-friendly interface.

    Parameters
    ----------
    returns : array-like or DataFrame, shape (T, n)
        Asset return matrix.
    gamma : float
        CRRA risk aversion parameter. Higher = more conservative.
    c : array-like, shape (4,), optional
        Custom preference vector [c1, c2, c3, c4]. Overrides gamma.
    tickers : list of str, optional
        Asset names. Inferred from DataFrame columns if available.
    weight_bounds : tuple (min, max)
        Per-asset weight bounds. Default (0, None) = long-only, no upper limit.
    """

    def __init__(
        self,
        returns,
        gamma: float = 6.0,
        c: Optional[np.ndarray] = None,
        tickers: Optional[Sequence[str]] = None,
        weight_bounds: Tuple[float, Optional[float]] = (0, None),
    ):
        try:
            import pandas as pd
            if isinstance(returns, pd.DataFrame):
                if tickers is None:
                    tickers = list(returns.columns)
                returns = returns.values
        except ImportError:
            pass

        self.returns = np.asarray(returns, dtype=float)
        self.tickers = (
            list(tickers) if tickers is not None
            else [f"asset_{i}" for i in range(self.returns.shape[1])]
        )
        self.gamma = gamma
        self.c = np.asarray(c) if c is not None else crra_coefficients(gamma)

        if weight_bounds[1] is not None:
            raise NotImplementedError(
                "Upper weight bounds are not yet supported. "
                "Use weight_bounds=(lower, None) for now."
            )
        self.weight_bounds = weight_bounds

        self._result: Optional[MVSKResult] = None
        self._weights: Optional[np.ndarray] = None

    @classmethod
    def from_prices(cls, prices, gamma: float = 6.0, **kwargs) -> EfficientMVSK:
        """Create from a price DataFrame or array. Computes simple returns."""
        try:
            import pandas as pd
            if isinstance(prices, pd.DataFrame):
                returns = prices.pct_change().dropna()
                return cls(returns, gamma=gamma, **kwargs)
        except ImportError:
            pass
        prices = np.asarray(prices, dtype=float)
        returns = prices[1:] / prices[:-1] - 1.0
        return cls(returns, gamma=gamma, **kwargs)

    def optimize(self, **solver_kwargs) -> OrderedDict:
        """Run the YAND-MVSK optimizer.

        Extra keyword arguments are forwarded to ``yand_mvsk_solve``.

        Returns
        -------
        OrderedDict of {ticker: weight}.
        """
        lb = self.weight_bounds[0] if self.weight_bounds[0] is not None else 0.0
        tau = max(lb, 1e-8)
        solver_kwargs.setdefault("tau", tau)

        self._result = yand_mvsk_solve(self.returns, self.c, **solver_kwargs)
        self._weights = self._result.x
        return OrderedDict(zip(self.tickers, self._weights))

    def clean_weights(self, cutoff: float = 1e-4, rounding: int = 5) -> OrderedDict:
        """Zero out negligible weights and round.

        Parameters
        ----------
        cutoff : weights below this are set to zero.
        rounding : decimal places to round to.
        """
        if self._weights is None:
            raise RuntimeError("Call optimize() first.")
        w = self._weights.copy()
        w[w < cutoff] = 0.0
        if w.sum() > 0:
            w /= w.sum()
        w = np.round(w, rounding)
        return OrderedDict(zip(self.tickers, w))

    def portfolio_performance(
        self, verbose: bool = False, risk_free_rate: float = 0.0,
    ) -> dict:
        """Portfolio statistics including higher moments.

        Parameters
        ----------
        verbose : print the statistics.
        risk_free_rate : annualized risk-free rate for Sharpe ratio.

        Returns
        -------
        dict with return, volatility, sharpe, skewness, excess_kurtosis,
        mvsk_objective, kkt_residual, n_iter, converged.
        """
        if self._weights is None or self._result is None:
            raise RuntimeError("Call optimize() first.")

        port_ret = self.returns @ self._weights
        mu = port_ret.mean() * 252
        vol = port_ret.std() * np.sqrt(252)
        sharpe = (mu - risk_free_rate) / vol if vol > 1e-10 else 0.0

        centered = port_ret - port_ret.mean()
        m2 = np.mean(centered ** 2)
        skew = np.mean(centered ** 3) / (m2 ** 1.5) if m2 > 1e-30 else 0.0
        kurt = np.mean(centered ** 4) / (m2 ** 2) - 3.0 if m2 > 1e-30 else 0.0

        stats = {
            "return": mu,
            "volatility": vol,
            "sharpe": sharpe,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "mvsk_objective": self._result.f_val,
            "kkt_residual": self._result.kkt_residual,
            "n_iter": self._result.n_iter,
            "converged": self._result.converged,
        }

        if verbose:
            print(f"Expected annual return: {mu:>8.2%}")
            print(f"Annual volatility:      {vol:>8.2%}")
            print(f"Sharpe ratio:           {sharpe:>8.3f}")
            print(f"Skewness:               {skew:>8.3f}")
            print(f"Excess kurtosis:        {kurt:>8.3f}")
            print(f"MVSK objective:         {self._result.f_val:>12.6e}")
            print(f"Converged:              {self._result.converged} ({self._result.n_iter} iters)")

        return stats

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._weights

    @property
    def result(self) -> Optional[MVSKResult]:
        return self._result

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    @property
    def convex(self) -> bool:
        return check_convexity(self.c)
