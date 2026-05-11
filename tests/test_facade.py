import numpy as np
import pytest
from yand_mvsk import EfficientMVSK


def _synthetic_returns(n=20, T=252, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.02 + 0.0003


class TestEfficientMVSK:
    def test_numpy_input(self):
        R = _synthetic_returns()
        ef = EfficientMVSK(R, gamma=6)
        weights = ef.optimize()
        assert len(weights) == 20
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert ef.result.converged

    def test_tickers(self):
        R = _synthetic_returns(n=5)
        tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
        ef = EfficientMVSK(R, gamma=6, tickers=tickers)
        weights = ef.optimize()
        assert list(weights.keys()) == tickers

    def test_from_prices_numpy(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.cumprod(1 + rng.standard_normal((253, 10)) * 0.02, axis=0)
        ef = EfficientMVSK.from_prices(prices, gamma=6)
        weights = ef.optimize()
        assert len(weights) == 10
        assert ef.result.converged

    def test_clean_weights(self):
        R = _synthetic_returns()
        ef = EfficientMVSK(R, gamma=6)
        ef.optimize()
        cleaned = ef.clean_weights(cutoff=0.01)
        assert all(v == 0 or v >= 0.01 for v in cleaned.values())
        assert abs(sum(cleaned.values()) - 1.0) < 0.01

    def test_portfolio_performance(self):
        R = _synthetic_returns()
        ef = EfficientMVSK(R, gamma=6)
        ef.optimize()
        stats = ef.portfolio_performance()
        assert "return" in stats
        assert "volatility" in stats
        assert "sharpe" in stats
        assert "skewness" in stats
        assert "excess_kurtosis" in stats
        assert "mvsk_objective" in stats
        assert stats["converged"]

    def test_optimize_before_performance_raises(self):
        R = _synthetic_returns()
        ef = EfficientMVSK(R, gamma=6)
        with pytest.raises(RuntimeError):
            ef.portfolio_performance()
        with pytest.raises(RuntimeError):
            ef.clean_weights()

    def test_convex_property(self):
        R = _synthetic_returns()
        ef = EfficientMVSK(R, gamma=6)
        assert ef.convex

    def test_custom_coefficients(self):
        R = _synthetic_returns()
        c = np.array([1.0, 3.0, 7.0, 15.0])
        ef = EfficientMVSK(R, c=c)
        weights = ef.optimize()
        assert len(weights) == 20

    def test_upper_bound_raises(self):
        R = _synthetic_returns()
        with pytest.raises(NotImplementedError, match="Upper weight bounds"):
            EfficientMVSK(R, gamma=6, weight_bounds=(0, 0.1))


class TestFromPricesDataFrame:
    @pytest.fixture
    def prices_df(self):
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(42)
        data = 100 * np.cumprod(1 + rng.standard_normal((253, 5)) * 0.02, axis=0)
        return pd.DataFrame(data, columns=["SPY", "QQQ", "TLT", "GLD", "IEF"])

    def test_from_prices_dataframe(self, prices_df):
        ef = EfficientMVSK.from_prices(prices_df, gamma=6)
        weights = ef.optimize()
        assert list(weights.keys()) == ["SPY", "QQQ", "TLT", "GLD", "IEF"]
        assert ef.result.converged

    def test_returns_dataframe(self, prices_df):
        returns = prices_df.pct_change().dropna()
        ef = EfficientMVSK(returns, gamma=6)
        weights = ef.optimize()
        assert list(weights.keys()) == ["SPY", "QQQ", "TLT", "GLD", "IEF"]

    def test_verbose_output(self, prices_df, capsys):
        ef = EfficientMVSK.from_prices(prices_df, gamma=6)
        ef.optimize()
        ef.portfolio_performance(verbose=True)
        captured = capsys.readouterr()
        assert "Sharpe" in captured.out
        assert "Skewness" in captured.out
