import numpy as np
import pytest
from yand_mvsk import yand_mvsk_solve, crra_coefficients, check_convexity, MVSKOracle


def _synthetic_returns(n=10, T=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.02 + 0.0003


class TestConvexity:
    def test_crra_gamma6(self):
        c = crra_coefficients(6.0)
        assert check_convexity(c)

    def test_crra_all_positive_gamma(self):
        for gamma in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            assert check_convexity(crra_coefficients(gamma))


class TestOracle:
    def test_gradient_finite_difference(self):
        R = _synthetic_returns()
        c = crra_coefficients(6.0)
        oracle = MVSKOracle(R, c)
        x = np.ones(10) / 10
        _, g = oracle.value_gradient(x)

        eps = 1e-7
        g_fd = np.zeros(10)
        for i in range(10):
            e = np.zeros(10)
            e[i] = eps
            g_fd[i] = (oracle.value(x + e) - oracle.value(x - e)) / (2 * eps)

        np.testing.assert_allclose(g, g_fd, rtol=1e-5)

    def test_hessian_vec_finite_difference(self):
        R = _synthetic_returns()
        c = crra_coefficients(6.0)
        oracle = MVSKOracle(R, c)
        x = np.ones(10) / 10
        v = np.random.default_rng(0).standard_normal(10)

        hv = oracle.hessian_vec(x, v)

        eps = 1e-6
        _, g_plus = oracle.value_gradient(x + eps * v)
        _, g_minus = oracle.value_gradient(x - eps * v)
        hv_fd = (g_plus - g_minus) / (2 * eps)

        np.testing.assert_allclose(hv, hv_fd, rtol=1e-4)


class TestSolver:
    def test_convergence_small(self):
        R = _synthetic_returns(n=10, T=100)
        c = crra_coefficients(6.0)
        res = yand_mvsk_solve(R, c)
        assert res.converged
        assert res.kkt_residual < 1e-6

    def test_simplex_constraint(self):
        R = _synthetic_returns(n=20, T=200)
        c = crra_coefficients(6.0)
        res = yand_mvsk_solve(R, c)
        assert abs(res.x.sum() - 1.0) < 1e-12
        assert np.all(res.x >= 0)

    def test_beats_equal_weight(self):
        R = _synthetic_returns(n=20, T=200)
        c = crra_coefficients(6.0)
        res = yand_mvsk_solve(R, c)
        oracle = MVSKOracle(R, c)
        f_ew = oracle.value(np.ones(20) / 20)
        assert res.f_val <= f_ew

    def test_matches_scipy(self):
        from scipy.optimize import minimize

        R = _synthetic_returns(n=5, T=50)
        c = crra_coefficients(6.0)
        oracle = MVSKOracle(R, c)

        res_yand = yand_mvsk_solve(R, c)

        from scipy.optimize import LinearConstraint
        cons = LinearConstraint(np.ones(5), 1.0, 1.0)
        bounds = [(1e-8, None)] * 5
        res_scipy = minimize(oracle.value, np.ones(5)/5, bounds=bounds, constraints=cons)

        assert abs(res_yand.f_val - res_scipy.fun) < 1e-6

    def test_armijo_line_search(self):
        R = _synthetic_returns(n=8, T=100)
        c = crra_coefficients(6.0)
        res = yand_mvsk_solve(R, c, line_search='armijo')
        assert res.converged

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_scales(self, n):
        R = _synthetic_returns(n=n, T=252, seed=123)
        c = crra_coefficients(6.0)
        res = yand_mvsk_solve(R, c, max_iter=300)
        assert res.kkt_residual < 1e-4
