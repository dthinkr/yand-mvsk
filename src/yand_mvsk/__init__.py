"""
YAND-MVSK: portfolio optimization beyond mean-variance.

Minimize  f(x) = -c1*mean + c2*variance - c3*skewness + c4*kurtosis
over the simplex using Yau's Affine-Normal Descent.
"""

from yand_mvsk._core import (
    MVSKOracle,
    MVSKResult,
    yand_mvsk_solve,
    crra_coefficients,
    check_convexity,
)

__all__ = [
    "MVSKOracle",
    "MVSKResult",
    "yand_mvsk_solve",
    "crra_coefficients",
    "check_convexity",
]
