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
    asset_crra_scores,
    check_convexity,
)
from yand_mvsk._facade import EfficientMVSK

__all__ = [
    "EfficientMVSK",
    "MVSKOracle",
    "MVSKResult",
    "yand_mvsk_solve",
    "crra_coefficients",
    "asset_crra_scores",
    "check_convexity",
]
