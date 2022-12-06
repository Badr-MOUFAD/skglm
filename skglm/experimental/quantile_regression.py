import numpy as np
from numba import float64
from skglm.datafits import BaseDatafit
from skglm.utils.prox_funcs import ST_vec


class Pinball(BaseDatafit):
    r"""Pinball datafit.

    The datafit reads::

        quantile * max(y - Xw, 0) + (1 - quantile) * max(Xw - y, 0)

    such that quantile in ``[0, 1]``.

    Parameters
    ----------
    quantile : float
        Quantile must be in ``[0, 1]``. When ``quantile=0.5``,
        the datafit becomes a Least Absolute Deviation (LAD) datafit.
    """

    def __init__(self, quantile):
        self.quantile = quantile

    def value(self, y, w, Xw):
        # implementation of:
        # github.com/benchopt/benchmark_quantile_regression/blob/main/objective.py
        quantile = self.quantile

        residual = y - Xw
        sign = residual >= 0

        loss = quantile * sign * residual - (1 - quantile) * (1 - sign) * residual
        return np.sum(loss)

    def prox(self, w, step, y):
        """Prox of ||y - . || with step ``step``."""
        shifted_w = w - (2 * self.quantile - 1) * step
        return y - ST_vec(y - shifted_w, step / 2)

    def prox_conjugate(self, z, step, y):
        """Prox of ||y - . ||^* with step ``step``."""
        # using Moreau decomposition
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def subdiff_distance(self, Xw, z, y):
        """Distance of ``z`` to subdiff of ||y - . ||_1 at ``Xw``."""
        # computation note: \partial ||y - . ||_1(Xw) = -\partial || . ||_1(y - Xw)
        y_minus_Xw = y - Xw
        shifting_cst = self.alpha - 1/2

        max_distance = 0.
        for i in range(len(y)):

            if y_minus_Xw[i] == 0.:
                distance_i = max(0, abs(z[i] - shifting_cst) - 1)
            else:
                distance_i = abs(z[i] + shifting_cst + np.sign(y_minus_Xw[i]))

            max_distance = max(max_distance, distance_i)

        return max_distance

    def get_spec(self):
        spec = (
            ('quantile', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(quantile=self.quantile)
