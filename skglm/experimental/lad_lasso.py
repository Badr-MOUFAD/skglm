import numpy as np
from numpy.linalg import norm

from skglm.datafits import BaseDatafit
from skglm.utils.prox_funcs import ST_vec


class LAD(BaseDatafit):
    """Least Absolute Deviation (LAD) datafit.

    The datafit reads::

        ||y - Xw||_1
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = ()
        return spec

    def params_to_dict(self):
        return dict()

    def value(self, y, w, Xw):
        return norm(y - Xw, ord=1)

    def prox(self, w, step, y):
        """Prox of ||y - . ||_1 with step."""
        return y - ST_vec(y - w, step)

    def prox_conjugate(self, z, step, y):
        """Prox of ||y - . ||_1^* with step using Moreau decomposition."""
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def subdiff_distance(self, Xw, z, y):
        """Distance of ``z`` to subdiff of ||y - . ||_1 at ``Xw``."""
        # computation note: \partial ||y - . ||_1(Xw) = -\partial || . ||_1(y - Xw)
        y_minus_Xw = y - Xw

        max_distance = 0.
        for i in range(len(y)):

            # avoid comparing with 0 because of numerical errors
            # np.isclose is not used as Numba doesn't support it
            if abs(y_minus_Xw[i]) < 1e-10:
                distance_i = max(0, abs(z[i]) - 1)
            else:
                distance_i = abs(z[i] + np.sign(y_minus_Xw[i]))

            max_distance = max(max_distance, distance_i)

        return max_distance
