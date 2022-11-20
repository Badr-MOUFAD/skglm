import numpy as np
from numpy.linalg import norm

from skglm.utils import BST, proj_L2ball


class Quadratic:
    """(1/2) * ||y - Xw||^2"""

    def __init__(self):
        pass

    def value(self, y, Xw):
        return 0.5 * np.sum((y - Xw)**2)

    def prox(self, w, step, y):
        return (w + step * y) / (1 + step)

    def prox_conjugate(self, z, step, y):
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def subdiff_distance(self, Xw, z, y):
        return norm(Xw - y - z, ord=2)

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()


class SqrtQuadratic:
    """||y - Xw||"""

    def __init__(self):
        pass

    def value(self, y, Xw):
        return norm(y - Xw, ord=2)

    def prox(self, w, step, y):
        """Block soft-thresholding of vector x at level u."""
        return y - BST(y - w, step)

    def prox_conjugate(self, z, step, y):
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def subdiff_distance(self, Xw, z, y):
        y_minus_Xw = y - Xw

        if np.any(y_minus_Xw):
            return norm(z + y_minus_Xw / norm(y_minus_Xw))

        return norm(z - proj_L2ball(z))

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()
