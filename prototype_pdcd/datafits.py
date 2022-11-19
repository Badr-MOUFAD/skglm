import numpy as np
from numpy.linalg import norm

from skglm.utils import BST


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


class SqrtQuadratic:
    """||y - Xw||^2"""

    def __init__(self):
        pass

    def value(self, y, Xw):
        return norm(y - Xw, ord=2)

    def prox(self, w, step, y):
        """Block soft-thresholding of vector x at level u."""
        return BST(y - w, step)

    def prox_conjugate(self, z, step, y):
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)
