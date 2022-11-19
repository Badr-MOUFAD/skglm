import numpy as np
from numpy.linalg import norm

from skglm.utils import BST


class Quadratic:
    """(1/2) * ||y - Xw||^2"""

    def __init__(self):
        pass

    def value(self, y, Xw):
        return 0.5 * np.sum((y - Xw)**2)

    def prox(self, y, w, step):
        return (w + step * y) / (1 + step)

    def prox_conjugate(self, y, z, step):
        inv_step = 1 / step
        return z - step * self.prox(y, inv_step * z, inv_step)


class SqrtQuadratic:
    """||y - Xw||^2"""

    def __init__(self):
        pass

    def value(self, y, Xw):
        return norm(y - Xw, ord=2)

    def prox(self, y, w, step):
        """Block soft-thresholding of vector x at level u."""
        return BST(y - w, step)

    def prox_conjugate(self, y, z, step):
        inv_step = 1 / step
        return z - step * self.prox(y, inv_step * z, inv_step)
