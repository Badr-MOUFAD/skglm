import numpy as np
from numba import float64

from numpy.linalg import norm
from skglm.utils import ST, ST_vec


class L1:
    """alpha * ||w||_1"""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return self.alpha * norm(w)

    def prox(self, w, step):
        return ST_vec(w, self.alpha * step)

    def prox_1D(self, w_j, step):
        return ST(w_j, self.alpha * step)

    def subdiff_distance(self, grad, w):
        n_features = len(w)
        subdiff_dist = np.zeros(n_features)

        for j in range(n_features):
            if w[j] == 0:
                subdiff_dist[j] = max(0, np.abs(grad[j]) - self.alpha)
            else:
                subdiff_dist[j] = abs(-grad[j] - np.sign(w[j]) * self.alpha)

        return subdiff_dist

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)
