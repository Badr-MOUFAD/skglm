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
