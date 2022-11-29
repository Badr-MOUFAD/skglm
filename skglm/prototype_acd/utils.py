import numpy as np

from numba import int32, float64
from numba.experimental import jitclass


@jitclass(
    (('K', int32),
     ('current_iter', int32),
     ('arr_w', float64[:, ::1]))
)
class AndersonAcceleration:
    """Abstraction of Anderson Acceleration.

    Extrapolate the asymptotic VAR ``w`` and ``Xw``
    based on ``K`` previous iterations.

    Parameters
    ----------
    K : int
        Number of previous iterates to consider for extrapolation.
    """

    def __init__(self, K, n_features):
        self.K, self.current_iter = K, 0
        self.arr_w = np.zeros((n_features, K+1))

    def extrapolate(self, w):
        """Return w, Xw, and a bool indicating whether they were extrapolated."""

        if self.current_iter <= self.K:
            self.arr_w[:, self.current_iter] = w
            self.current_iter += 1
            return w, False

        # compute residuals
        U = np.diff(self.arr_w)

        # compute extrapolation coefs
        try:
            inv_UTU_ones = np.linalg.solve(U.T @ U, np.ones(self.K))
        except Exception:
            return w, False
        finally:
            self.current_iter = 0

        # extrapolate
        C = inv_UTU_ones / np.sum(inv_UTU_ones)
        # floating point errors may cause w and Xw to disagree
        return self.arr_w[:, 1:] @ C, True


def test_anderson_acceleration():
    from numpy.linalg import norm

    # VAR: w = rho * w + 1 with |rho| < 1
    # converges to w_star = 1 / (1 - rho)
    max_iter, tol = 1000, 1e-9
    n_features = 2
    rho = np.array([0.5, 0.8])
    w_star = 1 / (1 - rho)
    X = np.diag([2, 5])

    # with acceleration
    acc = AndersonAcceleration(K=5, n_samples=X.shape[0], n_features=X.shape[1])
    n_iter_acc = 0
    w = np.ones(n_features)
    Xw = X @ w
    for i in range(max_iter):
        w, Xw, _ = acc.extrapolate(w, Xw)
        w = rho * w + 1
        Xw = X @ w

        if norm(w - w_star, ord=np.inf) < tol:
            n_iter_acc = i
            break

    # without acceleration
    n_iter = 0
    w = np.ones(n_features)
    for i in range(max_iter):
        w = rho * w + 1

        if norm(w - w_star, ord=np.inf) < tol:
            n_iter = i
            break

    np.testing.assert_allclose(w, w_star)
    np.testing.assert_allclose(Xw, X @ w_star)

    print(n_iter_acc)
    print(n_iter)
    # np.testing.assert_array_equal(n_iter_acc, 13)
    np.testing.assert_array_equal(n_iter, 99)


if __name__ == "__main__":
    import time

    w1 = np.random.randn(10)
    w2 = np.random.randn(5)

    start = time.time()
    aa1 = AndersonAcceleration(5, 10, 10)
    aa1.extrapolate(w1, w1)
    mid = time.time()
    aa2 = AndersonAcceleration(5, 5, 5)
    aa2.extrapolate(w2, w2)
    end = time.time()

    print(mid - start)
    print(end - mid)

    start = time.time()
    aa1.extrapolate(w1, w1)
    mid = time.time()
    aa2.extrapolate(w2, w2)
    end = time.time()

    print(mid - start)
    print(end - mid)
