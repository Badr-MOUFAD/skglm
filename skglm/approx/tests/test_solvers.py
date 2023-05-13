import numpy as np
from numpy.linalg import norm
from skglm.utils.data import make_correlated_data

from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.approx.solver import Approx

from skglm import Lasso


def test_approx_lasso():
    random_state = 135
    n_samples, n_features = 10, 100

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    lmbd_max = norm(X.T @ y, ord=np.inf) / n_samples
    lmbd = 1e-2 * lmbd_max

    datafit = Quadratic()
    penalty = L1(lmbd)

    w = Approx(verbose=1, max_iter=1000).solve(X, y, datafit, penalty)

    estimator = Lasso(lmbd, fit_intercept=False)
    estimator.fit(X, y)

    print(Quadratic().value(y, w, X @ estimator.coef_) + L1(lmbd).value(estimator.coef_))


if __name__ == "__main__":
    test_approx_lasso()
