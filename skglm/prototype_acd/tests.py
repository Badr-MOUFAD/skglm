import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.utils.data import make_correlated_data

from sklearn.linear_model import Lasso
from skglm.prototype_acd.opt_anderson_cd import OptAndersonCD


def test_on_Lasso():
    rho = 0.1
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = rho * alpha_max

    quad_datafit = Quadratic()
    l1_penalty = L1(alpha)
    w, _ = OptAndersonCD().solve(X, y, quad_datafit, l1_penalty)

    lasso = Lasso(fit_intercept=False, alpha=alpha).fit(X, y)

    np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-6)


if __name__ == '__main__':
    test_on_Lasso()
