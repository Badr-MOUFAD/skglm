import pytest
from itertools import product

import numpy as np
from numpy.linalg import norm

from skglm.prototype_pdcd.penalties import L1
from skglm.prototype_pdcd.algorithms import ChambollePock, PDCD, PDCD_WS
from skglm.prototype_pdcd.datafits import Quadratic, SqrtQuadratic

from sklearn.linear_model import Lasso
from skglm.experimental import SqrtLasso
from skglm.utils import make_correlated_data


@pytest.mark.parametrize("datafit_class", [Quadratic, SqrtQuadratic])
def test_prox_prox_star(datafit_class):
    n_dim, step, random_state = 10, 0.3, 1236
    rng = np.random.RandomState(random_state)

    y = rng.randn(n_dim)
    rng.seed(random_state)
    w = rng.randn(n_dim)

    quad_datafit = datafit_class()
    # check using Moreau decomposition
    np.testing.assert_allclose(
        step * quad_datafit.prox(w/step, 1/step, y) +
        quad_datafit.prox_conjugate(w, step, y),
        w,
        atol=1e-15
    )


@pytest.mark.parametrize("solver_class, with_dual_init",
                         product([ChambollePock, PDCD, PDCD_WS], [True, False]))
def test_on_Lasso(solver_class, with_dual_init):
    rho = 0.1
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf)
    alpha = rho * alpha_max

    quad_datafit = Quadratic()
    l1_penalty = L1(alpha)
    dual_init = y if with_dual_init else None
    w, _ = solver_class(dual_init=dual_init).solve(X, y, quad_datafit, l1_penalty)

    lasso = Lasso(fit_intercept=False,
                  alpha=alpha / n_samples).fit(X, y)

    np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-6)


@pytest.mark.parametrize("solver_class, with_dual_init",
                         product([ChambollePock, PDCD, PDCD_WS], [True, False]))
def test_on_sqrt_lasso(solver_class, with_dual_init):
    rho = 0.1
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)
    alpha = rho * alpha_max

    quad_datafit = SqrtQuadratic()
    l1_penalty = L1(alpha)
    dual_init = y if with_dual_init else None
    w, _ = solver_class(dual_init=dual_init).solve(X, y, quad_datafit, l1_penalty)

    sqrt_lasso = SqrtLasso(alpha=alpha / np.sqrt(n_samples), tol=1e-6).fit(X, y)

    np.testing.assert_allclose(w, sqrt_lasso.coef_.flatten(), atol=1e-5)


if __name__ == '__main__':
    # rho = 0.1
    # n_samples, n_features = 50, 100
    # X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    # alpha_max = norm(X.T @ y, ord=np.inf)
    # alpha = rho * alpha_max

    # quad_datafit = Quadratic()
    # l1_penalty = L1(alpha)
    # w, _ = PDCD_WS(verbose=1, p0=10).solve(X, y, quad_datafit, l1_penalty)
    pass
