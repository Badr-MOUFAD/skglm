import pytest

import numpy as np
from numpy.linalg import norm

from prototype_pdcd.penalties import L1
from prototype_pdcd.algorithms import ChambollePock, PDCD
from prototype_pdcd.datafits import Quadratic, SqrtQuadratic

from sklearn.linear_model import Lasso
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


@pytest.mark.parametrize("solver_class", [ChambollePock, PDCD])
def test_on_Lasso(solver_class):
    rho = 0.1
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf)
    alpha = rho * alpha_max

    quad_datafit = Quadratic()
    l1_penalty = L1(alpha)
    w, _ = solver_class().solve(X, y, quad_datafit, l1_penalty)

    lasso = Lasso(fit_intercept=False,
                  alpha=alpha / n_samples).fit(X, y)

    np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-6)


if __name__ == '__main__':
    pass
