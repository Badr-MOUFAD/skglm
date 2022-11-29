import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.utils.data import make_correlated_data

from sklearn.linear_model import Lasso
from skglm.solvers import AndersonCD
from skglm.prototype_acd.opt_anderson_cd import OptAndersonCD
from skglm.utils.jit_compilation import compiled_clone


def test_on_Lasso():
    rho = 0.1
    n_samples, n_features = 50, 50
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = rho * alpha_max

    # quad_datafit = compiled_clone(Quadratic())
    # l1_penalty = compiled_clone(L1(alpha))
    # w, _, _ = AndersonCD(tol=1e-9, verbose=1,
    #                      fit_intercept=False).solve(X, y, quad_datafit, l1_penalty)

    quad_datafit = Quadratic()
    l1_penalty = L1(alpha)
    w, _ = OptAndersonCD(tol=1e-9, verbose=1).solve(X, y, quad_datafit, l1_penalty)

    lasso = Lasso(fit_intercept=False, alpha=alpha, tol=1e-9).fit(X, y)
    w_sklearn = lasso.coef_.flatten()
    #np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-6)

    print(
        0.5*norm(y - X @ w)**2 / n_samples + alpha * norm(w, ord=1)
    )

    print(
        0.5*norm(y - X @ w_sklearn)**2 / n_samples + alpha * norm(w_sklearn, ord=1)
    )


if __name__ == '__main__':
    test_on_Lasso()
