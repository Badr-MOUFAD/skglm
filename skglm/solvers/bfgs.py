import numpy as np
import scipy.optimize
from numpy.linalg import norm

from skglm.solvers import BaseSolver
from skglm.datafits import BaseDatafit


class BFGS(BaseSolver):
    """A wrapper for scipy BFGS solver."""

    def __init__(self, max_iter=50, tol=1e-4, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, X, y, datafit: BaseDatafit, penalty, w_init=None, Xw_init=None):

        def objective_function(w):
            Xw = X @ w
            datafit_value = datafit.value(y, w, Xw)
            penalty_value = penalty.value(w)

            return datafit_value + penalty_value

        def jacobian_function(w):
            Xw = X @ w
            datafit_grad = datafit.gradient(y, w, Xw)
            penalty_grad = penalty.gradient(w)

            return datafit_grad + penalty_grad

        n_features = X.shape[1]
        w = np.zeros(n_features) if w_init is not None else w_init
        p_objs_out = []

        result = scipy.optimize.minimize(
            fun=objective_function,
            jac=jacobian_function,
            x0=w,
            method="BFGS",
            options=dict(
                gtol=self.tol,
                maxiter=self.max_iter,
                disp=self.verbose
            ),
            callback=lambda w_k: p_objs_out.append(objective_function(w_k))
        )

        w = result.x
        stop_crit = norm(result.jac)

        return w, np.asarray(p_objs_out), stop_crit