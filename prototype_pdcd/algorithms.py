import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils import compiled_clone


class PDCD:

    def __init__(self):
        pass

    def solve(self):
        return

    @staticmethod
    def _compute_objective():
        return


class ChambollePock:

    def __init__(self, max_iter=1000, verbose=False, return_p_objs=False):
        self.max_iter = max_iter
        self.verbose = verbose
        self.return_p_objs = return_p_objs

    def solve(self, X, y, datafit_, penalty_):
        datafit, penalty = ChambollePock._initialize(datafit_, penalty_)
        n_samples, n_features = X.shape

        # init steps
        L = norm(X, ord=2)
        dual_step = 0.9 / L
        primal_step = 0.9 / L

        # primal vars
        w = np.zeros(n_features)
        w_bar = np.zeros(n_features)

        # dual vars
        z = np.zeros(n_samples)

        # store primal obj
        p_objs = []

        for iter in range(self.max_iter):
            # inplace update of w, w_bar, and z
            ChambollePock._one_iter(y, X, w, w_bar, z, datafit, penalty,
                                    primal_step, dual_step)

            if self.verbose:
                current_p_obj = datafit.value(y, X @ w) + penalty.value(w)
                print(f"Iter {iter+1}: {current_p_obj:.10f}")

            if self.return_p_objs:
                current_p_obj = datafit.value(y, X @ w) + penalty.value(w)
                p_objs.append(current_p_obj)

        return w, np.asarray(p_objs)

    @staticmethod
    def _initialize(datafit, penalty):
        compiled_datafit = compiled_clone(datafit)
        compiled_penalty = compiled_clone(penalty)

        return compiled_datafit, compiled_penalty

    @staticmethod
    @njit
    def _one_iter(y, X, w, w_bar, z, datafit, penalty, primal_step, dual_step):
        # dual update
        z[:] = datafit.prox_conjugate(z + dual_step * X @ w_bar,
                                      dual_step, y)

        # primal update
        old_w = w.copy()
        w[:] = penalty.prox(old_w - primal_step * X.T @ z,
                            primal_step)
        w_bar[:] = 2 * w - old_w
