import numpy as np
from numpy.linalg import norm


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
        self.return_p_obj = return_p_objs

    def solve(self, X, y, datafit, penalty):
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
            # dual update
            z = datafit.prox_conjugate(z + dual_step * X @ w_bar,
                                       dual_step, y)

            # primal update
            old_w = w.copy()
            w = penalty.prox(old_w - primal_step * X.T @ z,
                             primal_step, y)
            w_bar = 2 * w - old_w

            if self.verbose:
                current_p_obj = datafit.value(y, X, X @ w) + penalty.value(w)
                print(f"Iter {iter+1}: {current_p_obj:.10f}")

            if self.return_p_objs:
                current_p_obj = datafit.value(y, X, X @ w) + penalty.value(w)
                p_objs.append(current_p_obj)

        return w, np.asarray(p_objs)
