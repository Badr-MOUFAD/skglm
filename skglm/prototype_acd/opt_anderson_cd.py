import numpy as np
from numba import njit

from skglm.utils.jit_compilation import compiled_clone
from skglm.solvers.common import construct_grad


class OptAndersonCD:

    def __init__(self, max_iter=1000, max_epochs=10_000, tol=1e-4, p0=10, verbose=0):
        self.tol = tol
        self.p0 = p0
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_epochs = max_epochs

    def solve(self, X, y, datafit_, penalty_):
        # init
        datafit, penalty = OptAndersonCD._initialize(X, y, datafit_, penalty_)

        n_samples, n_features = X.shape
        all_features = np.arange(n_features)
        p_objs_out = []

        # vars
        w = np.zeros(n_features)
        Xw = np.zeros(n_samples)

        for iteration in range(self.max_iter):
            # compute scores
            grad = construct_grad(X, y, w, Xw, datafit, all_features)
            opt = penalty.subdiff_distance(w, grad, all_features)

            stop_crit = np.max(opt)

            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iteration+1}: {p_obj:.10f}, "
                    "stopping crit: {stop_crit:.2e}"
                )

            # check converge
            if stop_crit <= self.tol:
                break

            # build working sets
            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = max(min(n_features, self.p0),
                          min(n_features, 2 * gsupp_size))

            ws = np.argpartition(opt, -ws_size)[-ws_size:]

            # solve sub problem
            OptAndersonCD._solve_subproblem(X, y, w, Xw, datafit, penalty,
                                            self.max_epochs, ws, tol_in=0.3*stop_crit)

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        p_objs_out.append(p_obj)

        return w, np.asarray(p_objs_out)

    @staticmethod
    @njit
    def _solve_subproblem(X, y, w, Xw, datafit, penalty, max_epochs, ws, tol_in):
        lipschitz = datafit.lipschitz

        for epoch in range(max_epochs):

            # cd epoch
            for idx, j in enumerate(ws):

                # skip when X[:, j] == 0
                if lipschitz[idx] == 0:
                    continue

                old_w_j = w[j]
                stepsize = 1 / lipschitz[idx]
                w[j] = penalty.prox_1d(
                    old_w_j - stepsize * datafit.gradient_scalar(X, y, w, Xw, j),
                    stepsize, j)

                # keep Xw synchr with X @ w
                if w[j] != old_w_j:
                    Xw += (w[j] - old_w_j) * X[:, j]

            # TODO: AA acceleration

            # check convergence
            if epoch % 10 == 0:
                # compute scores
                grad_ws = construct_grad(X, y, w, Xw, datafit, ws)
                opt_ws = penalty.subdiff_distance(w, grad_ws, ws)

                stop_crit_in = np.max(opt_ws)

                if stop_crit_in <= tol_in:
                    break

    @staticmethod
    def _initialize(X, y, datafit_, penalty_):
        # jit compile classes
        datafit = compiled_clone(datafit_)
        penalty = compiled_clone(penalty_)

        # init lipschitz constants
        datafit.initialize(X, y)

        return datafit, penalty
