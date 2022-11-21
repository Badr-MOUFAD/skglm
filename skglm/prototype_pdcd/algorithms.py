import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils import compiled_clone


class PDCD_WS:

    def __init__(self, max_iter=1000, verbose=False,
                 p0=100, tol=1e-6, return_p_objs=False):
        self.max_iter = max_iter
        self.p0 = p0
        self.tol = tol
        self.verbose = verbose
        self.return_p_objs = return_p_objs

    def solve(self, X, y, datafit_, penalty_):
        datafit, penalty = PDCD_WS._initialize(datafit_, penalty_)
        n_samples, n_features = X.shape

        # init steps
        dual_step = 1 / norm(X, ord=2)
        primal_steps = 1 / norm(X, axis=0, ord=2)

        # primal vars
        w = np.zeros(n_features)
        Xw = np.zeros(n_samples)

        # dual vars
        z = np.zeros(n_samples)
        z_bar = np.zeros(n_samples)

        # store primal obj
        p_objs = []
        all_features = np.arange(n_features)

        for iter in range(self.max_iter):

            # check convergence
            opts_primal = penalty.subdiff_distance(X.T @ z, w, all_features)
            opt_dual = datafit.subdiff_distance(Xw, z, y)

            stop_crit = max(
                max(opts_primal),
                opt_dual
            )

            if self.verbose:
                current_p_obj = datafit.value(y, Xw) + penalty.value(w)
                print(
                    f"Iteration {iter+1}: {current_p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}")

            if self.return_p_objs:
                current_p_obj = datafit.value(y, Xw) + penalty.value(w)
                p_objs.append(current_p_obj)

            if stop_crit <= self.tol:
                break

            gsupp_size = (w != 0).sum()
            p0 = n_features if iter == 0 else self.p0
            ws_size = max(min(p0, n_features),
                          min(n_features, 2 * gsupp_size))

            # similar to np.argsort()[-ws_size:] but without sorting
            ws = np.argpartition(opts_primal, -ws_size)[-ws_size:]

            # solve sub problem
            PDCD_WS._solve_subproblem(y, X, w, Xw, z, z_bar, datafit, penalty,
                                      primal_steps, dual_step, ws, tol_in=0.3*stop_crit)

        return w, np.asarray(p_objs)

    @staticmethod
    @njit
    def _solve_subproblem(y, X, w, Xw, z, z_bar, datafit, penalty,
                          primal_steps, dual_step, ws, tol_in):
        n_features = X.shape[1]
        ws_size = len(ws)
        past_pseudo_grads = np.zeros(ws_size)

        for epoch in range(1000):

            for idx, j in enumerate(ws):
                # update primal
                old_w_j = w[j]
                past_pseudo_grads[idx] = X[:, j] @ (2 * z_bar - z)
                w[j] = penalty.prox_1D(old_w_j - primal_steps[j] * past_pseudo_grads[idx],
                                       primal_steps[j])

                # keep Xw syncr with X @ w
                if old_w_j != w[j]:
                    Xw += (w[j] - old_w_j) * X[:, j]

                # update dual
                z_bar[:] = datafit.prox_conjugate(z + dual_step * Xw,
                                                  dual_step, y)
                z += (z_bar - z) / n_features

            # check convergence
            if epoch % 10 == 0:
                opts_primal_in = penalty.subdiff_distance(past_pseudo_grads, w, ws)
                opt_dual_in = datafit.subdiff_distance(Xw, z, y)

                stop_crit_in = max(
                    max(opts_primal_in),
                    opt_dual_in
                )

                if stop_crit_in <= tol_in:
                    break

    @staticmethod
    def _initialize(datafit, penalty):
        compiled_datafit = compiled_clone(datafit)
        compiled_penalty = compiled_clone(penalty)

        return compiled_datafit, compiled_penalty


class PDCD:

    def __init__(self, max_iter=1000, verbose=False, tol=1e-6, return_p_objs=False):
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.return_p_objs = return_p_objs

    def solve(self, X, y, datafit_, penalty_):
        datafit, penalty = PDCD._initialize(datafit_, penalty_)

        n_samples, n_features = X.shape
        tol = self.tol

        # init steps
        dual_step = 1 / norm(X, ord=2)
        primal_steps = 1 / norm(X, axis=0, ord=2)

        # primal vars
        w = np.zeros(n_features)
        Xw = np.zeros(n_samples)

        # dual vars
        z = np.zeros(n_samples)
        z_bar = np.zeros(n_samples)

        # store primal obj
        p_objs = []
        all_features = np.arange(n_features)

        for iter in range(self.max_iter):
            # inplace update of w, , Xw, z, and z_bar
            PDCD._one_iter(y, X, w, Xw, z, z_bar, datafit, penalty,
                           primal_steps, dual_step)

            if self.verbose:
                current_p_obj = datafit.value(y, Xw) + penalty.value(w)
                print(f"Iter {iter+1}: {current_p_obj:.10f}")

            if self.return_p_objs:
                current_p_obj = datafit.value(y, Xw) + penalty.value(w)
                p_objs.append(current_p_obj)

            # check convergence
            if iter % 100 == 0:
                opts_primal = penalty.subdiff_distance(X.T @ z, w, all_features)
                opt_dual = datafit.subdiff_distance(Xw, z, y)

                stop_crit = max(
                    max(opts_primal),
                    opt_dual
                )

                if stop_crit <= tol:
                    break

        return w, np.asarray(p_objs)

    @staticmethod
    @njit
    def _one_iter(y, X, w, Xw, z, z_bar, datafit, penalty, primal_steps, dual_step):
        n_features = X.shape[1]

        for j in range(n_features):
            # update primal
            old_w_j = w[j]
            pseudo_grad_j = X[:, j] @ (2 * z_bar - z)
            w[j] = penalty.prox_1D(old_w_j - primal_steps[j] * pseudo_grad_j,
                                   primal_steps[j])

            # keep Xw syncr with X @ w
            if old_w_j != w[j]:
                Xw += (w[j] - old_w_j) * X[:, j]

            # update dual
            z_bar[:] = datafit.prox_conjugate(z + dual_step * Xw,
                                              dual_step, y)
            z += (z_bar - z) / n_features

    @staticmethod
    def _initialize(datafit, penalty):
        compiled_datafit = compiled_clone(datafit)
        compiled_penalty = compiled_clone(penalty)

        return compiled_datafit, compiled_penalty


class ChambollePock:

    def __init__(self, max_iter=1000, verbose=False, tol=1e-6, return_p_objs=False):
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.return_p_objs = return_p_objs

    def solve(self, X, y, datafit_, penalty_):
        datafit, penalty = ChambollePock._initialize(datafit_, penalty_)

        n_samples, n_features = X.shape
        tol = self.tol

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
        all_features = np.arange(n_features)

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

            # check convergence
            if iter % 100 == 0:
                opts_primal = penalty.subdiff_distance(X.T @ z, w, all_features)
                opt_dual = datafit.subdiff_distance(X @ w, z, y)

                stop_crit = max(
                    max(opts_primal),
                    opt_dual
                )

                if stop_crit <= tol:
                    break

        return w, np.asarray(p_objs)

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

    @staticmethod
    def _initialize(datafit, penalty):
        compiled_datafit = compiled_clone(datafit)
        compiled_penalty = compiled_clone(penalty)

        return compiled_datafit, compiled_penalty
