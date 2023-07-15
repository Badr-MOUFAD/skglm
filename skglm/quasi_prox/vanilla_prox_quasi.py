import numpy as np
from numba import njit

from skglm.solvers import BaseSolver

EPS_TOL = 0.3
MAX_CD_ITER = 20
MAX_BACKTRACK_ITER = 20


class QuasiProxNewton(BaseSolver):

    def __init__(self, max_iter=20, tol=1e-4, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, datafit, penalty, X, y, w_init=None, Xw_init=None):
        dtype = X.dtype
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype)
        Xw = np.zeros(n_samples, dtype)
        B = np.diagonal(np.full_like(w, fill_value=1 / n_samples))

        all_features = np.arange(n_features)
        grad = _construct_grad(X, y, w, Xw, datafit, all_features)
        old_grad = grad.copy()

        for it in range(self.max_iter):

            # check convergence
            opt = penalty.subdiff_distance(w, grad, all_features)
            stop_crit = np.max(opt)

            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {it+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            tol_in = EPS_TOL * stop_crit

            # determine descent direction
            delta_w_ws, X_delta_w_ws = _descent_direction(
                X, y, w, Xw, grad, B, datafit,
                penalty, all_features, tol=EPS_TOL*tol_in)

            # backtracking line search
            grad = _backtrack_line_search(
                X, y, w, Xw, datafit, penalty,
                delta_w_ws, X_delta_w_ws)

            # update Hessian approximation
            delta_grad = grad - old_grad
            B = _SR1_update(delta_w_ws, delta_grad, B)
            old_grad = grad

        return w


@njit
def _descent_direction(X, y, w_epoch, Xw_epoch, grad, B,
                       datafit, penalty, ws, tol):
    dtype = X.dtype
    n_samples, n_features = X.shape

    past_grads = np.zeros(n_features, dtype)
    w = w_epoch.copy()
    delta_w = np.zeros(n_features, dtype)
    X_delta_w = np.zeros(n_samples, dtype)

    for cd_iter in range(MAX_CD_ITER):
        for j in ws:
            if B[j, j] == 0:
                continue

            stepsize = 1 / B[j, j]
            old_w_j = w[j]
            past_grads[j] = grad[j] + B[j] @ delta_w

            w[j] = penalty.prox_1d(old_w_j - stepsize * past_grads[j], stepsize, j)

            if w[j] != old_w_j:
                delta_w[j] += w[j] - old_w_j
                X_delta_w += delta_w[j] * X[:, j]

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w, past_grads, ws)
            stop_crit = np.max(opt)

            if stop_crit <= tol:
                break

    return delta_w, X_delta_w


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w, X_delta_w):
    step, prev_step = 1., 0.
    n_features = X.shape[1]
    all_features = np.arange(n_features)
    old_penalty_val = penalty.value(w)

    for _ in range(MAX_BACKTRACK_ITER):
        w += (step - prev_step) * delta_w
        Xw += (step - prev_step) * X_delta_w

        grad = _construct_grad(X, y, w, Xw, datafit, all_features)
        stop_crit = penalty.value(w) - old_penalty_val

        dot = grad @ delta_w
        stop_crit += step * dot

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws), dtype=X.dtype)
    for idx, j in enumerate(ws):
        grad[idx] = X[:, j] @ raw_grad
    return grad


def _SR1_update(delta_w, delta_grad, B):
    Bs_minus_y = B @ delta_w - delta_grad

    gamma = Bs_minus_y @ delta_w
    if gamma == 0.:
        return B

    return B + np.outer(Bs_minus_y / gamma, Bs_minus_y)
