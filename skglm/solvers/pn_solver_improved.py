import numpy as np
from scipy.sparse import issparse
from numba import njit


def pn_solver_improved(X, y, datafit, penalty, w_init=None, p0=10,
                       max_iter=20, max_epochs=1000, tol=1e-4, verbose=0):
    """Run a Prox Newton solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    datafit : instance of BaseDatafit
        Datafit object.

    penalty : instance of BasePenalty
        Penalty object.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    p0 : int, default 10
        Minimum number of groups to be included in the working set.

    max_iter : int, default 20
        Maximum number of iterations.

    max_epochs : int, default 1000
        Maximum number of epochs.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.

    objs_out: array (max_iter,)
        The objective values at every outer iteration.

    stop_crit: float
        The value of the stop criterion.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features) if w_init is None else w_init
    Xw = np.zeros(n_samples) if w_init is None else X @ w_init
    all_features = np.arange(n_features)
    stop_crit = 0.
    p_objs_out = []

    is_sparse = issparse(X)

    for t in range(max_iter):
        # compute scores
        if is_sparse:
            grad = construct_grad_sparse(X.data, X.indptr, X.indices,
                                         y, w, Xw, datafit, all_features)
        else:
            grad = construct_grad(X, y, w, Xw, datafit, all_features)

        opt = penalty.subdiff_distance(w, grad, all_features)

        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print('Outer loop early exit')
            break

        # build working set
        gsupp_size = penalty.generalized_support(w).sum()
        ws_size = max(min(p0, n_features),
                      min(n_features, 2 * gsupp_size))
        # similar to np.argsort()[-ws_size:] but without sorting
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        grad_ws = grad[ws]
        tol_in = 0.3 * stop_crit

        for epoch in range(max_epochs):

            # find descent direction
            if is_sparse:
                (
                    delta_w_ws,
                    X_delta_w_ws
                ) = _compute_descent_direction_s(X.data, X.indptr, X.indices,
                                                 y, w, Xw, grad_ws, datafit,
                                                 penalty, ws, max_cd_iter=20,
                                                 tol=0.3*tol_in)
            else:
                (
                    delta_w_ws,
                    X_delta_w_ws
                ) = _compute_descent_direction(X, y, w, Xw, grad_ws, datafit, penalty,
                                               ws, max_cd_iter=20, tol=0.3*tol_in)

            # backtracking line search with inplace update of w, Xw
            if is_sparse:
                grad_ws[:] = _backtrack_line_search_s(X.data, X.indptr, X.indices,
                                                      y, w, Xw, datafit,
                                                      penalty, delta_w_ws,
                                                      X_delta_w_ws, ws,
                                                      max_backtrack_iter=20)
            else:
                grad_ws[:] = _backtrack_line_search(X, y, w, Xw, datafit, penalty,
                                                    delta_w_ws, X_delta_w_ws, ws,
                                                    max_backtrack_iter=20)

            # check convergence
            opt_in = penalty.subdiff_distance(w, grad_ws, ws)
            stop_crit_in = np.max(opt_in)

            if max(verbose-1, 0):
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"|—— Epoch {epoch+1}: {p_obj:.10f}, "
                    f"stopping crit in: {stop_crit_in:.2e}"
                )

            if stop_crit_in <= tol_in:
                if max(verbose-1, 0):
                    print("|—— Inner loop early exit")
                break

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, p_objs_out, stop_crit


@njit
def _compute_descent_direction(X, y, w_epoch, Xw_epoch, grad_ws, datafit, penalty,
                               ws, max_cd_iter, tol):
    # Given:
    #   - b = \nabla   F(X w_epoch)
    #   - D = \nabla^2 F(X w_epoch)   <------>  raw_hess
    # Minimize for delta_w = w - w_epoch:
    #  b.T @ X @ delta_w + \
    #  1/2 * delta_w.T @ (X.T @ D @ X) @ delta_w + penalty(w)
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        lipschitz[idx] = raw_hess @ X[:, j] ** 2

    cached_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(X.shape[0])
    w_ws = w_epoch[ws]

    for cd_iter in range(max_cd_iter):
        for idx, j in enumerate(ws):

            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            cached_grads[idx] = grad_ws[idx] + X[:, j] @ (raw_hess * X_delta_w_ws)
            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx],
                stepsize, j
            )

            if w_ws[idx] != old_w_idx:
                X_delta_w_ws += (w_ws[idx] - old_w_idx) * X[:, j]

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws], X_delta_w_ws


# sparse version of func above
@njit
def _compute_descent_direction_s(X_data, X_indptr, X_indices, y,
                                 w_epoch, Xw_epoch, grad_ws, datafit, penalty,
                                 ws, max_cd_iter, tol):
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        # equivalent to: lipschitz[idx] += raw_hess * X[:, j] ** 2
        lipschitz[idx] = sparse_squared_weighted_norm(X_data, X_indptr, X_indices,
                                                      j, raw_hess)

    cached_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(len(y))
    w_ws = w_epoch[ws]

    for cd_iter in range(max_cd_iter):
        for idx, j in enumerate(ws):

            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            cached_grads[idx] = grad_ws[idx]
            # equivalent to cached_grads[idx] += X[:, j] @ (raw_hess * X_delta_w_ws)
            cached_grads[idx] += sparse_weighted_dot(X_data, X_indptr, X_indices, j,
                                                     X_delta_w_ws, raw_hess)

            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx],
                stepsize, j
            )

            if w_ws[idx] != old_w_idx:
                update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w_ws,
                                 w_ws[idx] - old_w_idx, j)

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws], X_delta_w_ws


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w_ws, X_delta_w_ws,
                           ws, max_backtrack_iter):
    # inplace update of w and Xw
    # return grad_ws of the last w and Xw
    step, prev_step = 1., 0.
    prev_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(max_backtrack_iter):
        stop_crit = -prev_penalty_val
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = construct_grad(X, y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws


# sparse version of func above
@njit
def _backtrack_line_search_s(X_data, X_indptr, X_indices,
                             y, w, Xw, datafit, penalty, delta_w_ws,
                             X_delta_w_ws, ws, max_backtrack_iter):
    # inplace update of w and Xw
    step, prev_step = 1., 0.
    prev_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(max_backtrack_iter):
        stop_crit = -prev_penalty_val
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = construct_grad_sparse(X_data, X_indptr, X_indices,
                                        y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws.T @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws


@njit
def construct_grad(X, y, w, Xw, datafit, ws):
    """Compute grad of datafit restricted to ``ws``."""
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        grad[idx] = X[:, j] @ raw_grad
    return grad


@njit
def construct_grad_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, ws):
    """Compute grad of datafit restricted to ``ws`` in case ``X`` sparse."""
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        grad[idx] = sparse_xj_dot(X_data, X_indptr, X_indices, j, raw_grad)
    return grad


@njit(fastmath=True)
def sparse_xj_dot(data, indptr, indices, j, other):
    """Compute ``X[:, j] @ other`` in case ``X`` sparse."""
    res = 0.
    for i in range(indptr[j], indptr[j+1]):
        res += data[i] * other[indices[i]]
    return res


@njit(fastmath=True)
def sparse_weighted_dot(data, indptr, indices, j, other, weights):
    """Computes ``X[:, j] @ (weights * other)`` in case ``X`` sparse."""
    res = 0.
    for i in range(indptr[j], indptr[j+1]):
        res += data[i] * other[indices[i]] * weights[indices[i]]
    return res


@njit(fastmath=True)
def sparse_squared_weighted_norm(data, indptr, indices, j, weights):
    """Compute ``weights @ X[:, j]**2`` in case ``X`` sparse."""
    res = 0.
    for i in range(indptr[j], indptr[j+1]):
        res += weights[indices[i]] * data[i]**2
    return res


@njit(fastmath=True)
def update_X_delta_w(data, indptr, indices, X_delta_w, diff, j):
    """Compute ``X_delta_w += diff * X[:, j]`` case of ``X`` sparse."""
    for i in range(indptr[j], indptr[j+1]):
        X_delta_w[indices[i]] += diff * data[i]
