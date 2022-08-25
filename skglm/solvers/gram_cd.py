import numpy as np
from numba import njit
from scipy.sparse import issparse
from skglm.utils import AndersonAcceleration


def gram_cd_solver(X, y, penalty, max_iter=100, w_init=None,
                   use_acc=True, greedy_cd=True, tol=1e-4, verbose=False):
    """Run coordinate descent while keeping the gradients up-to-date with Gram updates.

    Minimize::
        1 / (2*n_samples) * norm(y - Xw)**2 + penalty(w)

    Which can be rewritten as::
        w.T @ Q @ w / (2*n_samples) - q.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X (gram matrix), and q = X.T @ y

    Parameters
    ----------
    X : array or sparse CSC matrix, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    penalty : instance of BasePenalty
        Penalty object.

    max_iter : int, default 100
        Maximum number of iterations.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    use_acc : bool, default True
        Extrapolate the iterates based on the past 5 iterates if set to True.

    greedy_cd : bool, default True
        Use a greedy strategy to select features to update in Gram CD epoch
        if set to True. A cyclic strategy is used otherwise.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.

    objs_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        The value of the stopping criterion when the solver stops.
    """
    n_samples, n_features = X.shape
    scaled_gram = X.T @ X / n_samples
    scaled_Xty = X.T @ y / n_samples
    scaled_y_norm2 = np.linalg.norm(y)**2 / (2*n_samples)

    if issparse(X):
        scaled_gram = scaled_gram.toarray()

    all_features = np.arange(n_features)
    stop_crit = np.inf  # prevent ref before assign
    p_objs_out = []

    w = np.zeros(n_features) if w_init is None else w_init
    scaled_gram_w = np.zeros(n_features) if w_init is None else scaled_gram @ w_init
    grad = scaled_gram_w - scaled_Xty
    opt = penalty.subdiff_distance(w, grad, all_features)

    if use_acc:
        accelerator = AndersonAcceleration(K=5)
        w_acc = np.zeros(n_features)
        scaled_gram_w_acc = np.zeros(n_features)

    for t in range(max_iter):
        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = (0.5 * w @ scaled_gram_w - scaled_Xty @ w +
                     scaled_y_norm2 + penalty.value(w))
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # inplace update of w, XtXw
        opt = _gram_cd_epoch(scaled_gram, scaled_Xty, w, scaled_gram_w,
                             penalty, greedy_cd)

        # perform Anderson extrapolation
        if use_acc:
            w_acc, scaled_gram_w_acc, is_extrapolated = accelerator.extrapolate(
                w, scaled_gram_w)

            if is_extrapolated:
                p_obj_acc = (0.5 * w_acc @ scaled_gram_w_acc - scaled_Xty @ w_acc +
                             penalty.value(w_acc))
                p_obj = 0.5 * w @ scaled_gram_w - scaled_Xty @ w + penalty.value(w)
                if p_obj_acc < p_obj:
                    w[:] = w_acc
                    scaled_gram_w[:] = scaled_gram_w_acc

        # store p_obj
        p_obj = 0.5 * w @ scaled_gram_w - scaled_Xty @ w + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, np.array(p_objs_out), stop_crit


@njit
def _gram_cd_epoch(scaled_gram, scaled_Xty, w, scaled_gram_w, penalty, greedy_cd):
    all_features = np.arange(len(w))
    for j in all_features:
        # compute grad
        grad = scaled_gram_w - scaled_Xty

        # select feature j
        if greedy_cd:
            opt = penalty.subdiff_distance(w, grad, all_features)
            chosen_j = np.argmax(opt)
        else:  # cyclic
            chosen_j = j

        # update w_j
        old_w_j = w[chosen_j]
        step = 1 / scaled_gram[chosen_j, chosen_j]  # 1 / lipchitz_j
        w[chosen_j] = penalty.prox_1d(old_w_j - step * grad[chosen_j], step, chosen_j)

        # Gram matrix update
        if w[chosen_j] != old_w_j:
            scaled_gram_w += (w[chosen_j] - old_w_j) * scaled_gram[:, chosen_j]

    # opt
    grad = scaled_gram_w - scaled_Xty
    return penalty.subdiff_distance(w, grad, all_features)