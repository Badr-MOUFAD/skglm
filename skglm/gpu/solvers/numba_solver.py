import math
import numpy as np
from numba import cuda


from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


# max number of threads that one block could contain
# https://forums.developer.nvidia.com/t/maximum-number-of-threads-on-thread-block/46392
N_THREADS = 1024


class NumbaSolver:

    def __init__(self, max_iter=1000, verbose=0):
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, X, y, lmbd):
        n_samples, n_features = X.shape

        # compute step
        lipschitz = np.linalg.norm(X, ord=2) ** 2
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # number of block to use along each axis when calling kernel
        n_blocks_axis_0, n_blocks_axis_1 = (math.ceil(n / N_THREADS) for n in X.shape)

        # transfer to device
        X_gpu = cuda.to_device(X)
        y_gpu = cuda.to_device(y)

        # init vars on device
        # CAUTION: should be init with specific values
        # otherwise, stale values in GPU memory are used
        w = cuda.to_device(np.zeros(n_features))
        mid_w = cuda.to_device(np.zeros(n_features))
        old_w = cuda.to_device(np.zeros(n_features))

        # needn't to be init with values as theses are
        # they store results of computation
        grad = cuda.device_array(n_features)
        minus_residual = cuda.device_array(n_samples)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # inplace update of minus_residual
            # minus_residual = X_gpu @ mid_w - y_gpu
            _compute_minus_residual[n_blocks_axis_0, N_THREADS](
                X_gpu, y_gpu, mid_w, minus_residual)

            # inplace update of grad
            # grad = X_gpu @ minus_residual
            _compute_grad[n_blocks_axis_1, N_THREADS](
                X_gpu, minus_residual, grad)

            # inplace update of w
            # w = ST(mid_w - step * grad, step * lmbd)
            _forward_backward[n_blocks_axis_1, N_THREADS](
                mid_w, grad, step, lmbd, w)

            if self.verbose:
                w_cpu = w.copy_to_host()

                p_obj = compute_obj(X, y, lmbd, w_cpu)
                opt_crit = eval_opt_crit(X, y, lmbd, w_cpu)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            coef = (t_old - 1) / t_new
            # mid_w = w + coef * (w - old_w)
            _extrapolate[n_blocks_axis_1, N_THREADS](
                w, old_w, coef, mid_w)

            # update FISTA vars
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            # in `copy_to_device`: `self` is destination and `other` is source
            old_w.copy_to_device(w)

        # transfer back to host
        w_cpu = w.copy_to_host()

        return w_cpu


@cuda.jit
def _compute_minus_residual(X_gpu, y_gpu, mid_w, out):
    # compute: out = X_gpu @ w - y_gpu
    i = cuda.grid(1)

    n_samples, n_features = X_gpu.shape
    if i >= n_samples:
        return

    tmp = 0.
    for j in range(n_features):
        tmp += X_gpu[i, j] * mid_w[j]
    tmp -= y_gpu[i]

    out[i] = tmp


@cuda.jit
def _compute_grad(X_gpu, minus_residual, out):
    # compute: out = X.T @ minus_residual
    j = cuda.grid(1)

    n_samples, n_features = X_gpu.shape
    if j >= n_features:
        return

    tmp = 0.
    for i in range(n_samples):
        tmp += X_gpu[i, j] * minus_residual[i]

    out[j] = tmp


@cuda.jit
def _forward_backward(mid_w, grad, step, lmbd, out):
    # forward: mid_w = mid_w - step * grad
    # backward: w = ST_vec(mid_w, step * lmbd)
    j = cuda.grid(1)

    n_features = len(mid_w)
    if j >= n_features:
        return

    # forward
    tmp = mid_w[j] - step * grad[j]

    # backward
    level = step * lmbd
    if abs(tmp) <= level:
        tmp = 0.
    elif tmp > level:
        tmp = tmp - level
    else:
        tmp = tmp + level

    out[j] = tmp


@cuda.jit
def _extrapolate(w, old_w, coef, out):
    # compute: out = w + coef * (w - old_w)
    j = cuda.grid(1)

    n_features = len(w)
    if j >= n_features:
        return

    out[j] = w[j] + coef * (w[j] - old_w[j])
