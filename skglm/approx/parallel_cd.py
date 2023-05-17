from numba import cuda

import math
import numpy as np
from numpy.linalg import norm


@cuda.jit(device=True)
def prox_penalty(value, step):
    if value > step:
        return value - step
    elif value < -step:
        return value + step
    else:
        return 0.


@cuda.jit(device=True)
def grad_datafit(y_i, Xw_i):
    return (Xw_i - y_i)


@cuda.jit
def _parallel_cd_epoch(X, y, w, Xw, lmbd, steps):
    j = cuda.grid(1)
    n_samples, n_features = X.shape

    if j >= n_features:
        return

    # compute grad
    grad_j = 0
    for i in range(n_samples):
        grad_j += X[i, j] * grad_datafit(y[i], Xw[i])

    # forward/backward
    step_j = steps[j]
    old_w_j = w[j]
    next_w_j = prox_penalty(old_w_j - step_j * grad_j, lmbd * step_j)

    # update variables
    delta_w_j = next_w_j - old_w_j
    w[j] = next_w_j
    for i in range(n_samples):
        cuda.atomic.add(Xw, i, delta_w_j * X[i, j])


class ParallelCD:

    N_THREADS = 1024

    def __init__(self, max_iter, verbose=False) -> None:
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, X, y, lmbd):
        n_samples, n_features = X.shape
        steps = 1 / (norm(X, axis=0, ord=2) ** 2 * n_features)

        # transfer data to device
        X_gpu = cuda.to_device(X)
        y_gpu = cuda.to_device(y)
        steps_gpu = cuda.to_device(steps)

        # init vars on GPU
        w_gpu = cuda.to_device(np.zeros(n_features))
        Xw_gpu = cuda.to_device(np.zeros(n_samples))

        # init grid dim
        block_dim = (ParallelCD.N_THREADS,)
        grid_dim = (math.ceil(n_features / ParallelCD.N_THREADS),)

        for it in range(self.max_iter):

            if self.verbose:
                w_cpu = w_gpu.copy_to_host()
                p_obj = 0.5 * norm(y - X @ w_cpu) ** 2 + lmbd * norm(w_cpu, ord=1)
                print(
                    f"Iteration {it}: p_obj={p_obj:.8f}"
                )

            for _ in range(n_features):
                _parallel_cd_epoch[grid_dim, block_dim](
                    X_gpu, y_gpu, w_gpu, Xw_gpu, lmbd, steps_gpu)

        return w_gpu.copy_to_host()
