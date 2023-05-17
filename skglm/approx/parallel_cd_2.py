from numba import cuda

import math
import numpy as np
from numpy.linalg import norm

N_THREADS = 32


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
def _parallel_cd_epoch(X, y, w, Xw, grad, lmbd, steps):

    i, j = cuda.grid(2)
    n_samples, n_features = X.shape

    if i >= n_samples or j >= n_features:
        return

    ii, jj = cuda.threadIdx.x, cuda.threadIdx.y

    X_shared = cuda.shared.array((N_THREADS, N_THREADS), dtype=np.float64)
    Xw_shared = cuda.shared.array((N_THREADS,), dtype=np.float64)
    w_shared = cuda.shared.array((N_THREADS,), dtype=np.float64)
    y_shared = cuda.shared.array((N_THREADS,), dtype=np.float64)

    X_shared[ii, jj] = X[i, j]

    # if jj == 0:
    Xw_shared[ii] = Xw[i]
    y_shared[ii] = y[i]

    # if ii == 0:
    w_shared[jj] = w[j]

    cuda.syncthreads()

    # compute grad
    val = X_shared[ii, jj] * grad_datafit(y_shared[ii], Xw_shared[ii])
    cuda.atomic.add(grad, j, val)

    cuda.syncthreads()

    # forward/backward
    # if ii == 0:
    grad_j = grad[j]
    step_j = steps[j]
    old_w_j = w_shared[jj]
    next_w_j = prox_penalty(old_w_j - step_j * grad_j, lmbd * step_j)

    w[j] = next_w_j

    cuda.syncthreads()

    # update variables
    delta_w_j = next_w_j - old_w_j
    if delta_w_j != 0.:
        cuda.atomic.add(Xw, i, delta_w_j * X_shared[ii, jj])


class ParallelCD:

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
        grad_gpu = cuda.to_device(np.zeros(n_features))

        # init grid dim
        block_dim = (N_THREADS, N_THREADS)
        grid_dim = (math.ceil(n_samples / N_THREADS), math.ceil(n_features / N_THREADS))

        for it in range(self.max_iter):

            if self.verbose:
                w_cpu = w_gpu.copy_to_host()
                p_obj = 0.5 * norm(y - X @ w_cpu) ** 2 + lmbd * norm(w_cpu, ord=1)
                print(
                    f"Iteration {it}: p_obj={p_obj:.8f}"
                )

            for _ in range(n_features):
                _parallel_cd_epoch[grid_dim, block_dim](
                    X_gpu, y_gpu, w_gpu, Xw_gpu, grad_gpu, lmbd, steps_gpu)

        return w_gpu.copy_to_host()
