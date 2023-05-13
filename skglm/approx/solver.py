import numpy as np

from skglm.penalties import L1
from skglm.datafits import Quadratic

from skglm.utils.jit_compilation import compiled_clone


class Approx:

    def __init__(self, max_iter=100, random_state=1235, verbose=False):
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def solve(self, X, y, datafit: Quadratic, penalty: L1):
        datafit, penalty = self._validate_init(datafit, penalty, X, y)
        w = Approx._run_approx(X, y, datafit, penalty, self.max_iter)
        return w

    @staticmethod
    def _run_approx(X, y, datafit: Quadratic, penalty: L1, max_iter, verbose=True):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(125)

        w = np.zeros(n_features)
        z = np.zeros(n_features)
        theta = np.zeros(n_features)

        acc_coef = 1 / n_features

        for it in range(max_iter):

            if verbose:
                p_obj = datafit.value(y, w, X @ w) + penalty.value(w)

                print(
                    f"Iteration {it}: p_obj_in={p_obj:.8f} "
                )

            for j in rng.choice(n_features, size=n_features):

                if datafit.lipschitz[j] == 0.:
                    continue

                step = 1 / (datafit.lipschitz[j] * acc_coef * n_features)

                theta = (1 - acc_coef) * w + acc_coef * z
                old_z = z.copy()

                grad_j = datafit.gradient_scalar(X, y, theta, X @ theta, j)
                z[j] = penalty.prox_1d(old_z[j] - step * grad_j, step, j)

                w = theta + (n_features * acc_coef) * (z - old_z)

                acc_coef = (np.sqrt(acc_coef ** 4 + 4 *
                            acc_coef ** 2) - acc_coef ** 2) / 2

        return w

    def _validate_init(self, datafit: Quadratic, penalty, X, y):
        # TODO: checks
        datafit_, penalty_ = compiled_clone(datafit), compiled_clone(penalty)

        # init
        datafit_.initialize(X, y)

        return datafit_, penalty_
