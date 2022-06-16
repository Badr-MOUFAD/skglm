from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.datafits.group import QuadraticGroup
    from skglm.penalties.block_separable import WeightedGroupL2
    from skglm.solvers.group_bcd_solver import bcd_solver


class Solver(BaseSolver):
    name = 'skglm'

    parameters = {
        'use_acc': [True, False],
        'K': [6]
    }

    def __init__(self, use_acc, K):
        self.use_acc = use_acc
        self.K = K

    def set_objective(self, X, y, alpha, groups, weights, grp_indices, grp_ptr):
        self.X, self.y = X, y
        self.alpha, self.weights = alpha, weights
        self.grp_indices, self.grp_ptr = grp_indices, grp_ptr

        self.datafit = QuadraticGroup(grp_ptr, grp_indices)
        self.penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)

        self.run(n_iter=10)  # cache numba compilation

    def run(self, n_iter):
        use_acc = self.use_acc
        K = self.K
        X, y = self.X, self.y
        datafit, penalty = self.datafit, self.penalty

        self.w = bcd_solver(X, y, datafit, penalty, K=K, use_acc=use_acc,
                            max_iter=n_iter, tol=1e-12, p0=10)[0]

    def get_result(self):
        return self.w
