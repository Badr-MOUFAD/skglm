from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.datafits import QuadraticGroup
    from skglm.penalties import WeightedGroupL2
    from skglm.utils import grp_converter


class Objective(BaseObjective):
    name = "Group Lasso objective"

    parameters = {
        'rho': [1., 1e-1, 1e-2, 1e-3],
        'groups': [10]
    }

    def __init__(self, rho, groups):
        self.rho = rho
        self.groups = groups

    def set_data(self, X, y):
        n_samples, n_features = X.shape
        groups, rho = self.groups, self.rho

        grp_indices, grp_ptr = grp_converter(groups, n_features)
        n_groups = len(grp_ptr) - 1
        np.random.seed(0)
        weights = abs(np.random.randn(n_groups))

        alpha_max = 0.
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            alpha_max = max(
                alpha_max,
                norm(X[:, grp_g_indices].T @ y) / n_samples / weights[g]
            )
        alpha = rho * alpha_max

        self.alpha = alpha
        self.grp_indices, self.grp_ptr = grp_indices, grp_ptr
        self.X, self.y = X, y
        self.weights = weights

        self.datafit = QuadraticGroup(grp_ptr, grp_indices)
        self.penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)

    def compute(self, beta):
        X, y = self.X, self.y
        return self.datafit.value(y, beta, X @ beta) + self.penalty.value(beta)

    def to_dict(self):
        return dict(X=self.X, y=self.y, alpha=self.alpha, groups=self.groups,
                    weights=self.weights, grp_indices=self.grp_indices,
                    grp_ptr=self.grp_ptr)
