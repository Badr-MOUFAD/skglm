from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from celer import GroupLasso


class Solver(BaseSolver):
    name = 'celer'

    requirements = [
        'pip: git+https://github.com/Badr-MOUFAD/celer.git@expose-acc-params'
    ]

    parameters = {
        'use_acc': [True, False],
        'K': [6]
    }

    def __init__(self, use_acc, K):
        self.use_acc = use_acc

    def set_objective(self, X, y, alpha, groups, weights, grp_indices, grp_ptr):
        self.X, self.y = X, y

        self.model = GroupLasso(groups, alpha, max_iter=1, max_epochs=100,
                                weights=weights, tol=1e-12, fit_intercept=False,
                                use_acc=self.use_acc)

    def run(self, n_iter):
        if n_iter == 0:
            self.w = np.zeros(self.X.shape[1])
            return

        X, y = self.X, self.y

        self.model.max_iter = n_iter
        self.model.fit(X, y)

        self.w = self.model.coef_

    def get_result(self):
        return self.w
