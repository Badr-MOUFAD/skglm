import pytest

import numpy as np
from prototype_pdcd.datafits import Quadratic, SqrtQuadratic


@pytest.mark.parametrize("datafit_class", [Quadratic, SqrtQuadratic])
def test_prox_prox_star(datafit_class):
    n_dim, step, random_state = 10, 0.3, 1236
    rng = np.random.RandomState(random_state)

    y = rng.randn(n_dim)
    rng.seed(random_state)
    w = rng.randn(n_dim)

    quad_datafit = datafit_class()
    # check using Moreau decomposition
    np.testing.assert_allclose(
        step * quad_datafit.prox(w/step, 1/step, y) +
        quad_datafit.prox_conjugate(w, step, y),
        w,
        atol=1e-15
    )


if __name__ == '__main__':
    pass
