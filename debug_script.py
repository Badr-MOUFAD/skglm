import numpy as np
from numpy.linalg import norm
from skglm.utils import AndersonAcceleration


max_iter, tol = 1000, 1e-9
n_features = 3
np.random.seed(0)
rho = np.random.rand(n_features)
w_star = 1 / (1 - rho)
X = np.eye(n_features)


# with acceleration
acc = AndersonAcceleration(K=n_features+1)
w = np.ones(n_features)
Xw = X @ w
for i in range(max_iter):
    w, Xw, log = acc.extrapolate(w, Xw)
    print(f"Iter {i}: {log}")
    w = rho * w + 1
    Xw = X @ w

    if norm(w - w_star, ord=np.inf) < tol:
        break

print("w - w_star:", norm(w - w_star, ord=np.inf))


print("===========")


# with acceleration
w = np.ones(n_features)
for i in range(max_iter):
    w = rho * w + 1

    if norm(w - w_star, ord=np.inf) < tol:
        break

print(f"total iter: {i}")
print("w - w_star:", norm(w - w_star, ord=np.inf))
