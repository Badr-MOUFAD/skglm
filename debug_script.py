import numpy as np
from numpy.linalg import norm
from skglm.utils import AndersonAcceleration


max_iter, tol = 1000, 1e-9
n_features = 30
np.random.seed(0)
rho = np.random.rand(n_features)
b = 0
w_star = b / (1 - rho)
X = np.eye(n_features)


# with acceleration
acc = AndersonAcceleration(K=n_features+1)
w = np.ones(n_features)
Xw = X @ w
for i in range(max_iter):
    w, Xw, log = acc.extrapolate(w, Xw)
    print(f"Iter {i}: {log}")
    w = rho * w + b
    Xw = X @ w

    if norm(w - w_star, ord=np.inf) < tol:
        break

print("w - w_star:", norm(w - w_star, ord=np.inf))


print("===========")


# with acceleration
w = np.ones(n_features)
for i in range(max_iter):
    w = rho * w + b

    if norm(w - w_star, ord=np.inf) < tol:
        break

print(f"total iter: {i}")
print("w - w_star:", norm(w - w_star, ord=np.inf))
