import numpy as np
from separating_hyperplane import separating_hyperplane


X = np.asarray([
    [0, 0],
    [1, 0],
    [0, 1],
    [2, 1],
    [2, 2],
    [1, 2]], dtype=float)

Y = np.asarray([-1, -1, -1, 1, 1, 1], dtype=int).reshape([-1, 1])

beta, beta0 = separating_hyperplane(X, Y, 1.)

print(beta, beta0)
