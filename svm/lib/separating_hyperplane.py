import numpy as np


def separating_hyperplane(X, Y, rho):
    '''
    X - m*n matrix,
        m - # of data points,
        n - dimension of data
    Y - m*1 vector, classification of data, be either 1 or -1
    rho - learning rate
    '''
    (m, n) = X.shape
    (_m, _n) = Y.shape
    assert m == _m, "X and Y should have the same number of rows"
    assert _n == 1, "Y should have only one column."

    # Hyperplane equation:
    #    x.dot(beta) + beta0 = 0
    #
    # Initialize beta and beta0
    beta = np.ones((n, 1))
    beta0 = 1

    M = (X.dot(beta) + beta0) * Y

    while (M < 0).any():

        for i, j in np.argwhere(M < 0):
            assert j == 0, "M matrix should have only one column"
            xi = X[i, :].reshape(-1,1)
            yi = Y[i, :][0]
            beta += rho * yi * xi
            beta0 += rho * yi
        M = (X.dot(beta) + beta0) * Y

    return beta, beta0