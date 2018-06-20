import numpy as np


class SeparatingHyperplane:
    def __init__(self, rho=1., randomize_initials=False):
        self.rho = rho
        self.betas = []
        self.beta0s= []
        self.randomize_initials = randomize_initials
        self.X = None
        self.Y = None

    def fit(self, X, Y):
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

        self.X = X
        self.Y = Y

        # Hyperplane equation:
        #    x.dot(beta) + beta0 = 0
        #
        # Initialize beta and beta0
        if self.randomize_initials:
            # between [-1, 1]
            beta = np.random.rand(n, 1) * 2 - 1

            # between [-5, 5]
            beta0 = np.random.uniform() * 2 - 1
        else:
            beta = np.ones([n, 1])
            beta0 = 1

        M = (X.dot(beta) + beta0) * Y
        self.betas = [beta.copy()]
        self.beta0s = [beta0]

        while (M < 0).any():
            for i, j in np.argwhere(M < 0):
                assert j == 0, "M matrix should have only one column"
                xi = X[i, :].reshape(-1,1)
                yi = Y[i, :][0]
                beta += self.rho * yi * xi
                beta0 += self.rho * yi
            M = (X.dot(beta) + beta0) * Y
            self.betas.append(beta.copy())
            self.beta0s.append(beta0)

        return beta, beta0