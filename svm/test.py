import numpy as np
from separating_hyperplane import SeparatingHyperplane
import unittest


class SeparatingHyperplaneTest(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray([
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 1],
            [2, 2],
            [1, 2]], dtype=float)

        self.Y = np.asarray([-1, -1, -1, 1, 1, 1], dtype=int).reshape([-1, 1])

    def test_initial_value_1(self):
        fitter = SeparatingHyperplane()
        beta, beta0 = fitter.fit(self.X, self.Y)
        print(beta, beta0)

    def test_randomized_initials(self):
        fitter = SeparatingHyperplane(randomize_initials=True)

        beta, beta0 = fitter.fit(self.X, self.Y)

        print(fitter.betas, fitter.beta0s)


if __name__ == '__main__':
    unittest.main()
