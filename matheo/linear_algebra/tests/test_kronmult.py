import unittest
import numpy as np
from scipy.linalg import toeplitz
from matheo.linear_algebra.tensor_product import TensorProduct

# matrices to kron
Q = {}
Q[0] = np.array(toeplitz([3, 1, 5]))
Q[1] = np.array(toeplitz([4, 5, 2, 8, 1]))

# outside matrix
X = np.linspace(1, 15, 15)

# For toeplitz
Q2 = {}
Q2[0] = np.array([3, 1, 5])
Q2[1] = np.array([4, 5, 2, 8, 1])


class Testtensorproduct(unittest.TestCase):
    def test_kronmult(self):
        TPclass = TensorProduct()
        expected_result = np.array([1613., 2004., 1476., 1932., 1667.,  785.,  980.,  720.,  940.,  815., 1213., 1524.,
                                    1116., 1452., 1267.])
        result = TPclass.kronmult(X, Q)

        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)


if __name__ == '__main__':
    unittest.main()
