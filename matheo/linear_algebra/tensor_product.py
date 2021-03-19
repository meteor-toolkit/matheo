"""describe class"""

"""___Built-In Modules___"""
import numpy as np
import sys
from matheo.linear_algebra.Toeplitz import Toeplitz

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class TensorProduct:
    def __init__(self):
        pass

    def kronmult(self, x, Q):
        """
        Function to evaluate the product of a matrix with a tensor/kronecker product of matrices

        :param x: matrix or vector
        :param Q: matrices that need to be "tensor producted" - Q1, Q2, ...
        :return: (Q1 kron Q2 kron ...)*x

        Created: 08-02-2021
        """

        N = len(Q)
        n = np.zeros(N)
        nright = 1
        nleft = 1

        for i in range(N-1):
            n[i] = Q[i].shape[0]
            nleft = int(nleft*n[i])

        n[N-1] = Q[N-1].shape[0]

        for i in range(N-1, -1, -1):
            base = 0
            jump = n[i]*nright
            for k in range(nleft):
                for j in range(nright):
                    index1 = base + j
                    index2 = base + j + nright*(n[i] - 1) + 1
                    index1 = int(index1)
                    index2 = int(index2)
                    nright = int(nright)
                    if Q[i].ndim == 2:
                        x[index1:index2:nright] = np.matmul(Q[i], x[index1:index2:nright])
                    elif Q[i].ndim == 1:
                        tclass = Toeplitz()
                        x[index1:index2:nright] = tclass.toepfftmult(x[index1:index2:nright], Q[i])
                    else:
                        sys.exit('Funky dimension of Q!')
                base = base + jump
            nleft = int(nleft/n[np.max(i-1, 0)])
            nright = int(nright*n[i])

        X = x
        return X

