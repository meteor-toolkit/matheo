"""describe class"""

"""___Built-In Modules___"""
import numpy as np
import numpy.matlib
from scipy.fft import fft, ifft
import sys

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


class Toeplitz:
    def __init__(self, x, K):
        self.x = x
        self.K = K

    def toepfftmult(self):
        """
        Function to evaluate Toeplitz matrix-vector or Toeplitz matrix-matrix product using FFT

        :param x: matrix or vector
        :param K: vector that defines the Toeplitz matrix (1st row of the matrix)
        :return: K*x

        Created: 08-02-2021
        """

        # Get size of x vector or matrix and create a padded y matrix of double the length
        lx = self.x.ndim
        if lx == 2:
            n, m = self.x.shape
            y = np.concatenate((self.x, np.zeros(self.x.shape)), axis=1)
        elif lx == 1:
            m = self.x.shape[0]
            n = 1
            y = np.concatenate((self.x, np.zeros(self.x.shape)))
        else:
            sys.exit('Funky dimension of x!')

        # Build circulant vector from Toeplitz vector and take fft
        ct = np.concatenate((self.K, np.zeros(1), self.K[:0:-1])).T
        fc = np.matlib.repmat(fft(ct), n, 1)[0]

        # FFT multiplication
        cy = ifft(np.multiply(fc, fft(y)))
        cy = cy.real
        if n == 1:
            return cy[0:m]
        elif n > 1:
            return cy[:, 0:m]





