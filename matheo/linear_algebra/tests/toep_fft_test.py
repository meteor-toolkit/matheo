# Test FFT Toeplitz mult against PMH Matlab version KTMfun.m
# T:\PUBLIC\KJ2\QA4EO\Regridding

import numpy as np
from matheo.linear_algebra.Toeplitz import Toeplitz

K = np.array([9, 1, 3, 2])
x = np.array([5, 1, 4, 3])
x2 = np.array([[5, 1, 4, 3], [3, 5, 2, 1]])

Tclass = Toeplitz(x2, K)
print(Tclass.toepfftmult())




