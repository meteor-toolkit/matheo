
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


def interpolate(y_1, x_1, x_2):
    """
    Interpolates data to defined coordinates

    :type y_1: numpy.ndarray
    :param y_1: data to interpolate

    :type x_1: numpy.ndarray
    :param x_1: initial coordinate data of y_1

    :type x_2: numpy.ndarray
    :param x_2: coordinate data to interpolate y_1 to

    :return: interpolate data
    :rtype: numpy.ndarray
    """

    y_1 = y_1[~np.isnan(x_1)]
    x_1 = x_1[~np.isnan(x_1)]

    x_1 = x_1[~np.isnan(y_1)]
    y_1 = y_1[~np.isnan(y_1)]

    ius = InterpolatedUnivariateSpline(x_1, y_1)
    return ius(x_2)