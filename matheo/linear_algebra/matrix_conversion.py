"""Class with methods that convert a covariance or correlation matrix into a related metric"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import warnings
import numpy as np

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

def nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=True):
    """
    Find the nearest positive-definite matrix

    :param A: correlation matrix or covariance matrix
    :type A: array
    :return: nearest positive-definite matrix
    :rtype: array

    Copied and adapted from [1] under BSD license.
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], which
    credits [3].
    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    try:
        return np.linalg.cholesky(A3)
    except:

        spacing = np.spacing(np.linalg.norm(A))

        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        if corr == True:
            A3 = correlation_from_covariance(A3)
            maxdiff = np.max(np.abs(A - A3))
            if maxdiff > diff:
                raise ValueError(
                    "One of the correlation matrices is not postive definite. "
                    "Correlation matrices need to be at least positive "
                    "semi-definite."
                )
            else:
                warnings.warn(
                    "One of the correlation matrices is not positive "
                    "definite. It has been slightly changed (maximum difference "
                    "of %s) to accomodate our method." % (maxdiff)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3
        else:
            maxdiff = np.max(np.abs(A - A3) / (A3 + diff))
            if maxdiff > diff:
                raise ValueError(
                    "One of the provided covariance matrices is not postive "
                    "definite. Covariance matrices need to be at least positive "
                    "semi-definite. Please check your covariance matrix."
                )
            else:
                warnings.warn(
                    "One of the provided covariance matrix is not positive"
                    "definite. It has been slightly changed (maximum difference of "
                    "%s percent) to accomodate our method." % (maxdiff * 100)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3


def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky

    :param B: matrix
    :type B: array
    :return: true when input is positive-definite
    :rtype: bool
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def correlation_from_covariance(covariance):
    """
    Convert covariance matrix to correlation matrix

    :param covariance: Covariance matrix
    :type covariance: array
    :return: Correlation matrix
    :rtype: array
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def uncertainty_from_covariance(covariance):
    """
    Convert covariance matrix to uncertainty

    :param covariance: Covariance matrix
    :type covariance: array
    :return: uncertainties
    :rtype: array
    """
    return np.sqrt(np.diag(covariance))


def convert_corr_to_cov(corr, u):
    """
    Convert correlation matrix to covariance matrix

    :param corr: correlation matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: covariance matrix
    :rtype: array
    """
    return u.reshape((-1, 1)) * corr * (u.reshape((1, -1)))


def convert_cov_to_corr(cov, u):
    """
    Convert covariance matrix to correlation matrix

    :param corr: covariance matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: correlation matrix
    :rtype: array
    """
    return 1 / u.reshape((-1, 1)) * cov / (u.reshape((1, -1)))
