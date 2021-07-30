"""
Functions to band integrate spectra for given spectral response function.
"""

from punpy import MCPropagation
import numpy as np
from typing import Union
from matheo.interpolation.interpolation import interpolate
import pandas as pd
import multiprocessing


__author__ = ["Sam Hunt", "Mattea Goalen"]
__created__ = "24/7/2020"


def max_dim(arrays):
    """
    Return max dimension of input numpy arrays

    :type arrays: iter
    :param arrays: n input numpy arrays

    :return: maximum dimension
    :rtype: float
    """

    dims = []
    for array in arrays:
        dims.append(np.ndim(array))

    return max(dims)


def unc_to_dim(unc, dim, x=None, x_len=None):

    original_dim = np.ndim(unc)

    if (original_dim > 2) or (dim > 2):
        raise ValueError(
            "Can only raise uncertainty to a max dimension of 2 (e.g. covariance matrix)"
        )

    if original_dim == dim:
        return unc
    elif unc is None:
        return None
    else:
        if original_dim == 0:
            if (x is None) and (x_len is None):
                raise AttributeError(
                    "Please define either x or x_len to raise dimension of shape 0 uncertainty"
                )
            elif x is not None:
                unc *= x
            else:
                unc = np.full(x_len, unc)

        if dim == 1:
            return unc

        return np.diag(unc)


def _func_with_unc(func, **kwargs):

    # unpack kwargs
    u_params = {k: v for k, v in kwargs.items() if k.startswith('u_')}
    params = {k: v for k, v in kwargs.items() if k not in u_params.keys()}

    # evaluate function
    y = func(**params)

    # if no uncertainties return only in band spectrum
    if all(v == 0 for v in u_params.values()):
        return y

    # Add None's for any undefined uncertainties
    u_params_missing = {k: None for k in kwargs.keys() if "u_"+k not in u_params}
    u_params = {**u_params, **u_params_missing}

    prop = MCPropagation(1000, parallel_cores=multiprocessing.cpu_count())

    # Find max dimension of uncertainty data
    unc_dim = max_dim(u_params.values())
    if unc_dim != 2:
        unc_dim = 1

    u_params_dims = {k: v for k, v in kwargs.items() if k.startswith('u_')}
    u_srf = unc_to_dim(u_srf, unc_dim, x=srf)
    u_spectrum = unc_to_dim(u_spectrum, unc_dim, x=spectrum)
    u_wl_spectrum = unc_to_dim(u_wl_spectrum, unc_dim)
    u_wl_srf = unc_to_dim(u_wl_srf, unc_dim)

    # Propagate uncertainties
    if unc_dim == 1:
        u_spectrum_band = prop.propagate_random(
            func=func,
            x=[spectrum, wl_spectrum, srf, wl_srf],
            u_x=[u_spectrum, u_wl_spectrum, u_srf, u_wl_srf],
        )
    elif unc_dim == 2:
        u_spectrum_band = prop.propagate_cov(
            func=_band_integrate_measurement_function,
            x=[spectrum, wl_spectrum, srf, wl_srf],
            cov_x=[u_spectrum, u_wl_spectrum, u_srf, u_wl_srf],
            return_corr=False,
        )
    else:
        u_spectrum_band = None

    del prop

    return d_band, u_band

def cutout_nonzero(y, x, buffer=0.2):
    """
    Returns continuous non-zero part of function y(x)

    :type y: numpy.ndarray
    :param y: function data values

    :type x: numpy.ndarray
    :param x: function coordinate data values

    :type buffer: float
    :param buffer: fraction of non-zero section of y to include as buffer on either side (default: 0.2)
    """

    # Find extent of non-zero region
    idx = np.nonzero(y)
    imin = min(idx[0])
    imax = max(idx[0]) + 1

    # Determine buffer
    width = imax - imin

    imin -= int(width * buffer)
    imax += int(width * buffer)

    imin = imin if imin >= 0 else 0
    imax = imax if imax <= len(y) else len(y)

    return y[imin:imax], x[imin:imax], [imin, imax]