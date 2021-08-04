"""
Functions to band integrate spectra for given spectral response function.
"""

from matheo.band_integration.utils import (
    return_srf,
    return_band_centres,
    return_band_names,
)

from matheo.utils.punpy_util import func_with_unc
from matheo.utils.function_def import iter_f, f_tophat, f_triangle, f_gaussian
import numpy as np
from typing import Union, Tuple
from matheo.interpolation.interpolation import interpolate


__author__ = "Sam Hunt"
__created__ = "30/7/2021"


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


def _band_int(d: np.ndarray, x: np.ndarray, r: np.ndarray, x_r: np.ndarray) -> float:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

    N.B.: This function is intended to be wrapped, so it can be applied to an array and run within punpy

    :param d: data to be band integrated
    :param x: data coordinates
    :param r: band response function
    :param x_r: band response function coordinates
    :return: band integrated data
    """

    # Cut out non-zero part of SRF to minimise risk of interpolation errors and optimise performance
    r, x_r, idx = cutout_nonzero(r, x_r)

    res_d = (max(x) - min(x)) / len(x)
    res_r = (max(x_r) - min(x_r)) / len(x_r)

    # If spectrum lower res than the SRF - interpolate spectrum onto SRF wavelength coordinates before integration
    if res_r < res_d:
        d_interp = interpolate(d, x, x_r)
        return np.trapz(r * d_interp, x_r) / np.trapz(r, x_r)

    # If spectrum lower res than the SRF - interpolate spectrum onto SRF wavelength coordinates before integration
    else:

        # First cut out spectrum to SRF wavelength range to avoid extrapolation errors in interpolation
        idx = np.where(
            np.logical_and(x < max(x_r), x > min(x_r))
        )
        d = d[idx]
        x = x[idx]

        r_interp = interpolate(r, x_r, x)
        return np.trapz(d * r_interp, x) / np.trapz(r_interp, x)


def _band_int_arr(d: np.ndarray, x: np.ndarray, r: np.ndarray, x_r: np.ndarray, d_axis_x: int = 0) -> np.ndarray:
    """
    Band integrates multi-dimensional data array along x axis

    N.B.: This function is intended to be wrapped, so it can be run within punpy

    :param d: data to be band integrated
    :param x: data coordinates along band integration axis
    :param r: band response function
    :param x_r: band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :return: band integrated data
    """

    if d.ndim == 1:
        return np.array([_band_int(d, x=x, r=r, x_r=x_r)])

    return np.apply_along_axis(_band_int, d_axis_x, arr=d, x=x, r=r, x_r=x_r)


def _band_int2ax_arr(
        d: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        x_rx: np.ndarray,
        ry: np.ndarray,
        y_ry: np.ndarray,
        d_axis_x: int = 0,
        d_axis_y: int = 1
) -> np.ndarray:
    """
    Sequentially band integrates multi-dimensional data array along x axis and y axis

    N.B.: This function is intended to be wrapped, so it can be run within punpy

    :param d: data to be band integrated
    :param x: data coordinates along first band integration axis
    :param y: data coordinates along second band integration axis
    :param rx: first band response function
    :param x_rx: first band response function coordinates
    :param ry: second band response function
    :param y_ry: second band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :param d_axis_y: (default 1) y axis in data array
    :return: band integrated data
    """

    d_intx = _band_int_arr(d, x=x, r=rx, x_r=x_rx, d_axis_x=d_axis_x)
    d_intx_inty = _band_int_arr(d_intx, x=y, r=ry, x_r=y_ry, d_axis_x=d_axis_y)

    return d_intx_inty


def _band_int3ax_arr(
        d: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        rx: np.ndarray,
        x_rx: np.ndarray,
        ry: np.ndarray,
        y_ry: np.ndarray,
        rz: np.ndarray,
        z_rz: np.ndarray,
        d_axis_x: int = 0,
        d_axis_y: int = 1,
        d_axis_z: int = 2
) -> np.ndarray:
    """
    Sequentially band integrates multi-dimensional data array along x, y and z axes

    N.B.: This function is intended to be wrapped, so it can be run within punpy

    :param d: data to be band integrated
    :param x: data coordinates along first band integration axis
    :param y: data coordinates along second band integration axis
    :param z: data coordinates along third band integration axis
    :param rx: first band response function
    :param x_rx: first band response function coordinates
    :param ry: second band response function
    :param y_ry: second band response function coordinates
    :param rz: third band response function
    :param z_rz: third band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :param d_axis_y: (default 1) y axis in data array
    :param d_axis_z: (default 2) z axis in data array
    :return: band integrated data
    """

    d_intx = _band_int_arr(d, x=x, r=rx, x_r=x_rx, d_axis_x=d_axis_x)
    d_intx_inty = _band_int_arr(d_intx, x=y, r=ry, x_r=y_ry, d_axis_x=d_axis_y)
    d_intx_inty_intz = _band_int_arr(d_intx_inty, x=z, r=rz, x_r=z_rz, d_axis_x=d_axis_z)

    return d_intx_inty_intz


def band_int(
    d: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    x_r: np.ndarray,
    d_axis_x: int = 0,
    u_d: Union[None, float, np.ndarray] = None,
    u_x: Union[None, float, np.ndarray] = None,
    u_r: Union[None, float, np.ndarray] = None,
    u_x_r: Union[None, float, np.ndarray] = None,
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

    :param d: data to be band integrated
    :param x: data coordinates
    :param r: band response function
    :param x_r: band response function coordinates
    :param d_axis_x: (default 0) if d greater than 1D, specify axis to band integrate along
    :param u_d: (optional) uncertainty in data
    :param u_x: (optional) uncertainty in data coordinates
    :param u_r: (optional) uncertainty in band response function
    :param u_x_r: (optional) uncertainty in band response function coordinates

    :return: band integrated data
    :return: uncertainty of band integrated data (skipped if no input uncertainties provided)
    """

    d_band, u_d_band = func_with_unc(
        _band_int_arr,
        params=dict(d=d, x=x, r=r, x_r=x_r, d_axis_x=d_axis_x),
        u_params=dict(d=u_d, x=u_x, r=u_r, x_r=u_x_r)
    )

    if u_d_band is None:
        return d_band

    return d_band, u_d_band


def band_int2ax(
        d: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        x_rx: np.ndarray,
        ry: np.ndarray,
        y_ry: np.ndarray,
        d_axis_x: int = 0,
        d_axis_y: int = 0,
        u_d: Union[None, float, np.ndarray] = None,
        u_x: Union[None, float, np.ndarray] = None,
        u_y: Union[None, float, np.ndarray] = None,
        u_rx: Union[None, float, np.ndarray] = None,
        u_x_rx: Union[None, float, np.ndarray] = None,
        u_ry: Union[None, float, np.ndarray] = None,
        u_y_ry: Union[None, float, np.ndarray] = None
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """
    Sequentially band integrates multi-dimensional data array along x axis and y axis

    :param d: data to be band integrated
    :param x: data coordinates along first band integration axis
    :param y: data coordinates along second band integration axis
    :param rx: first band response function
    :param x_rx: first band response function coordinates
    :param ry: second band response function
    :param y_ry: second band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :param d_axis_y: (default 1) y axis in data array
    :param u_d: (optional) uncertainty in data
    :param u_x: (optional) uncertainty in data coordinates along first band integration axis
    :param u_y: (optional) uncertainty in data coordinates along second band integration axis
    :param u_rx: (optional) uncertainty in first band response function
    :param u_x_rx: (optional) uncertainty in first band response function coordinates
    :param u_ry: (optional) uncertainty in second band response function
    :param u_y_ry: (optional) uncertainty in second band response function coordinates

    :return: band integrated data
    :return: uncertainty of band integrated data (skipped if no input uncertainties provided)
    """

    d_band, u_d_band = func_with_unc(
        _band_int2ax_arr,
        params=dict(d=d, x=x, y=y, rx=rx, x_rx=x_rx, ry=ry, y_ry=y_ry, d_axis_x=d_axis_x, d_axis_y=d_axis_y),
        u_params=dict(d=u_d, x=u_x, y=u_y, rx=u_rx, x_rx=u_x_rx, ry=u_ry, y_ry=u_y_ry)
    )

    if u_d_band is None:
        return d_band

    return d_band, u_d_band


def band_int3ax(
    d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rx: np.ndarray,
    x_rx: np.ndarray,
    ry: np.ndarray,
    y_ry: np.ndarray,
    rz: np.ndarray,
    z_rz: np.ndarray,
    d_axis_x: int = 0,
    d_axis_y: int = 1,
    d_axis_z: int = 2,
    u_d: Union[None, float, np.ndarray] = None,
    u_x: Union[None, float, np.ndarray] = None,
    u_y: Union[None, float, np.ndarray] = None,
    u_z: Union[None, float, np.ndarray] = None,
    u_rx: Union[None, float, np.ndarray] = None,
    u_x_rx: Union[None, float, np.ndarray] = None,
    u_ry: Union[None, float, np.ndarray] = None,
    u_y_ry: Union[None, float, np.ndarray] = None,
    u_rz: Union[None, float, np.ndarray] = None,
    u_z_rz: Union[None, float, np.ndarray] = None
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """
    Sequentially band integrates multi-dimensional data array along x, y and z axes

    :param d: data to be band integrated
    :param x: data coordinates along first band integration axis
    :param y: data coordinates along second band integration axis
    :param z: data coordinates along third band integration axis
    :param rx: first band response function
    :param x_rx: first band response function coordinates
    :param ry: second band response function
    :param y_ry: second band response function coordinates
    :param rz: third band response function
    :param z_rz: third band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :param d_axis_y: (default 1) y axis in data array
    :param d_axis_z: (default 2) z axis in data array
    :param u_d: (optional) uncertainty in data
    :param u_x: (optional) uncertainty in data coordinates along first band integration axis
    :param u_y: (optional) uncertainty in data coordinates along second band integration axis
    :param u_z: (optional) uncertainty in data coordinates along third band integration axis
    :param u_rx: (optional) uncertainty in first band response function
    :param u_x_rx: (optional) uncertainty in first band response function coordinates
    :param u_ry: (optional) uncertainty in second band response function
    :param u_y_ry: (optional) uncertainty in second band response function coordinates
    :param u_rz: (optional) uncertainty in second band response function
    :param u_z_rz: (optional) uncertainty in third band response function coordinates

    :return: band integrated data
    :return: uncertainty of band integrated data (skipped if no input uncertainties provided)
    """

    params = dict(
        d=d,
        x=x,
        y=y,
        z=z,
        rx=rx,
        x_rx=x_rx,
        ry=ry,
        y_ry=y_ry,
        rz=rz,
        z_ry=z_rz,
        d_axis_x=d_axis_x,
        d_axis_y=d_axis_y,
        d_axis_z=d_axis_z
    )

    u_params = dict(
        d=u_d,
        x=u_x,
        y=u_y,
        z=u_z,
        rx=u_rx,
        x_rx=u_x_rx,
        ry=u_ry,
        y_ry=u_y_ry,
        rz=u_rz,
        z_rz=u_z_rz
    )

    d_band, u_d_band = func_with_unc(_band_int3ax_arr, params=params, u_params=u_params)

    if u_d_band is None:
        return d_band

    return d_band, u_d_band


def spectral_band_int_sensor(
    d: np.ndarray,
    w: np.ndarray,
    platform_name: str,
    sensor_name: str,
    d_axis_w: int = 0,
    u_d: np.ndarray = None,
    u_w: np.ndarray = None,
):
    """
    Returns spectral band integrated data array for named sensor spectral bands

    :param d: data to be band integrated
    :param w: data wavelength coordinates
    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param d_axis_w: spectral axis in data array
    :param u_d: uncertainty in data
    :param u_w: uncertainty in data coordinates along first band integration axis

    :return: band integrated data
    """

    sensor_band_centres = return_band_centres(platform_name, sensor_name)
    sensor_band_names = return_band_names(platform_name, sensor_name)

    valid_idx = np.where(
        np.logical_and(sensor_band_centres < max(w), sensor_band_centres > min(w))
    )[0]

    sensor_band_centres = sensor_band_centres[valid_idx]
    sensor_band_names = [sensor_band_names[i] for i in valid_idx]

    bands = np.zeros(len(sensor_band_names))
    u_bands = np.zeros(len(sensor_band_names))
    for i, band_name in enumerate(sensor_band_names):
        bands[i] = band_integrate(
            spectrum,
            wl_spectrum,
            platform_name=platform_name,
            sensor_name=sensor_name,
            band_name=band_name,
        )

    band_data = pd.DataFrame(
        {
            "band_name": sensor_band_names,
            "band_centre": sensor_band_centres,
            "spectrum_in_band": bands,
            "u_spectrum_in_band": u_bands,
        }
    )

    return band_data


def pixel_int(
        d: np.ndarray,
        x: np.ndarray,
        x_pixel: np.ndarray,
        width_pixel: np.ndarray,
        band_shape: str = "triangle",
        d_axis_x: int = 0,
        u_d: Union[None, float, np.ndarray] = None,
        u_x: Union[None, float, np.ndarray] = None,
        u_x_pixel: Union[None, float, np.ndarray] = None,
        u_width_pixel: Union[None, float, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

    :param d: data to be band integrated
    :param x: data coordinates
    :param x_pixel: centre of band response per pixel
    :param width_pixel: width of band response per pixel
    :param band_shape: (default triangular) band shape - must be one of 'triangle', 'tophat', or 'gaussian'
    :param d_axis_x: (default 0) if d greater than 1D, specify axis pixels are along
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data coordinates
    :param u_x_pixel: uncertainty in centre of band response per pixel
    :param u_width_pixel: uncertainty in width of band response per pixel

    :return: band integrated data
    :return: uncertainty in band integrated data
    """

    d_pixel = np.zeros(len(x_pixel))
    u_d_pixel = np.zeros(len(x_pixel))

    if band_shape == "triangle":
        f = f_triangle
        xlim_width = 1
    elif band_shape == "tophat":
        f = f_tophat
        xlim_width = 1
    elif band_shape == "gaussian":
        f = f_gaussian
        xlim_width = 3
    else:
        raise ValueError("band_shape must be one of ['triangle', 'tophat', 'gaussian']")

    for i, (r, x_r) in enumerate(iter_f(f, x_pixel, width_pixel, xlim_width=xlim_width)):
        d_pixel[i], u_d_pixel[i] = band_int(d, x, r, x_r, d_axis_x, u_d, u_x)

    return d_pixel, u_d_pixel


def _band_int2d(
        d: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        psf: np.ndarray,
        x_psf: np.ndarray,
        y_psf: np.ndarray
) -> float:
    """
    Returns integral of a 2D data array over a response band defined by a 2D point spread function
    (i.e., d(x, y) * psf(x_psf, y_psf))

    N.B.: This function is intended to be wrapped, so it can be applied to an array and run within punpy

    :param d: two dimensional data to be band integrated
    :param x: data x coordinates
    :param y: data y coordinates
    :param psf: two dimensional point spread function of band response
    :param x_psf: psf x coordinates
    :param y_psf: psf y coordinates
    :return: band integrated data
    """

    # todo - implement _band_int2d
    raise NotImplementedError


def _band_int2d_arr(
        d: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        psf: np.ndarray,
        x_psf: np.ndarray,
        y_psf: np.ndarray,
        d_axis_x: int = 0,
        d_axis_y: int = 1
) -> np.ndarray:
    """
    Integrates two dimensional slice of multi-dimensional data array over a response band defined by a 2D point spread
    function

    N.B.: This function is intended to be wrapped, so it can be run within punpy

    :param d: two dimensional data to be band integrated
    :param x: data x coordinates
    :param y: data y coordinates
    :param psf: two dimensional point spread function of band response
    :param x_psf: psf x coordinates
    :param y_psf: psf y coordinates
    :param d_axis_x: (default 0) x axis in data array
    :param d_axis_y: (default 1) y axis in data array
    :return: band integrated data
    """

    # todo - implement _band_int2d_arr
    raise NotImplementedError


def band_int2d(
    d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    psf: np.ndarray,
    x_psf: np.ndarray,
    y_psf: np.ndarray,
    d_axis_x: int = 0,
    d_axis_y: int = 1,
    u_d: Union[None, float, np.ndarray] = None,
    u_x: Union[None, float, np.ndarray] = None,
    u_y: Union[None, float, np.ndarray] = None,
    u_psf: Union[None, float, np.ndarray] = None,
    u_x_psf: Union[None, float, np.ndarray] = None,
    u_y_psf: Union[None, float, np.ndarray] = None,
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """
    Returns integral of a 2D data array over a response band defined by a 2D point spread function
    (i.e., d(x, y) * psf(x_psf, y_psf))

    :param d: two dimensional data to be band integrated
    :param x: data x coordinates
    :param y: data y coordinates
    :param psf: two dimensional point spread function of band response
    :param x_psf: psf x coordinates
    :param y_psf: psf y coordinates
    :param d_axis_x: (default 0) x axis in data array, if d more than 2D
    :param d_axis_y: (default 1) y axis in data array, if d more than 2D
    :param u_d: (optional) uncertainty in data
    :param u_x: (optional) uncertainty in data x coordinates
    :param u_y: (optional) uncertainty in data y coordinates
    :param u_psf: (optional) uncertainty in point spread function of band response
    :param u_x_psf: (optional) uncertainty in psf x coordinates
    :param u_y_psf: (optional) uncertainty in psf x coordinates

    :return: band integrated data
    :return: uncertainty of band integrated data (skipped if no input uncertainties provided)
    """

    d_band, u_d_band = func_with_unc(
        _band_int2d_arr,
        params=dict(d=d, x=x, y=y, psf=psf, x_psf=x_psf, y_psf=y_psf, d_axis_x=d_axis_x, d_axis_y=d_axis_y),
        u_params=dict(d=u_d, x=u_x, y=u_y, psf=u_psf, x_psf=u_x_psf, y_psf=u_y_psf)
    )

    if u_d_band is None:
        return d_band

    return d_band, u_d_band


def pixel_int2d(
    d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_pixel: np.ndarray,
    y_pixel: np.ndarray,
    width_pixel: np.ndarray,
    psf_shape: str = "triangle",
    d_axis_x: int = 0,
    d_axis_y: int = 0,
    u_d: Union[None, float, np.ndarray] = None,
    u_x: Union[None, float, np.ndarray] = None,
    u_y: Union[None, float, np.ndarray] = None,
    u_x_pixel: Union[None, float, np.ndarray] = None,
    u_y_pixel: Union[None, float, np.ndarray] = None,
    u_width_pixel: Union[None, float, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

    :param d: data to be band integrated
    :param x: data x coordinates
    :param y: data y coordinates
    :param x_pixel: x positions of centre of psf per pixel
    :param y_pixel: y positions of centre of psf per pixel
    :param width_pixel: width of psf per pixel
    :param psf_shape: (default X) psf shape - must be one of...
    :param d_axis_x: (default 0) x axis in data array, if d more than 2D
    :param d_axis_y: (default 1) y axis in data array, if d more than 2D
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data x coordinates
    :param u_y: uncertainty in data y coordinates
    :param u_x_pixel: uncertainty in x positions of centre of psf per pixel
    :param u_y_pixel: uncertainty in y positions of centre of psf per pixel
    :param u_width_pixel: uncertainty in width of psf per pixel

    :return: band integrated data
    :return: uncertainty in band integrated data
    """

    # todo - implement _band_int2d_arr
    raise NotImplementedError


if __name__ == "__main__":
    pass
