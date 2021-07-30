"""
Functions to band integrate spectra for given spectral response function.
"""

from spectral_integration.read_srf import (
    return_srf,
    return_band_centres,
    return_band_names,
)

from matheo.band_integration.utils import cutout_nonzero, func_with_unc
from matheo.utils.function_def import iter_fs
import numpy as np
from typing import Union, Tuple
from matheo.interpolation.interpolation import interpolate


__author__ = "Sam Hunt"
__created__ = "30/7/2021"


def _band_int(d: np.ndarray, x: np.ndarray, r: np.ndarray, x_r: np.ndarray) -> float:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

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

    :param d: data to be band integrated
    :param x: data coordinates along band integration axis
    :param r: band response function
    :param x_r: band response function coordinates
    :param d_axis_x: (default 0) x axis in data array
    :return: band integrated data
    """
    return np.apply_along_axis(_band_int, d_axis_x, d, x, r, x_r)


def _band_int2d_arr(
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

    d_intx = np.apply_along_axis(_band_int, d_axis_x, d, x, rx, x_rx)
    d_intx_inty = np.apply_along_axis(_band_int, d_axis_y, d_intx, y, ry, y_ry)

    return d_intx_inty


def _band_int3d_arr(
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

    d_intx = np.apply_along_axis(_band_int, d_axis_x, d, x, rx, x_rx)
    d_intx_inty = np.apply_along_axis(_band_int, d_axis_y, d_intx, y, ry, y_ry)
    d_intx_inty_intz = np.apply_along_axis(_band_int, d_axis_z, d_intx_inty, z, rz, z_rz)

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
) -> Union[float, np.ndarray]:
    """
    Returns integral of data array over a response band (i.e., d(x) * r(x_r))

    :param d: data to be band integrated
    :param x: data coordinates
    :param r: band response function
    :param x_r: band response function coordinates
    :param d_axis_x: (default 0) if d greater than 1D, specify axis to band integrate along
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data coordinates
    :param u_r: uncertainty in band response function
    :param u_x_r: uncertainty in band response function coordinates

    :return: band integrated data
    """

    return func_with_unc(
        _band_int_arr,
        d=d,
        x=x,
        r=r,
        x_r=x_r,
        d_axis_x=d_axis_x,
        u_d=u_d,
        u_x=u_x,
        u_r=u_r,
        u_x_r=u_x_r,
    )


def band_int2d(
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
) -> Union[float, np.ndarray]:
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
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data coordinates along first band integration axis
    :param u_y: uncertainty in data coordinates along second band integration axis
    :param u_rx: uncertainty in first band response function
    :param u_x_rx: uncertainty in first band response function coordinates
    :param u_ry: uncertainty in second band response function
    :param u_y_ry: uncertainty in second band response function coordinates

    :return: band integrated data
    """

    return func_with_unc(
        _band_int_arr,
        d=d,
        x=x,
        y=y,
        rx=rx,
        x_rx=x_rx,
        ry=ry,
        y_ry=y_ry,
        d_axis_x=d_axis_x,
        d_axis_y=d_axis_y,
        u_d=u_d,
        u_x=u_x,
        u_y=u_y,
        u_rx=u_rx,
        u_x_rx=u_x_rx,
        u_ry=u_ry,
        u_y_ry=u_y_ry
    )


def band_int3d(
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
) -> Union[float, np.ndarray]:
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
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data coordinates along first band integration axis
    :param u_y: uncertainty in data coordinates along second band integration axis
    :param u_z: uncertainty in data coordinates along third band integration axis
    :param u_rx: uncertainty in first band response function
    :param u_x_rx: uncertainty in first band response function coordinates
    :param u_ry: uncertainty in second band response function
    :param u_y_ry: uncertainty in second band response function coordinates
    :param u_rz: uncertainty in second band response function
    :param u_z_rz: uncertainty in third band response function coordinates

    :return: band integrated data
    """

    return func_with_unc(
        _band_int_arr,
        d=d,
        x=x,
        y=y,
        z=z,
        rx=rx,
        x_rx=x_rx,
        ry=ry,
        y_ry=y_ry,
        rz=rz,
        z_rz=z_rz,
        d_axis_x=d_axis_x,
        d_axis_y=d_axis_y,
        d_axis_z=d_axis_z,
        u_d=u_d,
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
        u_rx=u_rx,
        u_x_rx=u_x_rx,
        u_ry=u_ry,
        u_y_ry=u_y_ry,
        u_rz=u_rz,
        u_z_rz=u_z_rz
    )


def band_int_sensor(
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
        np.logical_and(
            sensor_band_centres < max(wl_spectrum),
            sensor_band_centres > min(wl_spectrum),
        )
    )[0]
    sensor_band_centres = sensor_band_centres[valid_idx]
    sensor_band_names = [sensor_band_names[i] for i in valid_idx]

    bands = np.zeros(len(sensor_band_names))
    u_bands = np.zeros(len(sensor_band_names))
    for i, band_name in enumerate(sensor_band_names):
        if u_spectrum is None:
            bands[i] = band_integrate(
                spectrum,
                wl_spectrum,
                platform_name=platform_name,
                sensor_name=sensor_name,
                band_name=band_name,
            )
        else:
            bands[i], u_bands[i] = band_integrate(
                spectrum,
                wl_spectrum,
                platform_name=platform_name,
                sensor_name=sensor_name,
                band_name=band_name,
                u_spectrum=u_spectrum,
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
        band_shape: str = "triangular",
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
    :param band_shape: (default triangular) band shape
    :param d_axis_x: (default 0) if d greater than 1D, specify axis pixels are along
    :param u_d: uncertainty in data
    :param u_x: uncertainty in data coordinates
    :param u_x_pixel: uncertainty in centre of band response per pixel
    :param u_width_pixel: uncertainty in width of band response per pixel

    :return: band integrated data
    """

    d_pixel = np.zeros(len(x_pixel))
    u_d_pixel = np.zeros(len(x_pixel))

    for i, (r, x_r) in enumerate(iter_fs(x_pixel, width_pixel, shape=band_shape)):
        d_pixel[i], u_d_pixel[i] = band_int(d, x, r, x_r, d_axis_x, u_d, u_x)

    return d_pixel, u_d_pixel


if __name__ == "__main__":
    pass
