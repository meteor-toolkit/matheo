"""
Functions to read spectral response function data with pyspectral
"""

import numpy as np
from pyspectral.rsr_reader import RelativeSpectralResponse


"""___Authorship___"""
__author__ = "Sam Hunt"
__created__ = "5/11/2020"


def return_band_names(platform_name, sensor_name):
    """
    Returns band names for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :type platform_name: str
    :param platform_name: satellite name

    :type sensor_name: str
    :param sensor_name: name of instrument on satellite

    :return: band names
    :rtype: list
    """

    sensor = RelativeSpectralResponse(platform_name, sensor_name)

    return list(sensor.rsr.keys())


def return_band_centres(platform_name, sensor_name, detector_name=None):
    """
    Returns band centres for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :type platform_name: str
    :param platform_name: satellite name

    :type sensor_name: str
    :param sensor_name: name of instrument on satellite

    :type detector_name: str
    :param detector_name: (optional) name of sensor detector. Can be used in sensor has SRF data for for different detectors separately - if not specified in this case different

    :return: band centres in nm
    :rtype: np.ndarray
    """

    if detector_name is None:
        detector_name = "det-1"

    sensor = RelativeSpectralResponse(platform_name, sensor_name)
    band_names = return_band_names(platform_name, sensor_name)

    band_centres = np.array(
        [
            sensor.rsr[band_name][detector_name]["central_wavelength"]
            for band_name in band_names
        ]
    )

    # convert to nm
    band_centres = band_centres * sensor.si_scale / 1e-9

    return band_centres


def return_srf(platform_name, sensor_name, band_name, detector_name=None):
    """
    Returns spectral response function for specified sensor band from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :type platform_name: str
    :param platform_name: satellite name

    :type sensor_name: str
    :param sensor_name: name of instrument on satellite

    :type band_name: str
    :param band_name: name of sensor band

    :type detector_name: str
    :param detector_name: (optional) name of sensor detector. Can be used in sensor has SRF data for for different detectors separately - if not specified in this case different

    :return: spectral response function
    :rtype: np.ndarray

    :return: wavelength coordinates for spectral data
    :rtype: np.ndarray
    """

    if detector_name is None:
        detector_name = "det-1"

    sensor = RelativeSpectralResponse(platform_name, sensor_name)
    srf = sensor.rsr[band_name][detector_name]["response"]  # gets rsr for given band
    wl_srf = 1000 * sensor.rsr[band_name][detector_name]["wavelength"]

    return srf, wl_srf


if __name__ == "__main__":
    pass
