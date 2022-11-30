.. atbd - algorithm theoretical basis
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _atbd:

Algorithm Theoretical Basis
===========================

Band integration
###################

Earth Observation satellites typically have one or more sensors with multiple bands.
The spectral response function (SRF) of each band of the sensor describes its relative sensitivity
to energy of different wavelengths and is normally determined in the laboratory using
a tunable laser or a scanning monochromator.

The band data observed by the sensor are given by:

$`d_{\mathrm{int}} = \frac{\int r(x_r) d(x) \mathop{}\!\mathrm{d} x}{\int r(x_r) \mathop{}\!\mathrm{d} x} `$

where:

* $`d`$ - data
* $`x`$ - data coordinates
* $`r`$ - band response function
* $`x_r`$ - band response function coordinates

In the case that $`x = x_r`$ and $`x`$ is evenly-sampled (i.e. a common spacing between every consecutive $`x`$ element), this reduces to:


$`d_{\mathrm{int}} = \frac{r \cdot d}{\sum r} `$

This formulation also applies for the case where $`r`$ defines multiple band response functions as an N x M array, where N is number of response bands and M is the length of $`x_r`$.

Within matheo, the matheo.band_integration.band_int() function implements this spectral integration ove the SRF in an efficient way (automatically using either the above sum and dot product, or trapezium rule depending on whether the coordinates are evenly-sampled).
Alternatively, the matheo.band_integration.spectral_band_int_sensor() function can be used to automatically look up the values of the SRF using pyspectral, rather than manually providing $`r`$ and $`x_r`$.

Uncertainties can also be provided on the data, coordinates and SRF, in which case these will be propagated to the band data observed by the sensor using the `punpy <https://punpy.readthedocs.io/en/latest/>`_ Monte Carlo approach.

Utilities
############

There are also a number of functions defined in matheo.utils.function_def, which can be used to build SRF of a specific shape, or for other general uses.
These functions include tophat, Gaussian and triangular functions, as well as utilities to make normalised and repeating functions.
