"""matheo - Matheo is a python package with mathematical algorithms for use in earth observation data and tools."""

__author__ = "Kavya Jagan, Sam Hunt, Pieter De Vis <kavya.jagan@npl.co.uk>"
__all__ = []

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
