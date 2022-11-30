import io
import os
import re

from setuptools import find_packages
from setuptools import setup
import versioneer

exec(open('matheo/_version.py').read())


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())

setup(
    version=__version__,
    name="matheo",
    url="https://gitlab.npl.co.uk/eco/tools/matheo",
    license="None",
    author="Sam Hunt, Pieter De Vis",
    author_email="sam.hunt@npl.co.uk",
    description="Matheo is a python package with mathematical algorithms for use in earth observation data and tools.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy",
                      "numdifftools",
                      "scipy",
                      "xarray",
                      "pyspectral",
                      "punpy",
                      ],
    extras_require={"dev": ["pre-commit", "tox", "sphinx", "sphinx_rtd_theme"]},
)
