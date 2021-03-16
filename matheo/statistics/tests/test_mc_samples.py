"""Tests for classname module"""

"""___Built-In Modules___"""
from matheo.statistics.mc_samples import MCSamples
import matheo.linear_algebra.matrix_conversion as conv
from matheo.linear_algebra.toeplitz import Toeplitz
"""___Third-Party Modules___"""
import unittest
import numpy as np
import numpy.testing as npt

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

test_x = np.arange(5, 10, 0.1)
test_ux = np.ones(len(test_x))
test_corr_rand = np.eye(len(test_x))
test_corr_syst = np.ones_like(test_corr_rand)
test_cov_rand = conv.convert_corr_to_cov(test_corr_rand, test_ux)
test_cov_syst = conv.convert_corr_to_cov(test_corr_syst, test_ux)


class TestMCSamples(unittest.TestCase):
    def test_generate_samples_correlated(self):
        MCgen = MCSamples(100000)
        MCsample = MCgen.generate_samples_correlated(test_x, test_ux, test_corr_rand)
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_rand, np.corrcoef(MCsample), atol=0.015)

    def test_generate_samples_random(self):
        MCgen = MCSamples(100000)
        MCsample = MCgen.generate_samples_random(test_x, test_ux)
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_rand, np.corrcoef(MCsample), atol=0.015)

    def test_generate_samples_systematic(self):
        MCgen = MCSamples(100000)
        MCsample = MCgen.generate_samples_systematic(test_x, test_ux)
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_syst, np.corrcoef(MCsample), atol=0.015)

    def test_generate_samples_cov(self):
        MCgen = MCSamples(100000)
        MCsample = MCgen.generate_samples_cov(test_x, test_cov_rand)
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_rand, np.corrcoef(MCsample), atol=0.015)

        MCsample = MCgen.generate_samples_cov(test_x, test_cov_syst)
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_syst, np.corrcoef(MCsample), atol=0.015)

    def test_correlate_samples_corr(self):
        MCgen = MCSamples(100000)
        MC_data = np.empty(len(test_x), dtype=np.ndarray)
        for i in range(len(test_x)):
            MC_data[i] = MCgen.generate_samples_random(test_x[i], test_ux[i])
        MCsample = MCgen.correlate_samples_corr(MC_data, test_corr_syst)
        MCsample = [MCsample[i] for i in range(len(MCsample))]
        npt.assert_allclose(test_x, np.mean(MCsample, axis=1), rtol=0.01)
        npt.assert_allclose(test_ux, np.std(MCsample, axis=1), atol=0.01)
        npt.assert_allclose(test_corr_syst, np.corrcoef(MCsample), atol=0.015)


if __name__ == "__main__":
    unittest.main()
