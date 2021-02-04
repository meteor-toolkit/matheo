"""describe class"""

"""___Built-In Modules___"""
import matheo.linear_algebra.matrix_conversion as conv

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MCSamples:
    def __init__(self, MCsteps):
        self.MCsteps = MCsteps

    def generate_samples_correlated(self, param, u_param, corr_param):
        """
        Generate correlated MC samples of input quantity with given uncertainties and correlation matrix.
        Samples are generated using generate_samples_cov() after matching up the uncertainties to the right correlation matrix.
        It is possible to provide one correlation matrix to be used for each measurement (which each have an uncertainty) or a correlation matrix per measurement.

        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis, or list of correlation matrices for each repeated measurement.
        :type corr_x: list[array], optional
        :return: generated samples
        :rtype: array
        """
        if len(param.shape) == 2:
            if len(corr_param) == len(u_param):
                MC_data = np.zeros((u_param.shape) + (self.MCsteps,))
                for j in range(len(u_param[0])):
                    cov_x = conv.convert_corr_to_cov(corr_param, u_param[:, j])
                    MC_data[:, j, :] = self.generate_samples_cov(
                        param[:, j].flatten(), cov_x
                    ).reshape(param[:, j].shape + (self.MCsteps,))
            else:
                MC_data = np.zeros((u_param.shape) + (self.MCsteps,))
                for j in range(len(u_param[:, 0])):
                    cov_x = conv.convert_corr_to_cov(corr_param, u_param[j])
                    MC_data[j, :, :] = self.generate_samples_cov(
                        param[j].flatten(), cov_x
                    ).reshape(param[j].shape + (self.MCsteps,))
        else:
            cov_x = conv.convert_corr_to_cov(corr_param, u_param)
            MC_data = self.generate_samples_cov(param.flatten(), cov_x).reshape(
                param.shape + (self.MCsteps,)
            )

        return MC_data

    def generate_samples_random(self, param, u_param):
        """
        Generate MC samples of input quantity with random (Gaussian) uncertainties.

        :param param: values of input quantity (mean of distribution)
        :type param: float or array
        :param u_param: uncertainties on input quantity (std of distribution)
        :type u_param: float or array
        :return: generated samples
        :rtype: array
        """
        if not hasattr(param, "__len__"):
            return np.random.normal(size=self.MCsteps) * u_param + param
        elif len(param.shape) == 1:
            return (
                np.random.normal(size=(len(param), self.MCsteps)) * u_param[:, None]
                + param[:, None]
            )
        elif len(param.shape) == 2:
            return (
                np.random.normal(size=param.shape + (self.MCsteps,))
                * u_param[:, :, None]
                + param[:, :, None]
            )
        elif len(param.shape) == 3:
            return (
                np.random.normal(size=param.shape + (self.MCsteps,))
                * u_param[:, :, :, None]
                + param[:, :, :, None]
            )
        else:
            print("parameter shape not supported")
            exit()

    def generate_samples_systematic(self, param, u_param):
        """
        Generate correlated MC samples of input quantity with systematic (Gaussian) uncertainties.

        :param param: values of input quantity (mean of distribution)
        :type param: float or array
        :param u_param: uncertainties on input quantity (std of distribution)
        :type u_param: float or array
        :return: generated samples
        :rtype: array
        """
        if not hasattr(param, "__len__"):
            return np.random.normal(size=self.MCsteps) * u_param + param
        elif len(param.shape) == 1:
            return (
                np.dot(u_param[:, None], np.random.normal(size=self.MCsteps)[None, :])
                + param[:, None]
            )
        elif len(param.shape) == 2:
            return (
                np.dot(
                    u_param[:, :, None],
                    np.random.normal(size=self.MCsteps)[:, None, None],
                )[:, :, :, 0]
                + param[:, :, None]
            )
        elif len(param.shape) == 3:
            return (
                np.dot(
                    u_param[:, :, :, None],
                    np.random.normal(size=self.MCsteps)[:, None, None, None],
                )[:, :, :, :, 0, 0]
                + param[:, :, :, None]
            )
        else:
            print("parameter shape not supported")
            exit()

    def generate_samples_cov(self, param, cov_param):
        """
        Generate correlated MC samples of input quantity with a given covariance matrix.
        Samples are generated independent and then correlated using Cholesky decomposition.

        :param param: values of input quantity (mean of distribution)
        :type param: array
        :param cov_param: covariance matrix for input quantity
        :type cov_param: array
        :return: generated samples
        :rtype: array
        """
        try:
            L = np.linalg.cholesky(cov_param)
        except:
            L = conv.nearestPD_cholesky(cov_param)

        return (
            np.dot(L, np.random.normal(size=(len(param), self.MCsteps)))
            + param[:, None]
        )

    def correlate_samples_corr(self, samples, corr):
        """
        Method to correlate independent samples of input quantities using correlation matrix and Cholesky decomposition.

        :param samples: independent samples of input quantities
        :type samples: array[array]
        :param corr: correlation matrix between input quantities
        :type corr: array
        :return: correlated samples of input quantities
        :rtype: array[array]
        """
        if np.max(corr) > 1.000001 or len(corr) != len(samples):
            raise ValueError(
                "The correlation matrix between variables is not the right shape or has elements >1."
            )
        else:
            try:
                L = np.array(np.linalg.cholesky(corr))
            except:
                L = conv.nearestPD_cholesky(corr)

            # Cholesky needs to be applied to Gaussian distributions with mean=0 and std=1,
            # We first calculate the mean and std for each input quantity
            means = np.array([np.mean(samples[i]) for i in range(len(samples))])
            stds = np.array([np.std(samples[i]) for i in range(len(samples))])

            # We normalise the samples with the mean and std, then apply Cholesky, and finally reapply the mean and std.
            print(means.shape, stds.shape, samples.shape)
            if all(stds != 0):
                return np.dot(L, (samples - means) / stds) * stds + means

            # If any of the variables has no uncertainty, the normalisation will fail. Instead we leave the parameters without uncertainty unchanged.
            else:
                samples_out = samples[:]
                id_nonzero = np.where(stds != 0)
                samples_out[id_nonzero] = (
                    np.dot(
                        L[id_nonzero][:, id_nonzero],
                        (samples[id_nonzero] - means[id_nonzero]) / stds[id_nonzero],
                    )[:, 0]
                    * stds[id_nonzero]
                    + means[id_nonzero]
                )
                return samples_out
