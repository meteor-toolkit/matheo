"""describe class"""

"""___Built-In Modules___"""
#import here

"""___Third-Party Modules___"""
import warnings
import numdifftools as nd
import numpy as np

"""___NPL Modules___"""
#import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ClassName:
    def __init__(
        self,
        something
    ):
        self.something = something

    def calculate_Jacobian(self, fun, x, Jx_diag=False, step=None):
        """
        Calculate the local Jacobian of function y=f(x) for a given value of x

        :param fun: flattened measurement function
        :type fun: function
        :param x: flattened local values of input quantities
        :type x: array
        :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
        :rtype Jx_diag: bool, optional
        :return: Jacobian
        :rtype: array
        """
        Jfun = nd.Jacobian(fun, step=step)

        if Jx_diag:
            y = fun(x)
            Jfun = nd.Jacobian(fun)
            Jx = np.zeros((len(x), len(y)))
            print(Jx.shape)
            for j in range(len(y)):
                xj = np.zeros(int(len(x) / len(y)))
                for i in range(len(xj)):
                    xj[i] = x[i * len(y) + j]
                print(xj.shape, xj)
                Jxj = Jfun(xj)
                for i in range(len(xj)):
                    Jx[i * len(y) + j, j] = Jxj[0][i]
        else:
            Jx = Jfun(x)

        if len(Jx) != len(fun(x).flatten()):
            warnings.warn(
                "Dimensions of the Jacobian were flipped because its shape "
                "didn't match the shape of the output of the function "
                "(probably because there was only 1 input qty)."
            )
            Jx = Jx.T

        return Jx
