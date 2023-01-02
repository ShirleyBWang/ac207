#!/usr/bin/env python3
# File       : forward.py
# Description: forward: Defines a forward mode class
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang and Alaric Wei
"""This module contains Forward class, which inherits AutoDiff. 

The module will take in functions to perform autmatic differentiation in forward
mode, and be able to calculate the value of the functions and the jacobian matrix
of the functions at given values.
"""

import numpy as np
from pomeranian.autodiff import AutoDiff
from pomeranian.dual import Dual


class Forward(AutoDiff):
    """Forward Mode Automatic Differentiation Class, inherit AutoDiff.

    The class defines three dunder methods: __init__, __repr__ and __str__.

    __init__: Takes the function that have to be automatically differentiated 
    via forward mode 

    _dual_forward: Converts inputs into an array of dual numbers for both scalar and
    multivariable inputs, and calculate values and derivatives using Dual.

    get_value: Evaluates the function at inputs.

    forward: Computes jacobian matrix using forward mode AD.
    """

    def __init__(self, funcs):
        """Constructor of Forward class.
        
        Parameters
        ----------
        funcs: callable
            Functions to perform forward mode AD on
        """

        super().__init__(funcs)

    def _dual_forward(self, inputs):
        """Converts inputs into an array of dual numbers for both scalar and
        multivariable inputs, and calculate values and derivatives using Dual.
        
        Parameters
        ----------
        inputs : int, float, or a list-like of int/float
            Input values
        
        Returns
        -------
        Dual or array
            Result dual number if univariate inputs and functions, result
            jacobian matrix of duals if multivariate inputs and functions

        Examples
        --------
        >>> f = lambda x: x * 2
        >>> ad = Forward(f)
        >>> print(ad._dual_forward(3))
        Dual(6,2)
        >>> f = lambda x, y: [x * 2 + y ** 2, x * y]
        >>> ad = Forward(f)
        >>> print(ad._dual_forward([3, 4]))
        [[Dual(22,2) Dual(12,4)]
         [Dual(22,8) Dual(12,3)]]
        """

        # convert int, float to list
        if isinstance(inputs, (int, float)):
            inputs = np.asarray([inputs])

        # n_inputs mismatch
        if len(inputs) != self.n_inputs:
            raise Exception(f"Number of inputs (={len(inputs)}) is not " +
                            "consistent with number of function " +
                            f"arguments (={self.n_inputs}).")

        # output result conditioned on n_input, n_func
        if len(inputs) == 1:  # if univariate inputs

            # convert inputs into Dual numbers
            duals = np.asarray([Dual(input) for input in inputs])

            # parse into functions
            return self.funcs(*duals)

        else:  # if multivariate inputs

            jacob = []

            # convert inputs into Dual numbers
            duals = np.asarray([Dual(input, 0) for input in inputs])

            # parse into functions
            for idx in range(len(inputs)):
                duals[idx] = Dual(duals[idx].real, 1)  # condition on one var
                jacob.append(self.funcs(*duals))  # calculate jacobian value
                duals[idx] = Dual(duals[idx].real, 0)  # change back

            # convert jacobian to array
            return np.asarray(jacob)

    def get_value(self, inputs):
        """Evaluates the function at inputs.
                
        Parameters
        ----------
        inputs : int, float, or a list-like of int/float
            Input values
        
        Returns
        -------
        int, float, or array
            Function value at inputs if univariate inputs and functions,
            matrix of function value if multivariate inputs and functions

        Examples
        --------
        >>> f = lambda x: x * 2
        >>> ad = Forward(f)
        >>> print(ad.get_value(3))
        6
        >>> print(ad.get_value([4]))
        8
        >>> f = lambda x, y: [x * 2 + y * y, x * y]
        >>> ad = Forward(f)
        >>> print(ad.get_value([3, 4]))
        [22 12]
        """

        # calculate with forward mode
        result = self._dual_forward(inputs)

        # convert int, float to list
        if isinstance(inputs, (int, float)):
            inputs = np.asarray([inputs])

        # output result conditioned on n_input, n_func
        if len(inputs) == 1:  # if univariate inputs

            if isinstance(result, Dual):  # only one function/output
                return result.real

            else:  # multiple functions/outputs
                return np.asarray([elem.real for elem in result])

        else:  # if multivariate inputs

            if isinstance(result[0], Dual):  # only one function/output
                return np.asarray([elem.real for elem in result])[0]

            else:  # multiple functions/outputs
                return np.asarray([[elem.real for elem in sets]
                                   for sets in result])[0]

    def forward(self, inputs):
        """Computes jacobian matrix using forward mode AD.
                
        Parameters
        ----------
        inputs : int, float, or a list-like of int/float
            Input values
        
        Returns
        -------
        int, float, or array
            Function derivative value at inputs if univariate inputs and functions,
            jacobain matrix of function if multivariate inputs and functions

        Examples
        --------
        >>> f = lambda x: x * 2
        >>> ad = Forward(f)
        >>> print(ad.forward([4]))
        2
        >>> f = lambda x, y: x * 2 + y ** 2
        >>> ad = Forward(f)
        >>> print(ad.forward([3, 4]))
        [2 8]
        >>> f = lambda x, y: [x ** 2 - y ** 2, x * x]    
        >>> ad = Forward(f)
        >>> print(ad.forward([1, 1]))
        [[ 2 -2]
         [ 2  0]]
        """

        # calculate with forward mode
        result = self._dual_forward(inputs)

        # convert int, float to list
        if isinstance(inputs, (int, float)):
            inputs = np.asarray([inputs])

        # output result conditioned on n_input, n_func
        if len(inputs) == 1:  # if univariate inputs

            # parse into functions
            if isinstance(result, Dual):  # only one function/output
                return result.dual

            else:  # multiple functions/outputs
                return np.asarray([elem.dual for elem in result])

        else:  # if multivariate inputs

            if isinstance(result[0], Dual):  # only one function/output
                return np.asarray([elem.dual for elem in result])

            else:  # multiple functions/outputs
                return np.asarray([[elem.dual for elem in sets]
                                   for sets in result]).T

    def __str__(self):
        """String representation of the class and info about functions."""
        msg = ", computing Automatic Differentiation in forward mode."
        return super().__str__()[:-1] + msg
