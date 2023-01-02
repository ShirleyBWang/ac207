#!/usr/bin/env python3
# File       : reverse.py
# Description: reverse: Defines a reverse mode class
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang and Alaric Wei
"""This module contains Reverse class, which inherits AutoDiff. 

The module will take in functions to perform autmatic differentiation in reverse
mode, and be able to calculate the value of the functions and the jacobian matrix
of the functions at given values.
"""

import copy
import inspect
import re
import numpy as np
from collections import defaultdict

import pomeranian.elem_func as elementary_function
from pomeranian.autodiff import AutoDiff
from pomeranian.node import Node


class Reverse(AutoDiff):
    """Reverse Mode Automatic Differentiation Class, inherit AutoDiff.

    The class defines three dunder methods: __init__, __repr__ and __str__.

    __init__: Takes the function that have to be automatically differentiated 
    via reverse mode 

    _gradiant: Compute the chained derivatives at root Node for each child Node
    and return a dictionary of gradiants with key = node, value = gradiant.

    get_value: Evaluates the function at inputs.

    reverse: Computes jacobian matrix using reverse mode AD.
    """

    def __init__(self, funcs):
        """Constructor of Reverse class.
        
        Parameters
        ----------
        funcs: callable
            Functions to perform reverse mode AD on
        """

        super().__init__(funcs)

    def _gradiant(self, root):
        """Compute the chained derivatives at root Node for each child Node
        and return a dictionary of gradiants with key = node, value = gradiant.

        Parameters
        ----------
        root : Node
            Starting Node to compute chained derivatives

        Returns
        -------
        gradiant : dictionary
            Chained gradiant at root Node for each child Node
        """

        gradiant = defaultdict(lambda: 0)  # initialize to 0

        def _calc_grad(root, st=1):
            """Inner function to update chained derivatives.

            Parameters
            ----------
            root : Node
                Starting Node to compute chained derivatives
            st : int or float, optional
                Starting gradiant/weight of root, by default 1
            """

            for child, partial_deriv in root.partial_derivs:

                # calculate the chained derivatives
                chain_grad = st * partial_deriv

                # add to child's derivatives
                gradiant[child] += chain_grad

                # recurse through all children
                _calc_grad(child, st=chain_grad)

        _calc_grad(root)

        return gradiant

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
        >>> ad = Reverse(f)
        >>> print(ad.get_value(3))
        6
        >>> print(ad.get_value([4]))
        8
        >>> f = lambda x, y: [x * 2 + y * y, x * y]
        >>> ad = Reverse(f)
        >>> print(ad.get_value([3, 4]))
        [22 12]
        """

        # convert int, float to list
        if isinstance(inputs, (int, float)):
            inputs = np.asarray([inputs])

        # n_inputs mismatch
        if len(inputs) != self.n_inputs:
            raise Exception(f"Number of inputs (={len(inputs)}) is not " +
                            "consistent with number of function " +
                            f"arguments (={self.n_inputs}).")

        # convert inputs into Node numbers
        nodes = np.asarray([Node(input) for input in inputs])

        # parse into functions
        result = self.funcs(*nodes)

        # output result conditioned on n_func
        if isinstance(result, Node):  # only one function/output
            return result.real

        else:  # multiple functions/outputs
            return np.asarray([elem.real for elem in result])

    def reverse(self, inputs):
        """Computes jacobian matrix using reverse mode AD.
                
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
        >>> ad = Reverse(f)
        >>> print(ad.reverse([4]))
        2
        >>> f = lambda x, y: x * 2 + y ** 2
        >>> ad = Reverse(f)
        >>> print(ad.reverse([3, 4]))
        [2 8]
        >>> f = lambda x, y: [x ** 2 - y ** 2, x * x]    
        >>> ad = Reverse(f)
        >>> print(ad.reverse([1, 1]))
        [[ 2 -2]
         [ 2  0]]
        """

        # convert int, float to list
        if isinstance(inputs, (int, float)):
            inputs = np.asarray([inputs])

        # n_inputs mismatch
        if len(inputs) != self.n_inputs:
            raise Exception(f"Number of inputs (={len(inputs)}) is not " +
                            "consistent with number of function " +
                            f"arguments (={self.n_inputs}).")

        # convert inputs into Node numbers
        nodes = np.asarray([Node(input) for input in inputs])

        # parse into functions
        result = self.funcs(*nodes)

        # output result conditioned on n_func, n_input
        if isinstance(result, Node):  # only one function/output

            if len(inputs) == 1:  # if univariate inputs
                return self._gradiant(nodes[0])[result]

            else:  # if multivariate inputs
                return np.asarray(
                    [self._gradiant(elem)[result] for elem in nodes])

        else:  # multiple functions/outputs

            err_msg = f"""Check the following criteria: 
            1) Please use one function that returns a list of functions to define multiple functions. 
            2) Please define the function directly in the returned list, instead of using variable names assigning each function.
            3) Please keep the input function as neat as possible without extra comments and docstrings."""

            try:

                jacob = []

                # re-initialize inputs
                n_func = len(result)
                input_nodes = [copy.deepcopy(nodes) for _ in range(n_func)]

                # re-initialize functions
                # get variable names
                var_names = [elem for elem in self.funcs.__code__.co_varnames]
                # get function source code
                funcs_str = inspect.getsource(self.funcs).strip()
                # replace tab, newline, carriage return, space with space
                funcs_str = re.sub("[\\t\\n\\r\\s]+", " ", funcs_str)
                # regex the function and split into list
                funcs_lst = re.findall(r"(\:|return)\s*\[(.*)\]", funcs_str)[-1][-1].split(",")

                # extract package name
                pkg_names = set()
                for func_str in funcs_lst:

                    # find the package name before elementary functions
                    pattern = r"([A-Za-z_]+)\.(sin|cos|tan|exp|sqrt|log|logb|arcsin|arccos|arctan|sinh|cosh|tanh|logistic)"
                    pkg_name = re.findall(pattern, func_str)

                    # if found something
                    if len(pkg_name) != 0:
                        pkg_names.update(np.asarray(pkg_name)[:, 0])

                # exclude other python packages
                exclude_pkg = set(["math", "numpy", "np"])
                pkg = pkg_names - exclude_pkg

                # assign elementary_function to initialized package name
                if len(pkg) == 0:  # if no elemnetary function, only dunder
                    pass

                elif len(pkg) == 1:  # if only one package name (correct)
                    vars()[pkg.pop()] = elementary_function

                else:  # if somehow found multiple, ask user to debug
                    raise NameError("Please only use our elementary function.")

                # calculate jacobian for each function by each set of node(s)
                for input_node, function in zip(input_nodes, funcs_lst):

                    # execute/assign variable
                    for var, val in zip(var_names, input_node):
                        vars()[var] = val

                    # evaluate function (in string)
                    res = eval(function.strip())

                    # calculate jacobian
                    jacob.append(
                        [self._gradiant(elem)[res] for elem in input_node])

                # return np.array formatted jacobian
                if len(inputs) == 1:  # if univariate inputs
                    return np.asarray(jacob).flatten()

                else:  # if multivariate inputs
                    return np.asarray(jacob)

            except:

                raise ValueError(err_msg)

    def __str__(self):
        """String representation of the class and info about functions."""
        msg = ", computing Automatic Differentiation in reverse mode."
        return super().__str__()[:-1] + msg
