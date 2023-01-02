#!/usr/bin/env python3
# File       : autodiff.py
# Description: autodiff: Automatic Differentiation Class
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang and Alaric Wei
"""This module contains AutoDiff base class, which is the parent of Forward and Reverse. 

The module will instantiate a base class for Forward and Reverse, and implement the 
__init__, __repr__, and __str__.
"""


class AutoDiff:
    """Automatic Differentiation Class

    The class defines three dunder methods: __init__, __repr__ and __str__.

    __init__: Takes the function that has to be automatically differentiated and
    initializes it. Also initializes the number of arguments provided. 

    __repr__: Returns string with the instance of the class as well as the 
    provided functions.

    __str__: Returns string representation of the class and info about functions.
    """

    def __init__(self, funcs):
        """Constructor of AutoDiff class.
        
        Parameters
        ----------
        funcs: callable
            Functions to perform AD on
        """

        self.funcs = funcs
        self.n_inputs = self.funcs.__code__.co_argcount

    def __repr__(self):
        """String with the instance of the class as well as the provided functions."""
        class_name = type(self).__name__
        return f"{class_name} class with {self.funcs}."

    def __str__(self):
        """String representation of the class and info about functions."""
        class_name = type(self).__name__
        return f"This is {class_name} class with function(s) of {self.n_inputs} inputs."
