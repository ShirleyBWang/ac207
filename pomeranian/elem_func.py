#!/usr/bin/env python3
# File       : elem_func.py
# Description: elementary functions - function overloading for Node and Dual classes
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang, and Alaric Wei
"""Modules containing additional elementary functions for Dual and Node class.
"""

import numpy as np
from pomeranian.dual import Dual
from pomeranian.node import Node


def sin(z):
    """Operator overloading to calcualte sine of the input

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual 
        if input is Dual
    float
        if input is int or float

    Examples
    --------
    >>> sin(Dual(np.pi/2,1))
    Dual(1.0,6.123233995736766e-17)
    >>> z = Node(0)
    >>> sin(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.sin(z.real)
        dual_part = z.dual * np.cos(z.real)
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.sin(z.real))
        z.partial_derivs.append((child, np.cos(z.real)))
        return child
    elif isinstance(z, (int, float)):
        return np.sin(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def cos(z):
    """Operator overloading to calculate the cosine of the input

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        cosine of dual number
    float
        if input is int or float

    Examples
    --------
    >>> cos(Dual(np.pi/2,1))
    Dual(6.123233995736766e-17,-1.0)
    >>> z = Node(0)
    >>> cos(z)
    Node(1.0,[])
    >>> z.partial_derivs
    [(Node(1.0,[]), -0.0)]
    """

    if isinstance(z, Dual):
        real_part = np.cos(z.real)
        dual_part = -np.sin(z.real) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.cos(z.real))
        z.partial_derivs.append((child, -np.sin(z.real)))
        return child
    elif isinstance(z, (int, float)):
        return np.cos(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def tan(z):
    """Operator overloading to calculate tangent of input

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        sine of dual number
    float
        if input is int or float

    Examples
    --------
    >>> tan(Dual(0))
    Dual(0.0,1.0)
    >>> z = Node(0)
    >>> tan(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.tan(z.real)
        dual_part = 1. / (np.cos(z.real)**2) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.tan(z.real))
        z.partial_derivs.append((child, 1. / (np.cos(z.real)**2)))
        return child
    elif isinstance(z, (int, float)):
        return np.tan(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def exp(z):
    """Operator overloading to calcualte e raised to the power of the input

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        exp(Dual) if input is a Dual
    float
        exp(z) if input z is int or float

    Examples
    --------
    >>> exp(Dual(0))
    Dual(1.0,1.0)
    >>> z = Node(0)
    >>> exp(z)
    Node(1.0,[])
    >>> z.partial_derivs
    [(Node(1.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.exp(z.real)
        dual_part = np.exp(z.real) * z.dual
        return Dual(real_part, dual_part)
    if isinstance(z, Node):
        child = Node(np.exp(z.real))
        z.partial_derivs.append((child, np.exp(z.real)))
        return child
    elif isinstance(z, (int, float)):
        return np.exp(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def sqrt(z):
    """Operator overloading for square root

    Parameters
    ----------
    z: Dual, int, or float
    
    Returns
    -------
    Dual
        sqrt(Dual) if input z is a Dual
    float
        sqrt(z) if input z is int or float

    Examples
    --------
    >>> sqrt(Dual(4))
    Dual(2.0,0.25)
    >>> z = Node(4)
    >>> sqrt(z)
    Node(2.0,[])
    >>> z.partial_derivs
    [(Node(2.0,[]), 0.25)]
    """

    if isinstance(z, Dual):
        real_part = np.sqrt(float(z.real))
        dual_part = 0.5 / np.sqrt(float(z.real)) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.sqrt(z.real))
        z.partial_derivs.append((child, .5 * (z.real**-.5)))
        return child
    elif isinstance(z, (int, float)):
        return np.sqrt(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def log(z):
    """Operator overloading for logarithm base e

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
       Dual number representation of log(z) if z is type Dual
    float
        log(z) if input z is int or float

    Examples
    --------
    >>> log(Dual(1))
    Dual(0.0,1.0)
    >>> z = Node(1)
    >>> log(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.log(z.real)
        dual_part = 1 / z.real * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.log(z.real))
        z.partial_derivs.append((child, 1. / z.real))
        return child
    elif isinstance(z, (int, float)):
        return np.log(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def logb(z, base):
    """Operator overloading for logarithem with base specfied by input parameter b

    Parameters
    ----------
    z: Dual, int, or float
    base: int or float

    Returns
    -------
    Dual
        log base b of Dual input
    float
        log base b of int or float input

    Examples
    --------
    >>> logb(Dual(2),2)
    Dual(1.0,0.7213475204444817)
    >>> z = Node(2)
    >>> logb(z,2)
    Node(1.0,[])
    >>> z.partial_derivs
    [(Node(1.0,[]), 0.7213475204444817)]
    """

    if isinstance(z, Dual):
        real_part = np.log(z.real) / np.log(base)
        dual_part = (1 / (z.real * np.log(base))) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.log(z.real) / np.log(base))
        z.partial_derivs.append((child, 1. / (z.real * np.log(base))))
        return child
    elif isinstance(z, (int, float)):
        return np.log(z) / np.log(base)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def arcsin(z):
    """Operator overloading for arcsine, inverse sine of input z.

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        arcsin(z) if input z if type Dual
    float
        arcsin(z) is float if input z is int or float

    Examples
    --------
    >>> arcsin(Dual(0))
    Dual(0.0,1.0)
    >>> z = Node(0)
    >>> arcsin(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.arcsin(z.real)
        dual_part = 1. / np.sqrt(1 - z.real**2) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.arcsin(z.real))
        z.partial_derivs.append((child, 1. / np.sqrt(1 - z.real**2)))
        return child
    elif isinstance(z, (int, float)):
        return np.arcsin(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def arccos(z):
    """Operator overloading for arccosine, inverse cosine of input z.

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        arccos(z) if input z is type Dual
    float
        arccos(z) is float if input z is int or float

    Examples
    --------
    >>> arccos(Dual(0))
    Dual(1.5707963267948966,-1.0)
    >>> z = Node(0)
    >>> arccos(z)
    Node(1.5707963267948966,[])
    >>> z.partial_derivs
    [(Node(1.5707963267948966,[]), -1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.arccos(z.real)
        dual_part = -1. / np.sqrt(1 - z.real**2) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.arccos(z.real))
        z.partial_derivs.append((child, -1. / np.sqrt(1 - z.real**2)))
        return child
    elif isinstance(z, (int, float)):
        return np.arccos(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def arctan(z):
    """Operator overloading for arctangent, inverse tangent of input z.

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        arctan(z) if z is type Dual
    float
        arctan(z) if z is int or float

    Examples
    --------
    >>> arctan(Dual(0))
    Dual(0.0,1.0)
    >>> z = Node(0)
    >>> arctan(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.arctan(z.real)
        dual_part = 1. / (1 + z.real**2) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.arctan(z.real))
        z.partial_derivs.append((child, 1. / (1 + z.real**2)))
        return child
    elif isinstance(z, (int, float)):
        return np.arctan(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def sinh(z):
    """Operator overloading for sinh, hyperbolic sine of input z

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        sinh(z) if z is type Dual
    float
        sinh(z) if z is int or float

    Examples
    --------
    >>> sinh(Dual(0))
    Dual(0.0,1.0)
    >>> z = Node(0)
    >>> sinh(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.sinh(z.real)
        dual_part = np.cosh(z.real) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.sinh(z.real))
        z.partial_derivs.append((child, np.cosh(z.real)))
        return child
    elif isinstance(z, (int, float)):
        return np.sinh(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def cosh(z):
    """Operator overloading for cosh(dual), hyperbolic cosine of input z

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        cosh(z) if z is type Dual
    float
        cosh(z) if z is int or float

    Examples
    --------
    >>> cosh(Dual(0))
    Dual(1.0,0.0)
    >>> z = Node(0)
    >>> cosh(z)
    Node(1.0,[])
    >>> z.partial_derivs
    [(Node(1.0,[]), 0.0)]
    """

    if isinstance(z, Dual):
        real_part = np.cosh(z.real)
        dual_part = np.sinh(z.real) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.cosh(z.real))
        z.partial_derivs.append((child, np.sinh(z.real)))
        return child
    elif isinstance(z, (int, float)):
        return np.cosh(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def tanh(z):
    """Operator overloading for tanh(dual), hyperbolic tangent of input z.

    https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        tanh(z) if z is type Dual
    float
        tanh(z) if z is int or float

    Examples
    --------
    >>> tanh(Dual(0))
    Dual(0.0,1.0)
    >>> z = Node(0)
    >>> tanh(z)
    Node(0.0,[])
    >>> z.partial_derivs
    [(Node(0.0,[]), 1.0)]
    """

    if isinstance(z, Dual):
        real_part = np.tanh(z.real)
        dual_part = z.dual * (1 - np.tanh(z.real)**2)
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(np.tanh(z.real))
        z.partial_derivs.append((child, 1 - np.tanh(z.real)**2))
        return child
    elif isinstance(z, (int, float)):
        return np.tanh(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')


def logistic(z):
    """Operator overloading for logistic function f(z) = 1/(1+e**-z)

    Parameters
    ----------
    z: Dual, int, or float

    Returns
    -------
    Dual
        logistic(z) if z is type Dual
    float 
        logisitic(z) if z is int or float

    Examples
    --------
    >>> logistic(Dual(0))
    Dual(0.5,0.25)
    >>> z = Node(0)
    >>> logistic(z)
    Node(0.5,[])
    >>> z.partial_derivs
    [(Node(0.5,[]), 0.25)]
    """

    f = lambda x: 1. / (1 + np.exp(-x))

    if isinstance(z, Dual):
        real_part = f(z.real)
        dual_part = f(z.real) * (1 - f(z.real)) * z.dual
        return Dual(real_part, dual_part)
    elif isinstance(z, Node):
        child = Node(f(z.real))
        z.partial_derivs.append((child, f(z.real) * (1 - f(z.real))))
        return child
    elif isinstance(z, (int, float)):
        return f(z)
    else:
        raise TypeError('Input must be type float, int, Dual, or Node')
