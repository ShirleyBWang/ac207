#!/usr/bin/env python3
# File       : node.py
# Description: node: Node class for use in reverse mode AD
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang and Alaric Wei
"""Class to define Node object and and associated dunder methods for use in reverse mode AD.
"""

import numpy as np


class Node:
    """Node Class

    This class defines Node elements and all associated dunder methods. 
    Additional elementary functions for Node objects are in elem_func.py.
    Vector-valued functions are handled by reverse.py (for reverse mode).

    Methods
    -------
    __neg__()
        Negate both real and dual part of Node object; defines behavior for unary negation operator -.
    __add__(other)
        Add "other" object (Node, int, or float) to Node object; Node+other.
    __radd__(other)
        Add "other" object (Node, int, or float) to Node object; other+Node.
    __mul__(other)
        Multiply Node object with "other" object (Node, int, or float); Node*other.
    __rmul__(other)
        Multiply "other" object (Node, int, or float) with Node object; other*Node.
    __sub__(other)
        Subtract "other" object (Node, int, or float) from Node object; Node-other.
    __rsub__(other)
        Subtract Node object from "other" object (Node, int, or float); other-Node.
    __truediv__(other)
        Divide Node object by "other" object (int or float); Node/other.
    __rtruediv__(other)
        Divide "other" object (int or float) by Node object; other/Node.
    __pow__(other)
        Raise Node object to power "other" (float or int); Node**other.
    __rpow__(other)
        Raise "other" object (float or int) to power Node; other**Node.
    __repr__
        Return string constructor of Node object.

    """

    def __init__(self, real):
        """Constructor for Node class

        Parameters
        ----------
        real: float
            value of real portion of Node object
        partial_derivs: list
            list of tuples (child node,local gradient) for each child of Node

        """

        self.real = real
        self.partial_derivs = []

    def __neg__(self):
        """Operator overloading for negation operator -

        Parameters
        ----------
        none

        Returns
        -------
        Node
            Node object -1*self node
        
        Examples
        --------
        >>> -Node(5)
        Node(-5,[])

        """

        child = Node(-1 * self.real)
        self.partial_derivs.append((child, -1))
        return child

    def __add__(self, other):
        """Operator overloading of + for Node class for reverse mode AD

        Returns new Node object with value self.real + other.real.
        Appends childen with their parital derivatives for nodes being added.

        """

        if isinstance(other, Node):
            child = Node(self.real + other.real)
            self.partial_derivs.append((child, 1))
            other.partial_derivs.append((child, 1))
        elif isinstance(other, (int, float)):
            child = Node(self.real + other)
            self.partial_derivs.append((child, 1))
        else:
            raise TypeError('Input must be of type float, int, or Node')

        return child

    def __radd__(self, other):
        """Operator overloading for reverse addition for Node class
        
        Parameters
        ----------
        other: Node, int, or float

        Returns
        -------
        Node
            Node object calculated as the sum of self and other when the inputs are swapped
            other+self

        Examples
        --------
        >>> 5 + Node(5)
        Node(10,[])

        """

        if not isinstance(other, (int, float, Node)):
            raise TypeError('Input must be type float, int, or Node')
        return self.__add__(other)

    def __mul__(self, other):
        """Operator overloading of * for Node class for reverse mode AD

        Returns new Node object with value self.real*other.real.
        Appends children with their partial derivatives for nodes being multiplied.

        Parameters
        ----------
        other: Node, int, or float

        Returns
        -------
        Node
            Node object calculated as product of self and other

        Examples
        --------
        >>> Node(2)*Node(3)
        Node(6,[])
        >>> Node(2)*3
        Node(6,[])

        """

        if isinstance(other, Node):
            child = Node(self.real * other.real)
            self.partial_derivs.append((child, other.real))
            other.partial_derivs.append((child, self.real))
        elif isinstance(other, (int, float)):
            child = Node(self.real * other)
            self.partial_derivs.append((child, other))
        else:
            raise TypeError('Input must be of type float, int, or Node')

        return child

    def __rmul__(self, other):
        """Operator overloading for reverse multiplication for Node class
        
        Parameters
        ----------
        other: Node, int, or float

        Returns
        -------
        Node
            Node object product of self and other, other before self

        Examples
        --------
        >>> 3*Node(2)
        Node(6,[])

        """

        if not isinstance(other, (int, float, Node)):
            raise TypeError('Wrong input type for rmul')
        return self.__mul__(other)

    def __sub__(self, other):
        """Operator overloading to define - for Node class

        Parameters
        ----------
        other: Node, int, or float

        Returns 
        -------
        Node
            Node object difference between self and other

        Examples
        --------
        >>> Node(5) - 3
        Node(2,[])
        >>> Node(5) - Node(3)
        Node(2,[])

        """

        if isinstance(other, Node):
            child = Node(self.real - other.real)
            self.partial_derivs.append((child, 1))
            other.partial_derivs.append((child, -1))
        elif isinstance(other, (int, float)):
            child = Node(self.real - other)
            self.partial_derivs.append((child, 1))
        else:
            raise TypeError('Input must be of type float, int, or Node')

        return child

    def __rsub__(self, other):
        """Revert to __sub__ dunder method to handle input reversal of subtraction method

        Parameters
        ----------
        other: Node, int, or float

        Returns
        -------
        Node
            Node object difference between other and self

        Examples
        --------
        >>> 5 - Node(3)
        Node(2,[])

        """

        if isinstance(other, (int, float)):
            child = Node(other.real - self.real)
            self.partial_derivs.append((child, -1))
            return child
        else:
            raise TypeError('Input must be type float or int.')

    def __truediv__(self, other):
        """Operator overloading of division for Dual class

        Parameters
        ----------
        other: Node, int, or float

        Returns
        -------
        Node
            quotient self/other

        Examples
        --------
        >>> Node(6)/Node(3)
        Node(2.0,[])

        """
        if isinstance(other, Node):
            child = Node(self.real / other.real)
            self.partial_derivs.append((child, 1 / other.real))
            other.partial_derivs.append((child, -1 * self.real / other.real**2))
        elif isinstance(other, (int, float)):
            child = Node(self.real / other)
            self.partial_derivs.append((child, 1 / other))
        else:
            raise TypeError('Input must be type float or int.')
        return child

    def __rtruediv__(self, other):
        """"Operator overloading for reverse division for Node class

        Parameters
        ----------
        other: int, or float

        Returns
        -------
        Node
            quotient other/self

        Examples
        --------
        >>> 6/Node(2)
        Node(3.0,[])

        """

        if not isinstance(other, (int, float)):
            raise TypeError('Input must be float or int')
        child = Node(other / self.real)
        self.partial_derivs.append((child, -1 * other / self.real**2))

        return child

    def __pow__(self, other):
        """Operator overloading for exponential for Node class (self**other)

        Parameters
        ----------
        other: Node,int,float

        Returns
        -------
        Node
            Node object exponentiated to the power "other"

        Examples
        --------
        >>> Node(3)**2
        Node(9,[])
        >>> Node(2)**Node(2)
        Node(4,[])

        """

        if isinstance(other, Node):
            child = Node(self.real**other.real)
            self.partial_derivs.append(
                (child, (other.real) * self.real**(other.real - 1)))
            other.partial_derivs.append(
                (child, self.real**other.real * np.log(self.real)))
        elif isinstance(other, (int, float)):
            child = Node(self.real**other)
            self.partial_derivs.append((child, other * self.real**(other - 1)))
        else:
            raise TypeError('Input must be type float or int.')

        return child

    def __rpow__(self, other):
        """Operator overloading for reverse exponentiation for Nodel class (other**self)

        Parameters
        ----------
        other: int, float

        Returns
        -------
        Node
            other object exponentiated to the Node power

        Examples
        --------
        >>> 2**Node(2)
        Node(4,[])

        """

        if isinstance(other, (int, float)):
            child = Node(other**self.real)
            self.partial_derivs.append(
                (child, other**self.real * np.log(other)))
        else:
            raise TypeError('Input must be float or int')

        return child

    def __repr__(self):
        """Operator overloading for printing Node(real)

        Parameters
        ----------
        other: Node, with empty list of child partial derivatives

        Returns
        -------
        String
           Node object constructor string

        Examples
        --------
        >>> repr(Node(5.0))
        'Node(5.0,[])'

        """

        return f'Node({self.real},{self.partial_derivs})'
