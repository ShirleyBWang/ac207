#!/usr/bin/env python3
# File       : dual.py
# Description: dual: Dual number class for use in forward mode AD
# Copyright 2022 Haoxue Fan, Sarah Rathnam, Mahnum Shahzad, Shirley Wang and Alaric Wei
"""Class to define Dual number object and and associated dunder methods for use in forward mode AD.
"""

import numpy as np


class Dual:
    """Dual Number Class

    This class defines dual numbers and all associated dunder methods.
    Additional elementary methods for dual numbers are defined in elem_func.py.
    Vector-valued functions are handled by forward.py (for forward mode).   

    Methods
    -------
    __eq__(other)
        Test for equality between Dual object and "other" object; Dual==other.
    __ne__(other)
        Test for inequality between Dual object and "other" object; Dual!=other.
    __neg__()
        Negate both real and dual part of Dual object; defines behavior for unary negation operator -.
    __add__(other)
        Add "other" object (Dual, int, or float) to Dual object; Dual+other.
    __radd__(other)
        Add Dual object to "other" object (Dual, int, or float); other+Dual.
    __mul__(other)
        Multiply Dual object with "other" object (Dual, int, or float); Dual*other.
    __rmul__(other)
        Multiply "other" object (Dual, int, or float) with Dual object; other*Dual.
    __sub__(other)
        Subtract "other" object (Dual, int, or float) from Dual object; Dual-other.
    __rsub__(other)
        Subtract Dual object from "other" object (Dual, int, or float); other-Dual.
    __truediv__(other)
        Divide Dual object by "other" object (int or float); Dual/other.
    __rtruediv__(other)
        Divide "other" object (int or float) by Dual object; other/Dual.
    __pow__(other)
        Raise Dual object to power "other" (float or int); Dual**other.
    __rpow__(other)
        Raise "other" object (float or int) to power Dual; other**Dual.
    __repr__
        Return object representation of Dual object.

    """

    def __init__(self, real, dual=1):
        """Constructor for Dual class

        Parameters
        ----------
        real: float
              value of real portion of dual number
        dual: optional, default ot 1 if not specified
              value of dual part of dual number, to calculate the derivative
       
        """

        self.real = real
        self.dual = dual

    def __eq__(self, other):
        """ Operator overoading to define == for Dual class
    
        Parameters
        ----------
        other: Dual,int, or float

        Returns
        -------
        bool
            Boolean expression, True if the real and dual attributes of self and other are the same
            Otherwise return False

        Examples
        --------
        >>> Dual(1,1)==Dual(1,2)
        False
        >>> Dual(2,1) == Dual(2)
        True

        """

        if isinstance(other, (int, float)):
            return False
        elif isinstance(other, Dual):
            if (self.real == other.real and self.dual == other.dual):
                return True
            else:
                return False
        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __ne__(self, other):
        """ Operator overoading to define != for Dual class

        Parameters
        ----------
        other: Dual,int, or float

        Returns
        -------
        bool
            Boolean expression, False if the real and dual attributes of self and other are the same
            Otherwise return True

        Examples
        --------
        >>> Dual(1,2) != Dual(1.0,2.0)
        False

        """

        if isinstance(other, (int, float)):
            return True
        elif isinstance(other, Dual):
            if (self.real != other.real or self.dual != other.dual):
                return True
            else:
                return False
        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __gt__(self, other):
        """Operator overloading to define > for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        bool
            Boolean expression, true if self.val > passed comparison value
            Otherwise return False

        Examples
        --------
        >>> Dual(1.0,2.0) > 0
        True
        >>> Dual(1.0,2.0) > Dual(3.0,4.0)
        False

        """

        if isinstance(other, (int, float)):
            if self.real > other:
                return True
            else:
                return False
        elif isinstance(other, Dual):
            if self.real > other.real:
                return True
            else:
                return False
        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __lt__(self, other):
        """Operator overloading to define < for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        bool
            Boolean expression, true if self.val < passed comparison value
            Otherwise return False

        Examples
        --------
        >>> Dual(1.0,2.0) < 0
        False
        >>> Dual(1.0,2.0) < Dual(3.0,4.0)
        True

        """

        if isinstance(other, (int, float)):
            if self.real < other:
                return True
            else:
                return False
        elif isinstance(other, Dual):
            if self.real < other.real:
                return True
            else:
                return False
        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __le__(self, other):
        """Operator overloading to define <= for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        bool
            Boolean expression, true if self.val <= passed comparison value
            Otherwise return False

        Examples
        ----------
        >>> Dual(1.0,2.0) <= 1.0
        True
        >>> Dual(1.0,2.0) <= 2.0
        True

        """

        if isinstance(other, (int, float)):
            if self.real <= other:
                return True
            else:
                return False
        elif isinstance(other, Dual):
            if self.real <= other.real:
                return True
            else:
                return False
        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __ge__(self, other):
        """Operator overloading to define >= for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        bool
            Boolean expression, true if self.val <= passed comparison value
            Otherwise return False

        Examples
        ----------
        >>> Dual(1.0,2.0) >= 1.0
        True
        >>> Dual(1.0,2.0) >= Dual(2.0,1.0)
        False

        """

        if isinstance(other, (int, float)):
            if self.real >= other:
                return True
            else:
                return False
        elif isinstance(other, Dual):
            if self.real >= other.real:
                return True
            else:
                return False

        else:
            raise TypeError('Input must be of type float, int, or Dual')

    def __neg__(self):
        """Operator overloading for negation for Dual class

        Parameters 
        ----------
        None (unary operator)

        Returns 
        -------
        Dual
            Dual number object, negation of both real and dual parts of input

        Examples
        --------
        >>> -Dual(1.0,2.0)
        Dual(-1.0,-2.0)

        """

        return Dual(-self.real, -self.dual)

    def __add__(self, other):
        """Operator overloading to define + for Dual class 

        Parameters
        ----------
        other: Dual,int, or float

        Returns
        -------
        Dual
            Dual object calculated as the sum of self and other

        Examples
        --------
        >>> Dual(1,2) + 5
        Dual(6,2)
        >>> Dual(1,2) + Dual(3,4)
        Dual(4,6)

        """

        if isinstance(other, (int, float)):
            real_part = self.real + other
            dual_part = self.dual
        elif isinstance(other, Dual):
            real_part = self.real + other.real
            dual_part = self.dual + other.dual
        else:
            raise TypeError('Input must be type float, int, or Dual')

        return Dual(real_part, dual_part)

    def __radd__(self, other):
        """Operator overloading for reverse addition for Dual class

        Parameters
        ----------
        other: Dual,int, or float

        Returns
        -------
        Dual
            Dual object calculated as the sum of self and other when the inputs are swapped
            other + self

        Examples
        --------
        >>> 5 + Dual(1,2)
        Dual(6,2)

        """

        if not isinstance(other, (int, float, Dual)):
            raise TypeError('Input must be type float, int, or Dual')
        return self.__add__(other)

    def __mul__(self, other):
        """Operator overloading for multiplication for Dual class
        
        Parameters
        ----------
        other: Dual,int, or float

        Returns
        -------
        Dual
            Dual object product of self and other

        Examples
        --------
        >>> Dual(1,2) * Dual(3,4)
        Dual(3,10)

        """

        if isinstance(other, (int, float)):
            real_part = self.real * other
            dual_part = self.dual * other
        elif isinstance(other, Dual):
            real_part = self.real * other.real
            dual_part = self.real * other.dual + self.dual * other.real
        else:
            raise TypeError('Input must be type float, int, or Dual')

        return Dual(real_part, dual_part)

    def __rmul__(self, other):
        """Operator overloading for reverse multiplication for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual object product of self and other when order of inputs is swapped
        other * self

        Examples
        --------
        >>> 2*Dual(1,2)
        Dual(2,4)

        """

        if not isinstance(other, (int, float, Dual)):
            raise TypeError('Wrong input type for rmul')

        return self.__mul__(other)

    def __sub__(self, other):
        """Operator overloading to define - for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual
            Dual object calculated as the the difference self - other

        Examples
        --------
        >>> Dual(1,2) - Dual(3,4)
        Dual(-2,-2)
        >>> Dual(1,2) - 1
        Dual(0,2)

        """

        if isinstance(other, (int, float)):
            real_part = self.real - other
            dual_part = self.dual
        elif isinstance(other, Dual):
            real_part = self.real - other.real
            dual_part = self.dual - other.dual
        else:
            raise TypeError('Input must be type float, int, or Dual')

        return Dual(real_part, dual_part)

    def __rsub__(self, other):
        """Operator overloading for reverse subtraction for Dual class

        Parameters
        ----------
        other: int, or float

        Returns
        -------
        Dual
            Dual object calculated as difference other - self

        Examples
        --------
        >>> 1 - Dual(1,2) 
        Dual(0,-2)
        
        """

        if isinstance(other, (int, float)):
            real_part = other - self.real
            dual_part = -self.dual
        else:
            raise TypeError('Input must be type float or int.')

        return Dual(real_part, dual_part)

    def __truediv__(self, other):
        """Operator overloading for division for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual
            Dual division of self / other 

        Examples
        --------
        >>> Dual(3,4)/Dual(1,2)
        Dual(3.0,-2.0)

        """

        if isinstance(other, (int, float)):
            real_part = (self.real * other) / other**2
            dual_part = (self.dual * other) / other**2
        elif isinstance(other, Dual):
            real_part = self.real / other.real
            dual_part = ((self.dual * other.real) -
                         (self.real * other.dual)) / other.real**2
        else:
            raise TypeError('Input must be type float, int, or Dual')

        return Dual(real_part, dual_part)

    def __rtruediv__(self, other):
        """Operator overloading for reverse division for Dual class

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual
            Dual division of other / self 
        
        Examples
        --------
        >>> 4/Dual(2,4)
        Dual(2.0,-4.0)

        """

        if not isinstance(other, (int, float)):
            raise TypeError('Input must be type float or int')

        real_part = other / self.real
        dual_part = -(other / self.real**2) * self.dual
        return Dual(real_part, dual_part)

    def __pow__(self, other):
        """Operator overloading for exponential for dual class (dual**other)

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual
        Dual object exponentiated by other

        Examples
        --------
        >>> Dual(2,4)**2
        Dual(4,16)
        >>> Dual(1,2)**Dual(3,4)
        Dual(1,6.0)

        """

        if isinstance(other, (int, float)):
            real_part = self.real**other
            dual_part = self.real**(other - 1) * other * self.dual
        elif isinstance(other, Dual):
            real_part = self.real**other.real
            temp1 = np.log(self.real) * other.dual
            temp2 = self.dual * other.real / self.real

            dual_part = self.real**(other.real) * (temp1 + temp2)
        else:
            raise TypeError('Input must be type float, int, or Dual')

        return Dual(real_part, dual_part)

    def __rpow__(self, other):
        """Operator overloading for reverse exponentiation for Dual class (other**dual)

        Parameters
        ----------
        other: Dual, int, or float

        Returns
        -------
        Dual
            Other exponentiated by dual object

        Examples
        --------
        >>> 2**Dual(2,1)
        Dual(4,2.772588722239781)

        """

        if not isinstance(other, (int, float)):
            raise TypeError('Input must be type float or int')
        real_part = other**self.real
        dual_part = np.log(other) * other**self.real * self.dual
        return Dual(real_part, dual_part)

    def __repr__(self):
        """prints class definition to construct Dual object. Can evaluate with eval().


        Returns
        -------
        string
            Dual object contructor string

        Examples
        --------
        >>> repr(Dual(1,2))
        'Dual(1,2)'

        """

        return f'Dual({self.real},{self.dual})'
