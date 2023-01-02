import numpy as np
import pytest
from pomeranian.dual import Dual
from pomeranian.elem_func import *

class Test_Elem:
    """Test class for Elementary Functions"""

    def test_sin(self):
        """Test of sin method (sin) """
        # for dual number
        x1 = Dual(2,7)
        x2 = sin(x1)
        assert np.isclose(x2.real, np.sin(2))
        assert np.isclose(x2.dual, 7 * np.cos(2))
        
        # for not dual number
        x3 = 10.0
        assert np.isclose(sin(x3), np.sin(10))

        #for node
        x4=Node(2)
        x2 = sin(x4)
        assert np.isclose(x2.real, np.sin(2))
        assert x4.partial_derivs[-1][1] == np.cos(2)

        x5="a"
        with pytest.raises(TypeError):
            sin(x5)


    def test_cos(self):
        """Test of cos method (cos) """
        # between dual numbers
        x1 = Dual(2,7)
        x2 = cos(x1)
        assert np.isclose(x2.real, np.cos(2))
        assert np.isclose(x2.dual, -7 * np.sin(2))

        # for not dual number
        x3 = 10.0
        assert np.isclose(cos(x3), np.cos(10))

        #for node
        x4=Node(2)
        x2 = cos(x4)
        assert np.isclose(x2.real, np.cos(2))
        assert x4.partial_derivs[-1][1] == -1*np.sin(2)

        x5="a"
        with pytest.raises(TypeError):
            cos(x5)

    def test_tan(self):
        """Test of tan method (tan) """
        # between dual numbers
        x1 = Dual(2,7)
        x2 = tan(x1)
        assert np.isclose(x2.real, np.tan(2))
        assert np.isclose(x2.dual, 7/(np.cos(2) ** 2))

        # for not dual number
        x3 = 10.0
        assert np.isclose(tan(x3), np.tan(10))

        #for node
        x4=Node(2)
        x2 = tan(x4)
        assert np.isclose(x2.real, np.tan(2))
        assert x4.partial_derivs[-1][1] == 1/(np.cos(2) ** 2)

        x5="a"
        with pytest.raises(TypeError):
            tan(x5)

    def test_exp(self):
        """Test of exponentiation method (exp) """
        # between dual numbers
        x1 = Dual(2,7)
        x2 = exp(x1)
        assert np.isclose(x2.real, np.exp(2))
        assert np.isclose(x2.dual, 7 * np.exp(2))

        # for not dual number
        x3 = 10.0
        assert np.isclose(exp(x3), np.exp(10))

        #for node
        x4=Node(2)
        x2 = exp(x4)
        assert np.isclose(x2.real, np.exp(2))
        assert x4.partial_derivs[-1][1] == np.exp(2)

        x5="a"
        with pytest.raises(TypeError):
            exp(x5)

    def test_sqrt(self):
        """Test of square root method (sqrt) """
        # between dual numbers
        x1 = Dual(2,7)
        x2 = sqrt(x1)
        assert np.isclose(x2.real, np.sqrt(2))
        assert np.isclose(x2.dual, 3.5 / np.sqrt(2))

        # scalar
        x3 = 5
        assert np.isclose(sqrt(x3), np.sqrt(5))

        #for node
        x4=Node(2)
        x2 = sqrt(x4)
        assert np.isclose(x2.real, np.sqrt(2))
        assert np.isclose(x4.partial_derivs[-1][1],  1/(2*(np.sqrt(2))))

        x5="a"
        with pytest.raises(TypeError):
            sqrt(x5)
    
    def test_log(self):
        """Test of log method (log) """
        # between dual numbers
        x1 = Dual(2,7)
        x2 = log(x1)
        assert np.isclose(x2.real, np.log(2))
        assert x2.dual == 7 / 2

        # scalar
        x3 = 5
        assert np.isclose(log(x3), np.log(5))

        #for node
        x4=Node(2)
        x2 = log(x4)
        assert np.isclose(x2.real, np.log(2))
        assert x4.partial_derivs[-1][1] == 1/2

        x5="a"
        with pytest.raises(TypeError):
            log(x5)
    
    def test_logb(self):
        """Test of log method with customized base(logb) """
        x1 = Dual(2,7)
        x2 = logb(x1, 4)
        assert np.isclose(x2.real, np.log(2) / np.log(4))
        assert np.isclose(x2.dual, 3.5 / np.log(4))

        # scalar
        x3 = 5
        assert np.isclose(logb(x3, 4), np.log(5) / np.log(4))

        #for node
        x4=Node(2)
        x2 = logb(x4, 4)
        assert np.isclose(x2.real, np.log(2)/np.log(4))
        assert x4.partial_derivs[-1][1] == 1 / (2*np.log(4))

        x5="a"
        with pytest.raises(TypeError):
            logb(x5, 4)
        
    def test_arcsin(self):
        """Test of arcsin method (arcsin)"""
        x1 = Dual(0.25,0.5)
        x2 = arcsin(x1)
        assert np.isclose(x2.real, np.arcsin(0.25))
        assert np.isclose(x2.dual, 0.5/np.sqrt(1-(0.25**2)))

        # scalar
        x3=0.3
        assert np.isclose(arcsin(x3), np.arcsin(0.3))

        #for node
        x4=Node(0.3)
        x2 = arcsin(x4)
        assert np.isclose(x2.real, np.arcsin(0.3))
        assert np.isclose(x4.partial_derivs[-1][1], 1/np.sqrt(1-(0.3**2)))

        x5="a"
        with pytest.raises(TypeError):
            arcsin(x5)



    def test_arccos(self):
        """Test of arcsin method (arccos)"""
        x1 = Dual(0.25,0.5)
        x2 = arccos(x1)
        assert np.isclose(x2.real, np.arccos(0.25))
        assert np.isclose(x2.dual, -0.5/np.sqrt(1.0-0.25**2))

        # scalar
        x3=0.3
        assert np.isclose(arccos(x3), np.arccos(0.3))

        #for node
        x4=Node(0.3)
        x2 = arccos(x4)
        assert np.isclose(x2.real, np.arccos(0.3))
        assert np.isclose(x4.partial_derivs[-1][1], -1/np.sqrt(1.0-0.3**2))

        x5="a"
        with pytest.raises(TypeError):
            arccos(x5)


    def test_arctan(self):
        """Test of arctan method (arcsinarctan)"""
        x1 = Dual(0.25,0.5)
        x2 = arctan(x1)
        assert np.isclose(x2.real, np.arctan(0.25))
        assert np.isclose(x2.dual, 0.5/(1+0.25**2))

        # scalar
        x3=5
        assert np.isclose(arctan(x3), np.arctan(5))

        #for node
        x4=Node(0.25)
        x2 = arctan(x4)
        assert np.isclose(x2.real, np.arctan(0.25))
        assert np.isclose(x4.partial_derivs[-1][1], 1/(1+0.25**2))

        x5="a"
        with pytest.raises(TypeError):
            arctan(x5)


    def test_sinh(self):
        """Test of sinh method (sinh)"""
        x1 = Dual(2,7)
        x2 = sinh(x1)
        assert np.isclose(x2.real, np.sinh(2))
        assert np.isclose(x2.dual, 7*np.cosh(2))
        # for not dual number
        x3 = 10.0
        assert np.isclose(sinh(x3), np.sinh(10.0))

        #for node
        x4=Node(2)
        x2 = sinh(x4)
        assert np.isclose(x2.real, np.sinh(2))
        assert np.isclose(x4.partial_derivs[-1][1], np.cosh(2))

        x5="a"
        with pytest.raises(TypeError):
            sinh(x5)


    def test_cosh(self):
        """Test of cosh method (cosh)"""
        x1 = Dual(2,7)
        x2 = cosh(x1)
        assert np.isclose(x2.real, np.cosh(2))
        assert np.isclose(x2.dual, 7 * np.sinh(2))

        # for not dual number
        x3 = 10.0
        assert np.isclose(cosh(x3), np.cosh(10))

        #for node
        x4=Node(2)
        x2 = cosh(x4)
        assert np.isclose(x2.real, np.cosh(2))
        assert np.isclose(x4.partial_derivs[-1][1], np.sinh(2))

        x5="a"
        with pytest.raises(TypeError):
            cosh(x5)


    def test_tanh(self):
        """Test of tanh method (tanh)"""
        x1 = Dual(2,7)
        x2 = tanh(x1)
        assert np.isclose(x2.real, np.tanh(2))
        assert np.isclose(x2.dual, 7*(1-(np.tanh(2))**2))

        # for not dual number
        x3 = 10.0
        assert np.isclose(tanh(x3), np.tanh(10.0))

        #for node
        x4=Node(2)
        x2 = tanh(x4)
        assert np.isclose(x2.real, np.tanh(2))
        assert np.isclose(x4.partial_derivs[-1][1], (1-(np.tanh(2))**2))

        x5="a"
        with pytest.raises(TypeError):
            tanh(x5)

    def test_logistic(self):
        """Test of logistic method (logistic)"""
        x1 = Dual(2,7)
        x2 = logistic(x1)
        assert np.isclose(x2.real, 1.0/(1+np.exp(-x1.real)))
        assert np.isclose(x2.dual, 7*(np.exp(-2)/(np.exp(-2)+1)**2))

        # scalar
        x3=5
        assert np.isclose(logistic(x3), 1.0/(1+np.exp(-5)))

        #for node
        x4=Node(2)
        x2 = logistic(x4)
        assert np.isclose(x2.real, 1.0/(1+np.exp(-x1.real)))
        assert np.isclose(x4.partial_derivs[-1][1], (np.exp(-2)/(np.exp(-2)+1)**2))

        x5="a"
        with pytest.raises(TypeError):
            logistic(x5)
