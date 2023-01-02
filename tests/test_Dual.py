import numpy as np
import pytest
from pomeranian.dual import Dual

class Test_Dual:
    """Test class for Dual Number Class"""

    def test_init(self):
        """Test of initialization for Dual Number Class"""
        # test whether dual part is initialized to 1
        # Haoxue: I am currently assuming that Dual number is initialized with a float
        x = Dual(1)
        assert x.real == 1.0
        assert x.dual == 1.0
        # test initialization when dual part is given
        y = Dual(2.0,4)
        assert y.real == 2.0
        assert y.dual == 4.0
        
    def test_add(self):
        """Test of addition special method (__add__) for Dual Number Class"""
        # Dual addition
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = x1 + x2
        assert x3.real == 3.0
        assert x3.dual == 8.0

        # Integer scalar
        x4 = x1 + 4
        assert x4.real == 5.0
        assert x4.dual == 1.0

        # Float scalar
        x5 = x1 + 1.2
        assert np.isclose(x5.real, 2.2)
        assert x5.dual == 1.0

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 + '1'

    def test_radd(self):
        """Test of swapped addition special method (__radd__) for Dual Number
        Class"""

        x1 = Dual(1,1)
        # Integer scalar
        x4 = 4 + x1
        assert x4.real == 5.0
        assert x4.dual == 1.0

        # Float scalar
        x5 = 1.2 + x1
        assert np.isclose(x5.real, 2.2)
        assert x5.dual == 1.0

        with pytest.raises(TypeError):
            '1' + x1
    
    def test_sub(self):
        """Test of subtraction special method (__sub__) for Dual Number Class"""
        # Dual addition
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = x1 - x2
        assert x3.real == -1.0
        assert x3.dual == -6.0

        # Integer scalar
        x4 = x1 - 4
        assert x4.real == -3.0
        assert x4.dual == 1.0

        # Float scalar
        x5 = x1 - 1.5
        assert np.isclose(x5.real, -0.5)
        assert x5.dual == 1

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 - '1'

    def test_rsub(self):
        """Test of swapped subtraction special method (__rsub__) for Dual Number
        Class"""

        x1 = Dual(1,1)
        x2 = Dual(2, 7)
        # Between dual numbers
        x3 = x1 - x2
        assert x3.real == -1.0
        assert x3.dual == -6.0

        # Integer scalar
        x4 = 4 - x1
        assert x4.real == 3.0
        assert x4.dual == -1.0 

        # Float scalar
        x5 = 1.2 - x1
        assert np.isclose(x5.real, 0.2)
        assert x5.dual == -1.0

        # check unsupported types throw error
        with pytest.raises(TypeError):
            '1' - x1
    
    def test_mul(self):
        """Test of multiplication special method (__mul__) for Dual Number
        Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)

        # Dual multiplication
        # Haoxue: may want to double check this
        x3 = x1 * x2
        assert x3.real == 2.0
        assert x3.dual == 9.0
        
        # Integer scalar
        x4 = x1 * 3
        assert x4.real == 3.0
        assert x4.dual == 3.0
        
        # Float scalar
        x5 = x2 * 2.5
        assert x5.real == 5.0
        assert np.isclose(x5.dual, 17.5)

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 * '1'

    def test_rmul(self):
        """Test of swapped multiplication special method (__rmul__) for Dual
        Number Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)

        # Integer scalar
        x4 = 3 * x1
        assert x4.real == 3.0
        assert x4.dual == 3.0
        
        # Float scalar
        x5 = 2.5 * x2
        assert x5.real == 5.0
        assert np.isclose(x5.dual, 17.5)

        # check unsupported types throw error
        with pytest.raises(TypeError):
            '1' * x1
    
    def test_truediv(self):
        """Test of division special method (__truediv__) for Dual Number
        Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)

        # Dual multiplication
        # Haoxue: may want to double check this
        x3 = x1 / x2
        assert x3.real == 1/2.0
        assert x3.dual == -5/4.0
        
        # Integer scalar
        x4 = x1 / 3
        assert x4.real == 1/3.0
        assert x4.dual == 1/3.0
        
        # Float scalar
        x5 = x2 / 2.5
        assert np.isclose(x5.real, 2/2.5)
        assert np.isclose(x5.dual, 7/2.5)

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 / '1'

    def test_rtruediv(self):
        """Test of swapped division special method (__rtruediv__) for Dual
        Number Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)

        # Integer scalar
        x4 = 3 / x1
        assert x4.real == 3.0
        assert x4.dual == -3.0

        # Float scalar
        x5 = 2.5 / x2
        assert np.isclose(x5.real, 2.5/2)
        assert np.isclose(x5.dual, -(2.5*7)/4)

        # check unsupported types throw error
        with pytest.raises(TypeError):
            '1' / x1
    
    def test_pow(self):
        """Test of power special method (__pow__) for Dual Number
        Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        # between dual numbers
        x3 = x1 ** x2
        assert x3.real == 1.0
        assert x3.dual == 2.0

        # Integer scalar
        x4 = x1 ** 3
        assert x4.real == 1.0
        assert x4.dual == 3.0

        # Float scalar
        x5 = x2 ** 2.5
        assert np.isclose(x5.real, 2 ** 2.5)
        assert np.isclose(x5.dual, 2 ** 1.5 * 2.5 * 7)

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 ** '1'
        
    def test_rpow(self):
        """Test of swapped power special method (__rpow__) for Dual
        Number Class"""
        x1 = Dual(1,1)
        x2 = Dual(2,7)

        # Integer scalar
        x4 = 3 ** x1
        assert x4.real == 3.0
        assert np.isclose(x4.dual, 3 * np.log(3))

        # Float scalar
        x5 = 2.5 ** x2
        assert np.isclose(x5.real, 2.5 ** 2)
        assert np.isclose(x5.dual, 2.5 ** 2 * np.log(2.5) * 7)

        with pytest.raises(TypeError):
            '1' ** x1

    def test_eq(self):
        """Test of equal special method (__eq__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        x4 = 1.0
        assert (x1 == x2) == False
        assert (x2 == x3) == True
        assert (x1 == x4) == False

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 == '1'
        

    def test_lt(self):
        """Test of less than special method (__lt__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        assert (x1 < x2) == True
        assert (x2 < x1) == False
        assert (x1 < 1.1) == True
        assert (x1 < 0.9) == False
        assert (x2 < x3) == False

        with pytest.raises(TypeError):
            x1 < '1'

    def test_gt(self):
        """Test of greater than special method (__gt__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        assert (x1 > x2) == False
        assert (x2 > x1) == True
        assert (x1 > 1.1) == False
        assert (x1 > 0.9) == True
        assert (x2 > x3) == False

        with pytest.raises(TypeError):
            x1 > '1'

    def test_le(self):
        """Test of less or equal than special method (__le__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        assert (x1 <= x2) == True
        assert (x2 <= x1) == False
        assert (x1 <= 1.1) == True
        assert (x1 <= 0.9) == False
        assert (x2 <= x3) == True

        with pytest.raises(TypeError):
            x1 <= '1'

    def test_ge(self):
        """Test of greater or equal than special method (__ge__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        assert (x1 >= x2) == False
        assert (x2 >= x1) == True
        assert (x1 >= 1.1) == False
        assert (x1 >= 0.9) == True
        assert (x2 >= x3) == True

        with pytest.raises(TypeError):
            x1 >= '1'

    def test_ne(self):
        """Test of not equal special method (__ne__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(1,1)
        x2 = Dual(2,7)
        x3 = Dual(2,7)
        x4 = 1.0
        assert (x1 != x2) == True
        assert (x2 != x3) == False
        assert (x1 != x4) == True

        # check unsupported types throw error
        with pytest.raises(TypeError):
            x1 != '1'
        

    
    def test_neg(self):
        """Test of neg special method (__neg__) for Dual Number Class"""
        # between dual numbers
        x1 = Dual(2,7)
        x2 = -x1
        assert x2.real == -2
        assert x2.dual == -7

    def test_repr(self):
        """Test repr special method (__repr__) for Dual Number Class"""
        x1 = Dual(2,7)

    def test_str(self):
        """Test str special method (__str__) for Dual Number Class"""
        x1 = Dual(2,7)
        assert repr(x1) == 'Dual(2,7)'

    