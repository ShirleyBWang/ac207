import numpy as np
import pytest
from pomeranian.reverse import Reverse
import pomeranian.elem_func as f

class Test_RM:
    """Test class for Reverse Mode Class"""
    def test_eval(self):
        """Test evaluation method (get_value) for Reverse Mode Class """
        
        # univariate input, one function
        f1 = lambda x: x * 1.5
        ad1 = Reverse(f1)
        assert ad1.get_value(3) == 4.5
        assert ad1.get_value([3]) == 4.5

        # input dimension not match
        with pytest.raises(Exception):
            assert ad1.get_value([3, 7]) == 4.5

        # univariate input, multiple function
        f2 = lambda x: [x * 1.5, x + 7]
        ad2 = Reverse(f2)
        assert np.array_equal(ad2.get_value(3), [4.5, 10])
        assert np.array_equal(ad2.get_value([3]), [4.5, 10])

        # multiple input, one function
        f3 = lambda x, y: x * 1.5 + x * y
        ad3 = Reverse(f3)
        assert ad3.get_value([1, 2]) == 3.5
        f4 = lambda x, y, z: x * 1.5 + x * y + y * z
        ad4 = Reverse(f4)
        assert ad4.get_value([1, 2, 3]) == 9.5

        # multiple input, multiple function
        f5 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad5 = Reverse(f5)
        assert np.array_equal(ad5.get_value([1, 2]), [3.5, 1])
        f6 = lambda x, y, z: [x * 1.5 + x * y + y * z, (x**y)**z]
        ad6 = Reverse(f6)
        assert np.array_equal(ad6.get_value([1, 2, 3]), [9.5, 1])

    def test_deriv(self):
        """Test derivative method (Reverse) for Reverse Mode Class"""
        # univariate input, one function
        f1 = lambda x: x * 1.5
        ad1 = Reverse(f1)
        assert ad1.reverse(3) == 1.5
        assert ad1.reverse([3]) == 1.5

        # univariate input, multiple function
        f2 = lambda x: [x * 1.5, x + 7]
        ad2 = Reverse(f2)
        assert np.array_equal(ad2.reverse(3), [1.5, 1])
        assert np.array_equal(ad2.reverse([3]), [1.5, 1])

        # test for functions imported from other modules 
        f2_2 = lambda x: [f.sqrt(x)+1, x**2]
        ad2_2 = Reverse(f2_2)
        assert np.array_equal(ad2_2.reverse(4), [0.25, 8])
        
        # multiple input, one function
        f3 = lambda x, y: x * 1.5 + x * y
        ad3 = Reverse(f3)
        assert np.array_equal(ad3.reverse([1, 2]), [3.5, 1])
        f4 = lambda x, y, z: x * 1.5 + x * y + y * z
        ad4 = Reverse(f4)
        assert np.array_equal(ad4.reverse([1, 2, 3]), [3.5, 4, 2])

        # multiple input, multiple function
        f5 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad5 = Reverse(f5)
        assert np.array_equal(ad5.reverse([1, 2]), [[3.5, 1], [2, 0]])
        f6 = lambda x, y, z: [x * 1.5 + x * y + y * z, z * (x**y)]
        ad6 = Reverse(f6)
        assert np.array_equal(ad6.reverse([1, 2, 3]), [[3.5, 4, 2], [6, 0, 1]])

        # raise exception
        f7 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad7 = Reverse(f7)
        with pytest.raises(Exception):
            ad7.reverse([1,2,3])
            


    def test_repr(self):
        """Test repr special method (__repr__) for Reverse Mode Class"""
        pass
    def test_str(self):
        """Test str special method (__str__) for Reverse Mode Class"""
        # not sure how to test this yet (how to test when super.__init__ is involved?)
        pass
