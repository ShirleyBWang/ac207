import numpy as np
import pytest
from pomeranian.forward import Forward


class Test_FM:
    """Test class for Forward Mode Class"""

    def test_eval(self):
        """Test evaluation method (get_value) for Forward Mode Class """
        # univariate input, one function
        f1 = lambda x: x * 1.5
        ad1 = Forward(f1)
        assert ad1.get_value(3) == 4.5
        assert ad1.get_value([3]) == 4.5

        # univariate input, multiple function
        f2 = lambda x: [x * 1.5, x + 7]
        ad2 = Forward(f2)
        assert np.array_equal(ad2.get_value(3), [4.5, 10])
        assert np.array_equal(ad2.get_value([3]), [4.5, 10])

        # multiple input, one function
        f3 = lambda x, y: x * 1.5 + x * y
        ad3 = Forward(f3)
        assert ad3.get_value([1, 2]) == 3.5
        f4 = lambda x, y, z: x * 1.5 + x * y + y * z
        ad4 = Forward(f4)
        assert ad4.get_value([1, 2, 3]) == 9.5

        # multiple input, multiple function
        f5 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad5 = Forward(f5)
        assert np.array_equal(ad5.get_value([1, 2]), [3.5, 1])
        f6 = lambda x, y, z: [x * 1.5 + x * y + y * z, (x**y)**z]
        ad6 = Forward(f6)
        assert np.array_equal(ad6.get_value([1, 2, 3]), [9.5, 1])

    def test_deriv(self):
        """Test derivative method (forward) for Forward Mode Class"""
        # univariate input, one function
        f1 = lambda x: x * 1.5
        ad1 = Forward(f1)
        assert ad1.forward(3) == 1.5
        assert ad1.forward([3]) == 1.5

        # univariate input, multiple function
        f2 = lambda x: [x * 1.5, x + 7]
        ad2 = Forward(f2)
        assert np.array_equal(ad2.forward(3), [1.5, 1])
        assert np.array_equal(ad2.forward([3]), [1.5, 1])

        # multiple input, one function
        f3 = lambda x, y: x * 1.5 + x * y
        ad3 = Forward(f3)
        assert np.array_equal(ad3.forward([1, 2]), [3.5, 1])
        f4 = lambda x, y, z: x * 1.5 + x * y + y * z
        ad4 = Forward(f4)
        assert np.array_equal(ad4.forward([1, 2, 3]), [3.5, 4, 2])

        # multiple input, multiple function
        f5 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad5 = Forward(f5)
        assert np.array_equal(ad5.forward([1, 2]), [[3.5, 1], [2, 0]])
        f6 = lambda x, y, z: [x * 1.5 + x * y + y * z, z * (x**y)]
        ad6 = Forward(f6)
        assert np.array_equal(ad6.forward([1, 2, 3]), [[3.5, 4, 2], [6, 0, 1]])

        # raise exception
        f7 = lambda x, y: [x * 1.5 + x * y, x**y]
        ad7 = Forward(f7)
        with pytest.raises(Exception):
            ad7.forward([1, 2, 3])

    def test_repr(self):
        """Test repr special method (__repr__) for Forward Mode Class"""
        pass

    def test_str(self):
        """Test str special method (__str__) for Forward Mode Class"""
        # not sure how to test this yet (how to test when super.__init__ is involved?)
        pass
