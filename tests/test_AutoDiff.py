import numpy as np
from pomeranian.autodiff import AutoDiff
import pytest

class Test_AutoDiff:
    """Test class for AutoDiff Class"""
    def test_init(self):
        """Test init special method (__init__) for AutoDiff Class"""
        # multiple input function has not been implemented yet
        func1 = lambda x, y: x**y
        ad = AutoDiff(func1)
        assert ad.funcs == func1
        assert ad.n_inputs == 2
                
    def test_repr(self):
        """Test repr special method (__repr__) for AutoDiff Class"""
        func1 = lambda x, y: x**y
        ad = AutoDiff(func1)
        assert repr(ad) == f"AutoDiff class with {func1}."

    def test_str(self):
        """Test str special method (__str__) for AutoDiff Class"""
        func1 = lambda x, y: x**y
        ad = AutoDiff(func1)
        assert str(ad) == f"This is AutoDiff class with function(s) of {ad.n_inputs} inputs."
