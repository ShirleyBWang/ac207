import numpy as np
import pytest
from pomeranian.node import Node

class Test_Node:
    """Test class for Node Class"""

    def test_init(self):
        """Test of initialization for Node Class"""
        x = Node(1)
        assert x.real == 1.0
        assert x.partial_derivs == []

    def test_neg(self):
        """Test of negation special method (__neg__) for Node Class"""
        x = Node(1)
        neg_x = -x
        assert neg_x.real == -1.0
        assert neg_x.partial_derivs == []

    def test_add(self):
        """Test of addition special method (__add__) for Node Class"""
        x1 = Node(1)
        x2 = Node(5)
        x3 = x1 + x2
        assert x3.real == 6.0
        assert x3.partial_derivs == []
        assert x1.real == 1.0
        assert x1.partial_derivs[-1][0] == x3
        assert x1.partial_derivs[-1][1] == 1
        assert x2.real == 5.0
        assert x2.partial_derivs[-1][0] == x3
        assert x2.partial_derivs[-1][1] == 1

        x4 = Node(7)
        x5 = x4 + 2.5
        assert x5.real == 9.5
        assert x4.partial_derivs[-1][0] == x5
        assert x4.partial_derivs[-1][1] == 1

        with pytest.raises(TypeError):
            x1 + '1'

    def test_radd(self):
        """Test of swapped addition special method (__radd__) for Node
        Class"""
        x1 = Node(1)
        x2 = 2.5 + x1
        assert x2.real == 3.5
        assert x1.partial_derivs[-1][0] == x2
        assert x1.partial_derivs[-1][1] == 1
        
        with pytest.raises(TypeError):
            '1' + x1

    def test_sub(self):
        """Test of subtraction special method (__sub__) for Node Class"""
        x1 = Node(1)
        x2 = Node(5)
        x3 = x1 - x2
        assert x3.real == -4.0
        assert x3.partial_derivs == []
        assert x1.real == 1.0
        assert x1.partial_derivs[-1][0] == x3
        assert x1.partial_derivs[-1][1] == 1
        assert x2.real == 5.0
        assert x2.partial_derivs[-1][0] == x3
        assert x2.partial_derivs[-1][1] == -1

        x4 = Node(7)
        x5 = x4 - 2.5
        assert x5.real == 4.5
        assert x4.partial_derivs[-1][0] == x5
        assert x4.partial_derivs[-1][1] == 1

        with pytest.raises(TypeError):
            x1 - '1'

    def test_rsub(self):
        """Test of swapped subtraction special method (__rsub__) for Node
        Class"""
        x1 = Node(1)
        x2 = 2.5 - x1
        assert x2.real == 1.5
        assert x1.partial_derivs[-1][0] == x2
        assert x1.partial_derivs[-1][1] == -1
        
        with pytest.raises(TypeError):
            '1' - x1

    def test_mul(self):
        """Test of multiplication special method (__mul__) for Node
        Class"""
        x1 = Node(1)
        x2 = Node(5)
        x3 = x1 * x2
        assert x3.real == 5.0
        assert x3.partial_derivs == []
        assert x1.real == 1.0
        assert x1.partial_derivs[-1][0] == x3
        assert x1.partial_derivs[-1][1] == x2.real
        assert x2.real == 5.0
        assert x2.partial_derivs[-1][0] == x3
        assert x2.partial_derivs[-1][1] == x1.real

        x4 = Node(7)
        x5 = x4 * 2.5
        assert x5.real == 17.5
        assert x4.partial_derivs[-1][0] == x5
        assert x4.partial_derivs[-1][1] == 2.5

        with pytest.raises(TypeError):
            x1 * '1'        

    def test_rmul(self):
        """Test of swapped multiplication special method (__rmul__) for Node
        Class"""
        x1 = Node(1)
        x2 = 2.5 * x1
        assert x2.real == 2.5
        assert x1.partial_derivs[-1][0] == x2
        assert x1.partial_derivs[-1][1] == 2.5
        with pytest.raises(TypeError):
            '1' * x1

    def test_truediv(self):
        """Test of division special method (__truediv__) for Node
        Class"""
        x1 = Node(1)
        x2 = Node(5)
        x3 = x1 / x2
        assert x3.real == 0.2
        assert x3.partial_derivs == []
        assert x1.real == 1.0
        assert x1.partial_derivs[-1][0] == x3
        assert x1.partial_derivs[-1][1] == 1.0/x2.real
        assert x2.real == 5.0
        assert x2.partial_derivs[-1][0] == x3
        assert x2.partial_derivs[-1][1] == -0.04

        x4 = Node(5)
        x5 = x4 / 2.5
        assert x5.real == 2.0
        assert x4.partial_derivs[-1][0] == x5
        assert x4.partial_derivs[-1][1] == 0.4

        with pytest.raises(TypeError):
            x1 / '1'        

    def test_rtruediv(self):
        """Test of swapped division special method (__rtruediv__) for Node
        Class"""
        x1 = Node(1)
        x2 = 2.5 / x1
        assert x2.real == 2.5
        assert x1.partial_derivs[-1][0] == x2
        assert x1.partial_derivs[-1][1] == -2.5
        
        with pytest.raises(TypeError):
            '1' / x1

    def test_pow(self):
        """Test of exponentiation special method (__pow__) for Node
        Class"""
        x1 = Node(4)
        x2 = Node(2)
        x3 = x1 ** x2
        assert x3.real == 16
        assert x3.partial_derivs == []
        assert x1.real == 4.0
        assert x1.partial_derivs[-1][0] == x3
        assert x1.partial_derivs[-1][1] == 8.0
        assert x2.real == 2.0
        assert x2.partial_derivs[-1][0] == x3
        assert x2.partial_derivs[-1][1] == 16 * np.log(4)

        x4 = Node(5)
        x5 = x4 ** 2
        assert x5.real == 25.0
        assert x4.partial_derivs[-1][0] == x5
        assert x4.partial_derivs[-1][1] == 10.0

        with pytest.raises(TypeError):
            x1 ** '1'        

    def test_rpow(self):
        """Test of swapperd exponentiation special method (__rpow__) for Node
        Class"""
        
        x1 = Node(5)
        x2 = 2 ** x1
        assert x2.real == 32.0
        assert x1.partial_derivs[-1][0] == x2
        assert x1.partial_derivs[-1][1] == 32 * np.log(2)

        with pytest.raises(TypeError):
            '1' ** x1

    def test_repr(self):
        """Test repr special method (__repr__) for Node Class"""
        x1 = Node(1.0)
        output = repr(x1)
        assert output == f"Node({x1.real},[])"

    
    
    
