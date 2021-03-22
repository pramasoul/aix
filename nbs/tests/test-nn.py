import unittest
import numpy as np
import sympy

import lib.nn as nn

def arangep(n, starting_index=0):
    sympy.sieve.extend_to_no(starting_index + n)
    return np.array(sympy.sieve._list[starting_index:starting_index + n])


class TestNN(unittest.TestCase):

    def testIdentityLayer(self):
        iL = nn.IdentityLayer()
        self.assertTrue(all(iL(np.arange(5)) == np.arange(5)))
        self.assertTrue(all(iL.backprop(np.arange(7)) == np.arange(7)))

    def testMapLayer(self):
        mL = nn.MapLayer(lambda x:x**2, lambda d:2*d)
        self.assertTrue(all(mL(np.array([7,3,-11])) == np.array([49,9,121])))
        self.assertTrue(all(mL.backprop(np.array([2,3,4])) == 2 * np.array([2,3,4]) * np.array([7,3,-11])))

    def testMapLayerBatch(self):
        mL = nn.MapLayer(lambda x:x**2, lambda d:2*d)
        x = np.random.randn(2*4).reshape(-1,2)
        y = mL(x)
        self.assertTrue(np.all(np.equal(y, x**2)))
        odE = np.random.standard_normal(y.shape)
        idE = mL.backprop(odE)
        self.assertTrue(np.all(np.equal(idE, 2*x*odE)))

    def testAffineLayer(self):
        a = nn.AffineLayer(2,3)
        self.assertEqual(a(np.arange(2)).shape, (3,))
        x = np.random.randn(2)
        self.assertTrue(all(a(x) == x @ a.M + a.b))

    def testAffineLayerBatch(self):
        a = nn.AffineLayer(2,3)
        x = np.random.randn(4*2).reshape(-1,2) # A batch of four input vectors, of two dimensions
        y = a(x)
        #print(f"x is:\n{x}\ny is:\n{y}")
        self.assertEqual(y.shape, (4,3)) # A batch of four output vectors of three dimensions
        self.assertTrue(np.all(y == x @ a.M + a.b))

    def testAffineLayer_state_vector(self):
        a = nn.AffineLayer(2,3)
        sv = a.state_vector()
        self.assertEqual(sv.shape, ((2+1)*3,))
        x = arangep(2)
        a.set_state_from_vector(arangep((2+1)*3, 2))
        y = a(x)
        #print((a.M,a.b))
        #print(x)
        #print(y)
        self.assertTrue(all(np.equal(y, np.array([ 72, 94, 110]))))
        a.set_state_from_vector(sv)
        self.assertTrue(all(np.equal(sv, a.state_vector())))


    def testAffineLayerBackprop(self):
        #FIXME: this doesn't test backprop yet
        a = nn.AffineLayer(2,3)
        x = np.random.randn(2)
        dx = np.random.randn(2) * 0.001
        ypb = a(x)
        dy = (a(x + dx) - a(x - dx)) / 2.0
        y = a(x)
        self.assertEqual(y.shape, (3,))
        #print(f"a.M is {a.M}\na.b is {a.b}\nx is {x}\ny is {y}\ndx is {dx}\ndy is {dy}")
        out_delE = np.array([2, 3, 5])
        in_delE = a.backprop(dy)
        self.assertEqual(in_delE.shape, x.shape)
        #print(f"out_delE is {out_delE}\nbackprop(out_delE) = {in_delE}")
        #op = np.outer(dy, dx)
        #print(f"outer product of dx and dy is {op}")





if __name__ == '__main__':
    unittest.main()
