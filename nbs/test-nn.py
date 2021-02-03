import unittest
import numpy as np
import nn

class TestNN(unittest.TestCase):
    
    def testIdentityLayer(self):
        iL = nn.IdentityLayer()
        self.assertTrue(all(iL(np.arange(5)) == np.arange(5)))
        self.assertTrue(all(iL.backprop(np.arange(7)) == np.arange(7)))

    def testMapLayer(self):
        mL = nn.MapLayer(lambda x:x**2, lambda d:2*d)
        self.assertTrue(all(mL(np.array([7,3,-11])) == np.array([49,9,121])))
        self.assertTrue(all(mL.backprop(np.array([2,3,4])) == 2 * np.array([2,3,4]) * np.array([7,3,-11])))
    
    def testAffineLayer(self):
        a = nn.AffineLayer(2,3)
        self.assertEqual(a(np.arange(2)).shape, (3,))
        x = np.random.randn(2)
        self.assertTrue(all(a(x) == x @ a.M + a.b))
        
    def testAffineLayerBackprop(self):
        a = nn.AffineLayer(2,3)
        return #FIXME
        dx = np.random.randn(2) * 0.001
        dy = (a(x + dx) - a(x - dx)) / 2.0
        y = a(x)
        print(f"a.M is {a.M}\na.b is {a.b}\nx is {x}\ny is {y}\ndx is {dx}\ndy is {dy}")
        out_delE = np.array([2, 3, 5])
        in_delE = a.backprop(dy)
        print(f"out_delE is {out_delE}\nbackprop(out_delE) = {in_delE}")
        #op = np.outer(dy, dx)
        #print(f"outer product of dx and dy is {op}")
        



        
if __name__ == '__main__':
    unittest.main()
