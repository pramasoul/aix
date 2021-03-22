#!/usr/bin/env python
# coding: utf-8

# # Neural Nets
# Version 0.5, in `nn`
#
# # Note:
# This is the source for `nn.py`, the foundation library of these neural net experiments. It contains a simple library, and a quantity, never sufficient, of test code, guarded by `if __name__ = '__main__'`. It makes various noises as it runs, but should not fail any assertsSee the bottom of the file for the procedure to produce the importable library file `nn.py`.

# Should do [Working efficiently with jupyter lab](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/)

# In[ ]:


import numpy as np


# A network built of components which:
# 1. accept an ordered set of reals (we'll use `numpy.array`, and  call them vectors) at the input port and produce another at the output port - this is forward propagation. ${\displaystyle f\colon \mathbf {R} ^{n}\to \mathbf {R} ^{m}}$
# 1. accept an ordered set of reals at the output port, representing the gradient of the loss function at the output, and produce the gradient of the loss function at the input port - this is back propagation, aka backprop. ${\displaystyle b\colon \mathbf {R} ^{m}\to \mathbf {R} ^{n}}$
# 1. from the gradient of the loss function at the output, calculate the partial of the loss function w.r.t the internal parameters ${\displaystyle \frac{\partial E}{\partial w} }$
# 1. accept a scalar $\eta$ to control the adjustment of internal parameters. _Or is this effected by scaling the loss gradient before passing? YES_
# 1. update internal parameters ${\displaystyle w \leftarrow w - \eta \frac{\partial E}{\partial w} }$
#

# In[ ]:


class Layer:
    def __init__(self):
        pass

    def __call__(self, x):
        """Compute response to input"""
        raise NotImplementedError

    def backprop(self, output_delE):
        """Use output error gradient to adjust internal parameters, return gradient of error at input"""
        raise NotImplementedError

    def state_vector(self):
        """Provide the layer's learnable state as a vector"""
        raise NotImplementedError

    def set_state_from_vector(self, sv):
        """Set the layer's learnable state from a vector"""
        raise NotImplementedError


# A network built of a cascade of layers:

# In[ ]:


class Network:
    version_string = "0.5"

    def __init__(self):
        self.layers = []
        self.eta = 0.1  # FIXME

    def extend(self, net):
        self.layers.append(net)
        return self

    def __call__(self, input):
        v = input
        for net in self.layers:
            v = net(v)
        return v

    def learn(self, facts, eta=None):
        eta = eta or self.eta
        for x, ideal in facts:
            y = self(x)
            e = y - ideal
            egrad = e * eta / e.shape[0]
            for net in reversed(self.layers):
                egrad = net.backprop(egrad)
        # loss = float(e.dot(e.T))/2.0
        loss = np.einsum("...ij,...ij", e, e) / (2.0 * e.shape[0])
        self.eta = eta
        return loss

    def losses(self, facts):
        return [
            np.einsum("...ij,...ij", e, e) / (2.0 * e.shape[0])
            for e in (self(x) - ideal for x, ideal in facts)
        ]

    def state_vector(self):
        """Provide the network's learnable state as a vector"""
        return np.concatenate([layer.state_vector() for layer in self.layers])

    def set_state_from_vector(self, sv):
        """Set the layer's learnable state from a vector"""
        i = 0
        for layer in self.layers:
            lsvlen = len(layer.state_vector())
            layer.set_state_from_vector(sv[i : i + lsvlen])
            i += lsvlen
        return self


# ___

# ## Useful Layers

# ### Identify

# In[ ]:


class IdentityLayer(Layer):
    def __call__(self, x):
        return x

    def backprop(self, output_delE):
        return output_delE

    def state_vector(self):
        return np.array([])

    def set_state_from_vector(self, sv):
        pass


# ### Affine
# A layer that does an [affine transformation](https://mathworld.wolfram.com/AffineTransformation.html) aka affinity, which is the classic fully-connected layer with output offsets.
#
# $$ \mathbf{M} \mathbf{x} + \mathbf{b} = \mathbf{y} $$
# where
# $$
# \mathbf{x} = \sum_{j=1}^{n} x_j \mathbf{\hat{x}}_j \\
# \mathbf{b} = \sum_{i=1}^{m} b_i \mathbf{\hat{y}}_i \\
# \mathbf{y} = \sum_{i=1}^{m} y_i \mathbf{\hat{y}}_i
# $$
# and $\mathbf{M}$ can be written
# $$
# \begin{bmatrix}
#     m_{1,1} & \dots & m_{1,n} \\
#     \vdots & \ddots & \vdots \\
#     m_{m,1} & \dots & m_{m,n}
# \end{bmatrix} \\
# $$

# #### Error gradient back-propagation
# $$
# \begin{align}
#  \frac{\partial loss}{\partial\mathbf{x}}
#   &= \frac{\partial loss}{\partial\mathbf{y}} \frac{\partial\mathbf{y}}{\partial\mathbf{x}} \\
#   &= \mathbf{M}^\mathsf{T}\frac{\partial loss}{\partial\mathbf{y}}
# \end{align}
# $$

# #### Parameter adjustment
# $$
#  \frac{\partial loss}{\partial\mathbf{M}}
#  = \frac{\partial loss}{\partial\mathbf{y}} \frac{\partial\mathbf{y}}{\partial\mathbf{M}}
#  = \frac{\partial loss}{\partial\mathbf{y}} \mathbf{x} \\
#  \frac{\partial loss}{\partial\mathbf{b}}
#  = \frac{\partial loss}{\partial\mathbf{y}} \frac{\partial\mathbf{y}}{\partial\mathbf{b}}
#  = \frac{\partial loss}{\partial\mathbf{y}}
# $$

# #### Adapting to `numpy`

# In `numpy` it is more convenient to use row vectors, particularly for calculating the transform on multiple inputs in one operation. We use the identity $ \mathbf{M} \mathbf{x} = (\mathbf{x} \mathbf{M}^\mathsf{T})^\mathsf{T}.$ To avoid cluttering names, we will use `M` in the code below to hold $\mathbf{M}^\mathsf{T}$.

# In[ ]:


class AffineLayer(Layer):
    """An affine transformation, which is the classic fully-connected layer with offsets.

    The layer has n inputs and m outputs, which numbers must be supplied
    upon creation. The inputs and outputs are marshalled in numpy arrays, 1-D
    in the case of a single calculation, and 2-D when calculating the outputs
    of multiple inputs in one call.
    If called with 1-D array having shape == (n,), e.g numpy.arange(n), it will
    return a 1-D numpy array of shape (m,).
    If called with a 2-D numpy array, input shall have shape (k,n) and will return
    a 2-D numpy array of shape (k,m), suitable as input to a subsequent layer
    that has input width m.
    """

    def __init__(self, n, m):
        self.M = np.empty((n, m))
        self.b = np.empty(m)
        self.randomize()

    def randomize(self):
        self.M[:] = np.random.randn(*self.M.shape)
        self.b[:] = np.random.randn(*self.b.shape)

    def __call__(self, x):
        self.input = x
        self.output = x @ self.M + self.b
        return self.output

    def backprop(self, output_delE):
        input_delE = output_delE @ self.M.T
        o_delE = np.atleast_2d(output_delE)
        self.M -= np.einsum("...ki,...kj->...ji", o_delE, np.atleast_2d(self.input))
        self.b -= np.sum(o_delE, 0)
        return input_delE

    def state_vector(self):
        return np.concatenate((self.M.ravel(), self.b.ravel()))

    def set_state_from_vector(self, sv):
        """Set the layer's learnable state from a vector"""
        l_M = len(self.M.ravel())
        l_b = len(self.b.ravel())
        self.M[:] = sv[:l_M].reshape(self.M.shape)
        self.b[:] = sv[l_M : l_M + l_b].reshape(self.b.shape)


# ### Map
# Maps a scalar function on the inputs, for e.g. activation layers.

# In[ ]:


class MapLayer(Layer):
    """Map a scalar function on the input taken element-wise"""

    def __init__(self, fun, dfundx):
        self.vfun = np.vectorize(fun)
        self.vdfundx = np.vectorize(dfundx)

    def __call__(self, x):
        self.input = x
        return self.vfun(x)

    def backprop(self, output_delE):
        input_delE = self.vdfundx(self.input) * output_delE
        return input_delE

    def state_vector(self):
        return np.array([])

    def set_state_from_vector(self, sv):
        pass


# ---

# # Tests
# *Incomplete* \
# Also `unittest` the `.py` version with a separate test script, see `test-nn_v3.py`.

# Make a few test arrays:

# In[ ]:


if __name__ == "__main__":
    one_wide = np.atleast_2d(np.arange(1 * 4)).reshape(-1, 1)
    print(f"one_wide is:\n{one_wide}")
    two_wide = np.arange(2 * 4).reshape(-1, 2)
    print(f"two_wide is:\n{two_wide}")
    three_wide = np.arange(3 * 4).reshape(-1, 3)
    print(f"three_wide is:\n{three_wide}\n")


# ## Tooling for Testing

# In[ ]:


if __name__ == "__main__":
    import sympy

    class VC:
        def grad(f, x, eps=1e-6):
            # epsihat = np.eye(x.size) * eps
            epsihat = np.eye(x.shape[-1]) * eps
            yp = np.apply_along_axis(f, 1, x + epsihat)
            ym = np.apply_along_axis(f, 1, x - epsihat)
            return (yp - ym) / (2 * eps)

        def tensor_grad(f, x, eps=1e-6):
            return np.apply_along_axis(lambda v: VC.grad(f, v, eps), 1, x)

    def closenuf(a, b, tol=0.001):
        return np.allclose(a, b, rtol=tol)

    def arangep(n, starting_index=0):
        sympy.sieve.extend_to_no(starting_index + n)
        return np.array(sympy.sieve._list[starting_index : starting_index + n])


# In[ ]:


# VC.grad(lambda x:x**2, three_wide[1])


# In[ ]:


# VC.tensor_grad(lambda x:x**2, three_wide)


# ---

# Input to a layer can be a single (row) vector, or a vertical stack of row vectors,
# a 2-d array that resembles a matrix. We need to test each layer class with both single and stacked input.

# ## Identity layer

# In[ ]:


if __name__ == "__main__":
    iL = IdentityLayer()

    # It's transparent from input to output
    assert np.equal(iL(np.arange(5)), np.arange(5)).all()
    assert (iL(three_wide) == three_wide).all()

    # It back-propagates the loss gradient without alteration
    assert np.equal(iL.backprop(np.arange(7)), np.arange(7)).all()
    assert (iL.backprop(three_wide) == three_wide).all()

    # It works for stacked input
    # (see above)


# ## Map layer

# #### Test single vector input behavior

# In[ ]:


if __name__ == "__main__":
    mL = MapLayer(lambda x: x ** 2, lambda d: 2 * d)

    # It applies the forward transformation
    assert np.equal(mL(np.array([-2, 1, 3])), np.array([4, 1, 9])).all()

    # It back-propagages the loss gradient
    x = np.array([1, 2, 2])
    y = mL(x)

    # for loss function, use L2-distance from some ideal
    # (divided by 2, for convenient gradient = error)
    ideal = np.array([2, 3, 5])
    loss = lambda v: (v - ideal).dot(v - ideal) / 2.0
    loss_at_y = loss(y)
    print(f"x = {x}, y = {y}, loss at y = {loss_at_y}")

    # find numerical gradient of loss function at y, the layer output
    grad_y = VC.grad(loss, y)
    print(f"∇𝑙𝑜𝑠𝑠(𝑦) = {grad_y}")

    # find the numerical gradient of the loss w.r.t. the input of the layer
    grad_x = VC.grad(lambda x: loss(mL(x)), x)
    print(f"∇𝑙𝑜𝑠𝑠(𝑥) = {grad_x}")

    # The backprop method does the same
    _ = mL(x)  # Make sure the last x is in the right place
    in_delE = mL.backprop(grad_y)
    print(f"backprop({grad_y}) = {in_delE}")
    assert closenuf(in_delE, grad_x)

    # The backprop operation did not change the behavior
    assert np.equal(mL(x), y).all()


# #### Test stacked-vectors input:

# In[ ]:


if __name__ == "__main__":
    mL = MapLayer(lambda x: x ** 2, lambda d: 2 * d)

    two_wide_sq = np.array([[0, 1], [4, 9], [16, 25], [36, 49]])
    # It applies the forward transformation
    assert np.equal(mL(two_wide), two_wide_sq).all()

    # It back-propagages the loss gradient
    x = two_wide
    y = mL(x)

    # for loss function, use L2-distance from some ideal
    # (divided by 2, for convenient gradient = error)
    ideal = two_wide * 2 + 11
    # print(y - ideal)
    # loss = lambda v: (v - ideal).dot(v - ideal) / 2.0
    loss = lambda v: np.einsum("ij,ij", v - ideal, v - ideal) / (2 * v.shape[0])
    loss_at_y = loss(y)
    print(f"x =\n{x}\ny =\n{y}, loss = {loss_at_y}\n")

    # find numerical gradient of loss function at y, the layer output
    grad_y = VC.tensor_grad(loss, y)
    print(f"∇𝑙𝑜𝑠𝑠(𝑦) = {grad_y}\n")

    # find the numerical gradient of the loss w.r.t. the input of the layer
    grad_x = VC.tensor_grad(lambda x: loss(mL(x)), x)
    print(f"∇𝑙𝑜𝑠𝑠(𝑥) = {grad_x}\n")

    # The backprop method does the same
    _ = mL(x)  # Make sure the last x is in the right place
    in_delE = mL.backprop(grad_y)
    print(f"backprop({grad_y}) =\n{in_delE}")
    assert closenuf(in_delE, grad_x)

    # The backprop operation did not change the behavior
    assert np.equal(mL(x), y).all()


# ## Affine layer

# #### Test single vector input behavior

# Test, for single input-vector operations:
# * input and output widths
# * state vector setting and getting
# * forward calculation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(2, 3)

    # The input and output widths are correct
    assert a(np.arange(2)).shape == (3,)

    # Its internal state can be set
    a.set_state_from_vector(np.arange(9))
    # and read back
    assert (a.state_vector() == np.arange(9)).all()
    # NOTE: The two assertions below are commented out because they depend
    # on white-box knowledge, and are duplicative of other tests
    # assert np.equal(a.M, np.array([[0, 1, 2],
    #                               [3, 4, 5]])).all()
    # assert np.equal(a.b, np.array([6, 7, 8])).all()

    # Its internal state observed using numerical gradient is correct
    x = np.random.rand(2)
    y = a(x)
    dydx = VC.grad(a, x)
    b = y - x.dot(dydx)
    # print(dydx, b)
    # print(dydx, np.arange(6).reshape(2,-1))
    assert closenuf(dydx, np.arange(6).reshape(2, -1))
    # print(b, np.arange(6, 9))
    assert closenuf(b, np.arange(6, 9))

    # It performs a single-input forward calculation correctly
    x = np.array([2, 1])
    y = a(x)
    # print(f"a.M is:\n{a.M}\na.b is {a.b}\nx is: {x}\ny is: {y}\n")
    assert (y == np.array([9, 13, 17])).all()

    # It performs a different single-input forward calculation correctly
    a.set_state_from_vector(np.array([2, 3, 5, 7, 11, 13, 17, 19, 23]))
    x = np.array([[29, 31]])
    y = a(x)
    assert (y == np.array([[292, 447, 571]])).all()


# Test, for single input-vector operations:
# * back-propagation of the loss gradient
# * learning (change in forward function) from the back-prop operation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(2, 3)
    a.set_state_from_vector(np.arange(9))

    # Doing a single-input-vector calculation
    x = np.array([2, 1])
    y = a(x)
    assert np.equal(y, np.array([9, 13, 17])).all()

    # It back-propagages the loss gradient
    ideal = np.array([11, 12, 10])
    loss = lambda v: (v - ideal).dot(v - ideal) / 2.0
    loss_at_y = loss(y)
    print(f"x = {x}, y = {y}, loss = {loss_at_y}")
    grad_y = VC.grad(loss, y)
    print(f"∇𝑙𝑜𝑠𝑠(𝑦) = {grad_y}")
    grad_x = VC.grad(lambda x: loss(a(x)), x)
    print(f"∇𝑙𝑜𝑠𝑠(𝑥) = {grad_x}")

    # Back-propagate the loss gradient from layer output to input
    _ = a(x)  # Make sure the last x is in the right place
    out_delE = grad_y * 0.1  # Backprop one-tenth of the loss gradient
    in_delE = a.backprop(out_delE)
    print(f"backprop({out_delE}) = {in_delE}")

    # The loss gradient back-propagated to the layer input is correct
    assert closenuf(in_delE / 0.1, grad_x)

    # And how did the learning affect the layer?
    print(f"Now a({x}) = {a(x)}, loss = {loss(a(x))}")
    print(f"state_vector is {a.state_vector()}")
    # FIXME: Check the change is correct


# #### Test batch operations

# Test, for batch operations:
# * input and output widths
# * forward calculation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(2, 3)
    a.set_state_from_vector(np.arange(9))

    # The input and output widths for the forward calculation are correct
    x = two_wide
    y = a(two_wide)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 3

    # The input and output widths for the backprop calculation are correct
    bp = a.backprop(three_wide * 0.001)
    assert bp.shape[0] == three_wide.shape[0]
    assert bp.shape[1] == x.shape[1]

    # The forward calculation is correct (in at least two instances)
    a.set_state_from_vector(np.arange(9))
    x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert (
        a(x) == np.array([[9, 11, 13], [15, 21, 27], [21, 31, 41], [27, 41, 55]])
    ).all()
    # print(f"a.M is:\n{a.M}\na.b is {a.b}\nx is: {x}\ny is: {y}")
    a.set_state_from_vector(np.array([2, 3, 5, 7, 11, 13, 17, 19, 23]))
    y = a(x)
    # print(f"x is: {x}\ny is: {y}")
    assert (
        y == np.array([[24, 30, 36], [42, 58, 72], [60, 86, 108], [78, 114, 144]])
    ).all()


# Test, for batch operations:
# * back-propagation of the loss gradient
# * learning (change in forward function) from the back-prop operation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(2, 3)
    a.set_state_from_vector(np.arange(9))
    x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    y = a(x)

    # It back-propagages the loss gradient

    # for loss function, use L2-distance from some ideal
    # (divided by 2, for convenient gradient = error)
    ideal = x @ arangep(2 * 3).reshape(2, 3) + arangep(
        3, 6
    )  # A known, different parameter setting
    print(f"y - ideal =\n{y - ideal}")
    # loss = lambda v: (v - ideal).dot(v - ideal) / 2.0
    loss = lambda v: np.einsum("ij,ij", v - ideal, v - ideal) / (2 * v.shape[0])
    loss_at_y = loss(y)
    print(f"x =\n{x}\nideal =\n{ideal}\ny =\n{y}, loss = {loss_at_y}\n")

    # find numerical gradient of loss function at y, the layer output
    grad_y = VC.tensor_grad(loss, y)
    print(f"∇𝑙𝑜𝑠𝑠(𝑦) =\n{grad_y}")

    # find the numerical gradient of the loss w.r.t. the input of the layer
    grad_x = VC.tensor_grad(lambda x: loss(a(x)), x)
    print(f"∇𝑙𝑜𝑠𝑠(𝑥) =\n{grad_x}")

    # Back-propagate the loss gradient from layer output to input
    _ = a(x)  # Make sure the last x is in the right place
    out_delE = grad_y * 0.01  # Backprop one percent of the loss gradient
    in_delE = a.backprop(out_delE)
    print(f"backprop({out_delE}) = {in_delE}")

    # The loss gradient back-propagated to the layer input is correct
    # assert closenuf(in_delE / 0.1, grad_x)

    # And how did the learning affect the layer?
    print(f"Now a({x}) = {a(x)}, loss = {loss(a(x))}")
    print(f"state_vector is {a.state_vector()}")
    # FIXME: Check the change is correct


# #### Test batch operations when the affine layer has only one input

# Test, for batch operations:
# * input and output widths
# * forward calculation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(1, 3)
    a.set_state_from_vector(np.arange(6))

    # The input and output widths for the forward calculation are correct
    x = one_wide
    y = a(one_wide)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 3

    # The input and output widths for the backprop calculation are correct
    bp = a.backprop(three_wide * 0.001)
    assert bp.shape[0] == three_wide.shape[0]
    assert bp.shape[1] == x.shape[1]

    # The forward calculation is correct (in at least two instances)
    a.set_state_from_vector(np.arange(6))
    x = np.array([[0], [1], [2], [3]])
    assert (
        y
        == np.array(
            [[3.0, 4.0, 5.0], [3.0, 5.0, 7.0], [3.0, 6.0, 9.0], [3.0, 7.0, 11.0]]
        )
    ).all()
    # print(f"a.M is:\n{a.M}\na.b is {a.b}\nx is: {x}\ny is: {y}")
    a.set_state_from_vector(np.array([2, 3, 5, 7, 11, 13]))
    y = a(x)
    # print(f"x is: {x}\ny is: {y}")
    assert (
        a(x) == np.array([[7, 11, 13], [9, 14, 18], [11, 17, 23], [13, 20, 28]])
    ).all()


# Test, for batch operations:
# * back-propagation of the loss gradient
# * learning (change in forward function) from the back-prop operation

# In[ ]:


if __name__ == "__main__":
    # Affine
    a = AffineLayer(1, 3)
    a.set_state_from_vector(np.arange(6))
    x = np.array([[0], [1], [2], [3]])
    y = a(x)
    # print(f"x =\n{x}\ny =\n{y}")

    # It back-propagages the loss gradient

    # for loss function, use L2-distance from some ideal
    # (divided by 2, for convenient gradient = error)
    ideal = x @ arangep(1 * 3).reshape(1, 3) + arangep(
        3, 6
    )  # A known, different parameter setting
    print(f"y - ideal =\n{y - ideal}")
    # loss = lambda v: (v - ideal).dot(v - ideal) / 2.0
    loss = lambda v: np.einsum("ij,ij", v - ideal, v - ideal) / (2 * v.shape[0])
    loss_at_y = loss(y)
    print(f"x =\n{x}\nideal =\n{ideal}\ny =\n{y}, loss = {loss_at_y}\n")

    # find numerical gradient of loss function at y, the layer output
    grad_y = VC.tensor_grad(loss, y)
    print(f"∇𝑙𝑜𝑠𝑠(𝑦) =\n{grad_y}")

    # find the numerical gradient of the loss w.r.t. the input of the layer
    grad_x = VC.tensor_grad(lambda x: loss(a(x)), x)
    print(f"∇𝑙𝑜𝑠𝑠(𝑥) =\n{grad_x}")

    # Back-propagate the loss gradient from layer output to input
    _ = a(x)  # Make sure the last x is in the right place
    out_delE = grad_y * 0.01  # Backprop one percent of the loss gradient
    in_delE = a.backprop(out_delE)
    print(f"backprop({out_delE}) = {in_delE}")

    # The loss gradient back-propagated to the layer input is correct
    # assert closenuf(in_delE / 0.1, grad_x)

    # And how did the learning affect the layer?
    print(f"Now a({x}) = {a(x)}, loss = {loss(a(x))}")
    print(f"state_vector is {a.state_vector()}")
    # FIXME: Check the change is correct


# ## Network

# ### Network assembly

# The simplest, the empty network, does identity:

# In[ ]:


if __name__ == "__main__":
    net = Network()
    assert all(x == net(x) for x in [0, 42, "cows in trouble"])
    assert all(
        (x == net(x)).all()
        for x in [np.arange(7), np.arange(3 * 4 * 5).reshape(3, 4, 5)]
    )


# A stack of maps composes the operations:

# In[ ]:


if __name__ == "__main__":
    net = Network()
    net.extend(MapLayer(lambda x: x ** 3, lambda d: 3 * d ** 2))
    assert all(net(x) == x ** 3 for x in [0, 42, -3.14])
    net.extend(MapLayer(lambda x: 7 - x, lambda d: -1))
    assert all(net(x) == 7 - x ** 3 for x in [0, 42, -3.14])

    # It operates on each element of an input vector separately
    assert (net(np.arange(4)) == 7 - np.arange(4) ** 3).all()


# A composition of affine transformations

# _[to do someday]_

# ### Network Learning

# Test simple batch learning of a single affine layer

# In[ ]:


if __name__ == "__main__":
    from pprint import pprint

    net = Network()
    a = AffineLayer(2, 3)
    a.set_state_from_vector(np.arange(9))  # A well-known initial state
    net.extend(a)
    print(f"\nNet has state {net.state_vector()}")

    x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])

    # The net wraps the layer
    y = a(x)
    assert (net(x) == y).all()

    # Make the training batch.
    # We use a separate affine layer, initialized differently, to determine the ideal
    t_a = AffineLayer(2, 3)
    t_a.set_state_from_vector(arangep(9))  # A known different initial state (of primes)
    ideal = t_a(x)

    fact = (x, ideal)
    print(f"fact is:")
    pprint(fact, indent=1)
    print(f"net(x) =\n{net(x)}")

    net.eta = 0.01
    for i in range(10):
        print(f"net.learn([fact]) = {net.learn([fact])}")
    print(f"net(x) =\n{net(x)}")

    # A simple fact yielder. Since it delivers multiple facts in succession,
    # it is a "facts", aka "batch cluster"
    def fact_ory(fact, n):
        for i in range(n):
            yield fact

    def facts_printer(facts):
        # after Network.learn
        for fact in facts:
            print(f"fact: ")
            pprint(fact)
            x, ideal = fact
            print(f"from which we get:\n\tx={x}\n\tideal={ideal}\n")

    # facts_printer(fact_ory(fact, 5))

    # print(f"list(fact_ory(facts[0], 3)) =\n{list(fact_ory(facts[0], 3))}\n")
    print(f"net.learn(fact_ory(fact,10)) = {net.learn(fact_ory(fact,10))}")
    print(f"net(x) =\n{net(x)}")
    for i in range(1000):
        loss = net.learn(fact_ory(fact, 10))
        if loss < 1e-7:
            break
    print(f"did {(i+1)*10} more learnings of fact. Now loss is {loss}")
    print(f"net(x) =\n{net(x)}")

    print(f"net.state_vector() = {net.state_vector()}")

    # The network has learned the target transform
    assert closenuf(net(x), fact[1])

if False:  # This section of the test is misconceived. Skip it
    # Save prior results and learn again, with different batch clustering
    prev_run_loss = loss
    prev_y = net(x)
    net.set_state_from_vector(np.arange(9))  # A well-known initial state
    print(f"\nReset net to state {net.state_vector()}")

    # Try multiple batches in each call to Network.learn
    def multibatch_fact_ory(fact, n):
        for i in range(n // 2):
            yield fact, fact

    facts_printer(multibatch_fact_ory(fact, 5))

    for i in range(1000):
        loss = net.learn(multibatch_fact_ory(fact, 10))
        if loss < 1e-7:
            break
    print(f"did {(i+1)*10} learnings of fact. Now loss is {loss}")
    print(f"net(x) =\n{net(x)}")

    # The results should match exactly
    assert loss == prev_run_loss
    assert (net(x) == prev_y).all()


# ### Test Network.losses

# In[ ]:


if __name__ == "__main__":
    # Make a network. Leave it with the default identity behavior.
    net = Network()

    x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    ideal = net(x)
    facts = [(x, ideal), (x, ideal - np.array([1, -1])), (x, 2 * x)]
    assert net.losses(facts) == [0, 1, 17.5]

    # Add some layers
    net.extend(AffineLayer(2, 3)).extend(MapLayer(np.sin, np.cos)).extend(
        AffineLayer(3, 2)
    )
    # Place it in a known state for test repeatability
    net.set_state_from_vector(np.arange(len(net.state_vector())))
    ideal = net(x)
    facts = [(x, ideal), (x, ideal - np.array([1, -1]))]
    # print(net.losses(facts))
    assert net.losses(facts) == [0, 1]


# ---

# # Publishing

# To produce an importable `nn.py`:
# 1. Save this notebook
# 1. Uncomment the `jupyter nbconvert` line below
# 1. Execute it.
# 1. Comment out the convert again
# 1. Save the notebook again in that form

# In[ ]:


###!jupyter nbconvert --to script nn.ipynb


# In[ ]:
