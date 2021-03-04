#!/usr/bin/env python
# coding: utf-8

# ## A Neural Net lab bench
# `nnbench.py`

import numpy as np
from scipy import ndimage

from collections import deque, namedtuple
import dill
import itertools
import re

from deprecated.sphinx import deprecated

from nn import Network, AffineLayer, MapLayer

"""
    Terminology:
      Batch: an ordered pair of (x, ideal), where x and ideal can be vectors
             or vstacks of vectors, which look like matricies. Each iteration
             of Network.learn's loop processes one batch, and makes one
             increment of state vector change. A batch is a computationally-efficient
             grouping of facts to learn from.

      Batch cluster: an iterable of batches. Network.learn iterates over each batch
                     in a batch cluster, in a Python loop.

      Fact: a batch. Can be as simple as (array([1,2]), array([-1])) for e.g.
            a subtractor. Could be (array([[1,2],[7,3]]), array([[-1],[4]]))

      Facts: a batch cluster. Can be as simple as [(array([1,2]), array([-1]))]
             for a cluster containing a single batch, containing a single truth.

      a Truth: A single input (vector) and the corresponding ideal output, e.g (2,1)->(1,3).

      Lesson: A single invocation of Network.learn consuming facts in the form of a batch cluster.
              A lesson produces a single loss value.

      Learning: A sequences of lessons. A learning causes the net to pass through states,
                one for each lesson learned. Such a trajectory results in a like number of
                losses in sequence.
"""

class Thing:
    """A generic object to hold attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class NNBench:
    @staticmethod
    def net_input_width(net):
        # Hack to figure out the input width of the net
        input_width = None
        for layer in net.layers:
            if hasattr(layer, 'M'):
                input_width = layer.M.shape[1]
                break
        return input_width

    def __init__(self, net, ideal=lambda x:x):
        self(net)
        self.ideal = ideal
        self.seed = 3

        ###self.training_data_gen = self.training_data_gen_randn

        # training_batch(n) returns a single batch of n truths
        # It may disregard n if necessary
        # It defaults to producing the batch by generating from truths
        self.training_batch = self.training_batch_from_gen
        self.training_batch_cluster = self.training_batch_cluster_from_training_batch

    def __call__(self, net):
        self.net = net
        self.input_width = self.__class__.net_input_width(self.net)
        return self

    def checkpoint_net(self):
        self.net_checkpoint = self.net.state_vector()
        return self

    def rollback_net(self):
        self.net.set_state_from_vector(self.net_checkpoint)
        return self

    def save_net_to_file(self, f):
        dill.dump(self.net, f)
        return self

    def load_net_from_file(self, f):
        self.net = dill.load(f)
        return self

    def save_net_to_filename(self, name):
        with open(name, 'wb') as f:
            dill.dump(self.net, f)
        return self

    def load_net_from_filename(self, name):
        with open(name, 'rb') as f:
            self.net = dill.load(f)
        return self

    def randomize_net(self):
        for layer in self.net.layers:
            if hasattr(layer, 'randomize'):
                layer.randomize()
        return self

    def learn(self, batches=100, batch_size=1):
        # Trains the net by feeding it with batch clusters containing a single batch each,
        # obtaining thereby a loss value for each batch. Returns a list of the losses.
        return [self.net.learn(self.training_batch_cluster(batch_size)) for i in range(batches)]

    def learn_track(self, batches=100, batch_size=1):
        # Trains the net by feeding it with batch clusters containing a single batch each,
        # obtaining thereby a loss value for each batch. Records the net's state vector
        # before each lesson. Returns a list of the ordered pairs (state vector, loss).
        return [(self.net.state_vector(), self.net.learn(self.training_batch_cluster(batch_size))) \
                for i in range(batches)]

    def accept_source_of_truth(self, iterable):
        self.plato = itertools.cycle(iterable)
        return self

    def training_data_gen(self, batch_size=1):
        for i in range(batch_size):
            yield next(self.plato)

    def training_batch_from_gen(self, batch_size):
        x, y = zip(*self.training_data_gen(batch_size))
        return np.array(x), np.array(y)

    def training_batch_cluster_from_training_batch(self, n):
        return [self.training_batch(n)]

    @deprecated(reason="Too limited. Use `accept_source_of_truth` or another method.",
               category=FutureWarning)
    def training_data_gen_randn(self, n):
        """Yields n instances of labelled training data (aka "truths"). """
        np.random.seed(self.seed) #FIXME: chain forward
        width = self.__class__.net_input_width(self.net)
        for i in range(n):
            v = np.random.randn(width)
            yield (v, self.ideal(v)) #FIXME: means to alter input
            #FIX by using a method to obtain examples from domain

    @deprecated(reason="Too limited. Use `accept_source_of_truth` or another method.",
               category=FutureWarning)
    def training_data_gen_from_fixed(self, n):
        for i in range(n):
            yield next(self.fixed_training_data_iterator)

    def learning_potential(self, n=100, eta=None):
        starting_sv = self.net.state_vector()
        if eta is not None: # only change the net's eta if a value was passed to us
            self.net.eta = eta
        loss = self.net.learn(fact for fact in self.training_data_gen(n))
        self.net.set_state_from_vector(starting_sv)
        return -np.log(loss)

    def learn_loss_cube(self, n, rates):
        losses = []
        self.checkpoint_net()
        for eta in rates:
            self.net.eta = eta
            losses.append(self.learn(n))
            self.rollback_net()
        self.loss_cube = {
            'net': bytes(self.net_checkpoint),
            'n': n,
            'rates': rates.copy(),
            'losses': np.array(losses),
            'version': 0.1,
            'README': """Losses for a set of learning rates"""
            }
        return self.loss_cube


    def analyze_learning_track(self, learning_track):
        """Process the output of learn_track()"""
        trajectory = np.vstack([v[0] for v in learning_track])
        losses = np.vstack([v[1] for v in learning_track])

        # Take first differences, which represent the changes at each step
        traj_steps = np.diff(trajectory, axis=0)
        loss_steps = np.diff(losses, axis=0)

        # Find the L2 norm of the trajectory steps  ‖𝑡𝑟𝑎𝑗‖
        traj_L2 = np.sqrt(np.einsum('...i,...i', traj_steps, traj_steps))

        # Find the angles between trajectory steps, from
        # 𝐚⋅𝐛 = ‖𝐚‖‖𝐛‖ cos𝜃
        # cos𝜃 = 𝐚⋅𝐛 / ‖𝐚‖‖𝐛‖
        # where 𝐚 and 𝐛 are a state-space trajectory step and the succeeding step respectively
        trajn_dot_nplus1 = np.einsum('...i,...i', traj_steps[:-1], traj_steps[1:])
        traj_cos_denom = np.multiply(traj_L2[:-1], traj_L2[1:])
        traj_cos = np.divide(trajn_dot_nplus1, traj_cos_denom, where=traj_cos_denom!=0.0)

        rv = Thing(trajectory = trajectory,
                   losses = losses,
                   traj_steps = traj_steps,
                   loss_steps = loss_steps,
                   traj_L2 = traj_L2,
                   traj_cos = traj_cos
                  )
        return rv

################################################################################
# NNMEG is an instrumented version of nn.Network
class NNMEG(Network):
    """An instrumented subclass of nn.Network"""

    def __init__(self):
        super().__init__()
        meg = self._meg = Thing()
        meg.states = deque([], 3)
        meg.losses = deque([], 3)
        self._Deltas = namedtuple('Deltas', 'loss loss_step traj_L2 traj_cos')


    def learn(self, facts, eta=None):
        """learn, recording state and loss"""
        meg = self._meg
        meg.states or meg.states.append(self.state_vector()) # Capture the initial state
        loss = super().learn(facts, eta)
        meg.states.append(self.state_vector())
        meg.losses.append(loss)
        return loss

    def deltas(self):
        """Calculate the deltas of interest"""
        meg = self._meg
        trajectory = np.array(meg.states)
        losses = np.array(meg.losses)

        # Take first differences, which represent the changes at each step
        traj_steps = np.diff(trajectory, axis=0)
        loss_steps = np.diff(losses, axis=0)

        # Find the L2 norm of the trajectory steps  ‖𝑡𝑟𝑎𝑗‖
        traj_L2 = np.sqrt(np.einsum('...i,...i', traj_steps, traj_steps))
        #traj_l2 = np.linalg.norm(delta_sv)

        # Find the angles between trajectory steps, from
        # 𝐚⋅𝐛 = ‖𝐚‖‖𝐛‖ cos𝜃
        # cos𝜃 = 𝐚⋅𝐛 / ‖𝐚‖‖𝐛‖
        # where 𝐚 and 𝐛 are a state-space trajectory step and the succeeding step respectively
        trajn_dot_nplus1 = np.einsum('...i,...i', traj_steps[:-1], traj_steps[1:])
        traj_cos_denom = np.multiply(traj_L2[:-1], traj_L2[1:])
        traj_cos = np.divide(trajn_dot_nplus1, traj_cos_denom, where=traj_cos_denom!=0.0)

        if len(traj_cos) > 0:
            rv = self._Deltas(loss = losses[-1],
                   loss_step = loss_steps[-1],
                   traj_L2 = traj_L2[-1],
                   traj_cos = traj_cos[-1]
                  )
        else:
            rv = self._Deltas(*[0.0]*4)
        return rv


class NetMaker():
    """Makes a Network from a shorthand string"""
    p1 = re.compile('^(\d+)((x\d+[rst]?)+)$')
    p2 = re.compile('x(\d+)([rst]?)')
    funlayers = {'r': (lambda x: max(0.0, x), lambda d: 1 if d>0 else 0),
                 's': (np.sin, np.cos),
                 't': (np.tanh, lambda d: 1.0 - np.tanh(d)**2),
                }
    def __init__(self, cls=Network):
        self.cls = cls

    def __call__(self, shorthand):
        net = self.cls()
        m1 = self.p1.match(shorthand)
        if not m1:
            raise TypeError(f"mal-formed shorthand string {shorthand}")
        widths = [int(m1.group(1))]
        for match in self.p2.finditer(m1.group(2)):
            nstr, funcode = match.groups()
            widths.append(int(nstr))
            net.extend(AffineLayer(*widths[-2:]))
            if funcode:
                net.extend(MapLayer(*self.funlayers[funcode]))
        net.shorthand = shorthand
        return net
