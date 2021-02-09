#!/usr/bin/env python
# coding: utf-8

# ## A Neural Net lab bench
# `nnbench.py`


from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator, FormatStrFormatter
import plotly.graph_objects as go

import dill
import math

class Thing:
    """A generic object to hold attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class NNBench:
    def __init__(self, net, ideal=lambda x:x):
        self.net = net
        self.ideal = ideal
        self.gc_protect = []
        self.seed = 3
        self.input_width = None
        for layer in net.layers:
            if hasattr(layer, 'M'):
                self.input_width = layer.M.shape[1]
                break
        self.training_data_gen = self.training_data_gen_randn
        self.training_batch = self.training_batch_from_gen
    
    def checkpoint_net(self):
        #self.net_checkpoint = dill.dumps(self.net)
        self.net_checkpoint = self.net.state_vector()
        
    def rollback_net(self):
        self.net.set_state_from_vector(self.net_checkpoint)

    def save_net_to_file(self, f):
        dill.dump(self.net, f)

    def load_net_from_file(self, f):
        self.net = dill.load(f)

    def save_net_to_filename(self, name):
        with open(name, 'wb') as f:
            dill.dump(self.net, f)
        
    def load_net_from_filename(self, name):
        with open(name, 'rb') as f:
            self.net = dill.load(f)

    def randomize_net(self):
        for layer in self.net.layers:
            if hasattr(layer, 'randomize'):
                layer.randomize()

    def training_data_gen_randn(self, n):
        """Generate n instances of labelled training data"""
        np.random.seed(self.seed) #FIXME: chain forward
        width = self.input_width
        for i in range(n):
            v = np.random.randn(width)
            yield (v, self.ideal(v)) #FIXME: means to alter input

    def training_data_gen_fixed(self, n):
        len_td = len(self.training_data)
        for i in range(n):
            yield self.training_data[i % len_td]

    def was_learn(self, n=100):
        return [self.net.learn([fact]) for fact in self.training_data_gen(n)]

    def learn(self, batches=100, batch_size=1):
        return [self.net.learn(self.training_batch(batch_size)) for i in range(batches)]
            
    def training_batch_from_gen(self, batch_size):
        x, y = zip(*self.training_data_gen(batch_size))
        return np.array(x), np.array(y)

    def learn_track(self, batches=100, batch_size=1):
        return [(self.net.state_vector(), self.net.learn(self.training_batch(batch_size))) for i in range(batches)]

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
    
