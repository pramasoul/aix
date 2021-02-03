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
    
    def checkpoint_net(self):
        self.net_checkpoint = dill.dumps(self.net)
        
    def rollback_net(self):
        self.net = dill.loads(self.net_checkpoint)

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
        np.random.seed(self.seed)
        width = self.input_width
        for i in range(n):
            v = np.random.randn(width)
            yield (v, self.ideal(v))

    def training_data_gen_fixed(self, n):
        len_td = len(self.training_data)
        for i in range(n):
            yield self.training_data[i % len_td]
        
    def learn(self, n=100):
        return [self.net.learn([fact]) for fact in self.training_data_gen(n)]
            
    def learn_track(self, n=100):
        return [(self.net.state_vector(), self.net.learn([fact])) for fact in self.training_data_gen(n)]

    def learning_potential(self, n=100, eta=None):
        stash = dill.dumps(self.net)
        if eta is not None: # only change the net's eta if a value was passed to us
            self.net.eta = eta
        loss = self.net.learn(fact for fact in self.training_data_gen(n))
        self.net = dill.loads(stash)
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
    
    def was_mpl_plot_loss_cube(self, cube=None):
        if cube is None:
            cube = self.loss_cube
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.log2(cube["rates"])
        Y = np.arange(1, cube["n"] + 1)
        Z = np.log10(cube["losses"])
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    
    def was_plot_loss_cube(self, cube=None):
        if cube is None:
            cube = self.loss_cube
        y = np.log2(cube['rates'])
        z = np.log10(cube['losses'])
        fig = go.Figure(data = go.Surface(z = z, y = y))
        fig.update_layout(width=800, height=800)
        fig.show()

    def was_plot_learning(self, n):
        # self.losses = losses = [self.net.learn(fact for fact in self.training_data_gen(n))]
        losses = self.learn(n)
        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.plot(losses, label=f"$\eta={self.net.eta}$")  # Plot some data on the axes.
        ax.set_xlabel('learnings')  # Add an x-label to the axes.
        ax.set_ylabel('loss')  # Add a y-label to the axes.
        ax.set_title("Losses")  # Add a title to the axes.
        ax.set_yscale('log')
        ax.legend()  # Add a legend.
        """
        plt.yscale('log')
        plt.plot(range(len(losses)),losses)
        plt.show(block=0)
        """
        
    def was_knobs_plot_learning(self, n):
        pickled_net = dill.dumps(self.net)
        # from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        a0 = 5
        f0 = 3
        
        ###
        losses = [self.net.learn([fact]) for fact in self.training_data_gen(n)]
        #l, = plt.plot(range(len(losses)), losses, lw=2)
        l, = ax.plot(losses, label=f"$\eta={self.net.eta}$")  # Plot some data on the axes.
        #ax.margins(x=0)
        #plt.yscale('log')
        ax.set_xlabel('learnings')  # Add an x-label to the axes.
        ax.set_ylabel('loss')  # Add a y-label to the axes.
        ax.set_title("Losses")  # Add a title to the axes.
        ax.set_yscale('log')
        ax.legend()  # Add a legend.

        axcolor = 'lightgoldenrodyellow'
        axeta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axnum = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        
        def sfunc(x):
            return 2**(-1.005/(x+.005))
        def sinv(x):
            return (-1.005/math.log2(x))-.005
        
        seta = Slider(axeta, '$\eta$', 0, 1, valinit=sinv(self.net.eta))
        snum = Slider(axnum, 'Num', 1, 10*n, valinit=n, valstep=1)
        
        filtfunc = [lambda x:x]
        
        
        big = max(losses)
        ax.set_title(f"$\eta$={self.net.eta:1.3e}")
        nlayers = [i for i in range(len(self.net.layers)) if hasattr(self.net.layers[i], 'M')]
        nl = len(nlayers)
        wpy = 0.8
        wph = .6
        weights_axes = [plt.axes([.025,wpy-wph*(i+1)/nl, 0.10,(wph-.1)/nl]) for i in range(nl)]
        def make_iax_images():
            return [weights_axes[i].imshow(np.concatenate(
                (self.net.layers[nlayers[i]].M,
                 np.atleast_2d(self.net.layers[nlayers[i]].b)),
                axis=0))
                    for i in range(len(nlayers))]
        def update_iax(imgs=[make_iax_images()]):
            for img in imgs[0]:
                img.remove()
            imgs[0] = make_iax_images()

        def update(val,ax=ax,loc=[l]):
            n = int(snum.val)
            self.net = dill.loads(pickled_net)
            
            self.net.eta = sfunc(seta.val)
            #seta.set_label("2.4e"%(self.net.eta,))
            losses = filtfunc[0]([self.net.learn([fact]) for fact in self.training_data_gen(n)])
            big = max(losses)
            ax.set_title(f"$\eta$={self.net.eta:1.3e}")
            loc[0].remove()
            loc[0], = ax.plot(range(len(losses)), losses, lw=2,color='xkcd:blue', label=f"$\eta={self.net.eta:.2g}$")
            ax.set_xlim((0,len(losses)))
            ax.set_ylim((min(losses),big))
            update_iax()
            ax.legend()
            fig.canvas.draw_idle()

        seta.on_changed(update)
        snum.on_changed(update)

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    
        def reset(event):
            self.seed += 1
            update()
        button.on_clicked(reset)

        rax = plt.axes([0.025, 0.025, 0.15, 0.15], facecolor=axcolor)
        radio = RadioButtons(rax, ('raw', 'low pass', 'green'), active=0)

        
        def colorfunc(label):
            if label == "raw":
                filtfunc[0] = lambda x:x
            elif label == "low pass":
                filtfunc[0] = lambda x:ndimage.gaussian_filter(np.array(x),3)
            #l.set_color(label)
            #fig.canvas.draw_idle()
            update()
        radio.on_clicked(colorfunc)

        plt.show()
        #return 'gc protect:', update, reset, colorfunc,seta,snum, radio, button
        self.gc_protect.append((update, reset, colorfunc,seta,snum, radio, button))

