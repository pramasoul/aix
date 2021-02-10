#!/usr/bin/env python
# coding: utf-8

# Neural Net visualization tools
# `nnvis`


from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator, FormatStrFormatter
import plotly.graph_objects as go

import dill
import math


class NNVis:
    def __init__(self, bench):
        self.bench = bench
        self.gc_protect = []
        self.seed = 3
    
    def plot_learning(self, batches=100, batch_size=1):
        bench = self.bench
        losses = bench.learn(batches=batches, batch_size=batch_size)
        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.plot(losses, label=f"$\eta={bench.net.eta}$")  # Plot some data on the axes.
        ax.set_xlabel('learnings')  # Add an x-label to the axes.
        ax.set_ylabel('loss')  # Add a y-label to the axes.
        ax.set_title("Losses")  # Add a title to the axes.
        ax.set_yscale('log')
        ax.legend()  # Add a legend.        

    def plot_loss_cube(self, cube=None):
        if cube is None:
            cube = self.bench.loss_cube
        y = np.log2(cube['rates'])
        z = np.log10(cube['losses'])
        fig = go.Figure(data = go.Surface(z = z, y = y))
        fig.update_layout(width=800, height=800)
        fig.show()
        
    def knobs_plot_learning(self, batches=100, batch_size=1):
        n = batches
        bench = self.bench
        net = bench.net
        initial_state_vector = net.state_vector()
        # from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        a0 = 5
        f0 = 3
        
        ###
        losses = bench.learn(batches=batches, batch_size=batch_size)
        l, = ax.plot(losses, label=f"$\eta={net.eta}$")
        ax.set_xlabel('learnings')
        ax.set_ylabel('loss')
        ax.set_title("Losses")
        ax.set_yscale('log')
        ax.legend()

        axcolor = 'lightgoldenrodyellow'
        axeta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axnum = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        
        def sfunc(x):
            return 2**(-1.005/(x+.005))
        def sinv(x):
            return (-1.005/math.log2(x))-.005
        
        seta = Slider(axeta, '$\eta$', 0, 1, valinit=sinv(net.eta))
        snum = Slider(axnum, 'Num', 1, 10*n, valinit=n, valstep=1)
        
        filtfunc = [lambda x:x]
        
        big = max(losses)
        ax.set_title(f"$\eta$={net.eta:1.3e}")
        nlayers = [i for i in range(len(net.layers)) if hasattr(net.layers[i], 'M')]
        nl = len(nlayers)
        wpy = 0.8
        wph = .6
        weights_axes = [plt.axes([.025,wpy-wph*(i+1)/nl, 0.10,(wph-.1)/nl]) for i in range(nl)]
        def make_iax_images():
            return [weights_axes[i].imshow(np.concatenate(
                (net.layers[nlayers[i]].M,
                 np.atleast_2d(net.layers[nlayers[i]].b)),
                axis=0))
                    for i in range(len(nlayers))]
        def update_iax(imgs=[make_iax_images()]):
            for img in imgs[0]:
                img.remove()
            imgs[0] = make_iax_images()

        def update(val,ax=ax,loc=[l]):
            n = int(snum.val)
            #net = dill.loads(pickled_net)
            net.set_state_from_vector(initial_state_vector)
            
            net.eta = sfunc(seta.val)
            #seta.set_label("2.4e"%(self.net.eta,))
            #losses = filtfunc[0]([net.learn([fact]) for fact in bench.training_data_gen(n)])
            losses = filtfunc[0](bench.learn(batches=n, batch_size=1))
            big = max(losses)
            ax.set_title(f"$\eta$={net.eta:1.3e}")
            loc[0].remove()
            loc[0], = ax.plot(range(len(losses)), losses, lw=2,color='xkcd:blue', label=f"$\eta={net.eta:.2g}$")
            ax.set_xlim((0,len(losses)))
            ax.set_ylim((min(losses),big))
            update_iax()
            ax.legend()
            fig.canvas.draw_idle()

        seta.on_changed(update)
        snum.on_changed(update)

        resetax = plt.axes([0.7, 0.05, 0.2, 0.04])
        button = Button(resetax, 'Randomize net', color=axcolor, hovercolor='0.975')

    
        def reset(event):
            self.seed += 1
            self.bench.randomize_net()
            #self.bench.checkpoint_net()
            initial_state_vector = net.state_vector()
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
        self.gc_protect.append((update, reset, colorfunc, seta, snum, radio, button))


    def plot_trajectory(self, traja):
        # Development space for plotting:
        fig, ax = plt.subplots()  # Create a figure and an axes.
        traj_color = 'xkcd:red'
        loss_color = 'xkcd:blue'
        cos_color = 'xkcd:green'
        ax.set_xlabel('$n$')  # Add an x-label to the axes.
        ax.set_ylabel('$|\Delta state|$', color=traj_color)
        ax.tick_params(axis='y', labelcolor=traj_color)
        ax.set_title(f"$\eta={self.bench.net.eta}$")  # Add a title to the axes.
        ax.set_xscale('log')
        ax.set_yscale('log')
        tnl, = ax.plot(traja.traj_L2, label=f"traj norm", color=traj_color)  # Plot some data on the axes.
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.tick_params(axis='y', labelcolor=loss_color)
        dll, = ax2.plot(traja.loss_steps, label=f"$\Delta loss$", color=loss_color)  # Plot some data on the axes.
        #dll, = ax2.plot(np.tanh(10*traja.loss_steps), label=f"$\tanh(\Delta loss)$", color=loss_color)  # Plot some data on the axes.
        cosl, = ax2.plot(traja.traj_cos, label=f"$\Delta state cosine$", color=cos_color)
        ax.legend([tnl, dll, cosl], ["$\\|\\Delta state \\|$", "$\\Delta loss$", "$cos(\\theta)\Delta$"])  # Add a legend.
        #ax.legend([tnl, dll, cosl], ["$\\|\\Delta state \\|$", "$\\tanh(\\Delta loss)$", "$cos(\\theta)\Delta$"])  # Add a legend.
        #ax2.legend()  # Add a legend.
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

#-----------------------------------------------------------------------------------------------
if False: #boneyard
    def mpl_plot_loss_cube(self, cube=None):
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

    if False:
        self.input_width = None
        for layer in bench.net.layers:
            if hasattr(layer, 'M'):
                self.input_width = layer.M.shape[1]
                break
