#!/usr/bin/env python
# coding: utf-8

import bqplot as bq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output
import ipywidgets as widgets
import time


# # Hacky and inefficient `matplotlib` approach

class MPLStripchart():
    def __init__(self, data_len):
        self.data_len = data_len
        self.data = np.array([], dtype=float)
    
    def __call__(self, new_data):
        l_new_data = new_data.size
        if l_new_data >= self.data_len:
            self.data = new_data[-self.data_len:]
        else:
            l_old_to_keep = max(0, self.data_len - l_new_data)
            l_new_to_add = min(self.data_len, l_new_data)
            #print(f"l_new_data = {l_new_data}"
            #      f", l_old_to_keep = {l_old_to_keep}"
            #      f", l_new_to_add = {l_new_to_add}")
            self.data = np.concatenate((self.data[-l_old_to_keep:], new_data[-l_new_to_add:]))
        data = self.data
        
        clear_output(True)
        plt.title("mean loss = %.3f" % np.mean(data[-10:]))
        plt.scatter(np.arange(len(data[-100:])), data[-100:], marker=".")
        plt.show()
        """
        fig, ax = plt.subplots(dpi=226)
        ax.plot(data, label=f"$\eta={1/7:.4g}$")
        ax.set_xlabel('learnings')
        ax.set_ylabel('loss')
        ax.set_title("Losses")
        ax.set_yscale('log')
        ax.legend()  # Add a legend.
        """


# # Using `bqplot`
class BQStripchart():
    def __init__(self, data_len, **kwargs):
        self.data_len = data_len
        data = self.data = np.array([], dtype=float)
        self.last_batch_number = 0
        x = np.arange(self.data.size)
        xs = bq.LinearScale()
        ys = bq.LogScale()
        xax = bq.Axis(scale=xs, label='batch', num_ticks=4)
        yax = bq.Axis(scale=ys, orientation='vertical', label='loss', num_ticks=4)
        scat = self.scat = bq.Scatter(x=x, y=data,
                                      scales={'x': xs, 'y': ys},
                                      default_size = 5)
        self.fig = bq.Figure(marks=[scat], axes=[xax, yax], title='Loss')
        # https://stackoverflow.com/a/20256491/3870917 :
        dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
        self.fig.layout = dictfilt(kwargs, ('height', 'width'))

    def __call__(self, new_data):
        #print(f"BQS({new_data})")
        if isinstance(new_data, (float, int)):
            new_data = np.array([new_data])
        l_new_data = new_data.size
        self.last_batch_number += l_new_data
        if l_new_data >= self.data_len:
            self.data = new_data[-self.data_len:]
        else:
            l_old_to_keep = max(0, self.data_len - l_new_data)
            l_new_to_add = min(self.data_len, l_new_data)
            #print(f"l_new_data = {l_new_data}"
            #      f", l_old_to_keep = {l_old_to_keep}"
            #      f", l_new_to_add = {l_new_to_add}")
            self.data = np.concatenate((self.data[-l_old_to_keep:], new_data[-l_new_to_add:]))
        data = self.data
        scat = self.scat
        with self.fig.hold_sync():
            scat.x = np.arange(self.last_batch_number - data.size, self.last_batch_number)
            scat.y = data
            self.fig.title = "mean loss = %.3f" % np.mean(data[-10:])

    ## FIXME
    def _repr_html_(self):
        #return self.fig._repr_html_()
        return r"""<strong>FIXME:</strong> Sorry, this doesn't work yet.
        (<code>'Figure' object has no attribute '_repr_html_'</code>!
        How does JL do it? I'm still trying to figure that out.)
            Just reference the <code>.fig</code> member directly instead for now."""

# A stripchart that handles non-uniform intervals
class NUStripchart():
    def __init__(self, data_len, title, y_label,
                 margin=20,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2,
                 log_scale = False, **kwargs):
        self.data_len = data_len
        self.xdq = deque([], data_len)
        self.ydq = deque([], data_len)
        xs = bq.LinearScale()
        ys = log_scale and bq.LogScale() or bq.LinearScale()
        xax = bq.Axis(scale=xs, label='batch', num_ticks=4)
        yax = bq.Axis(scale=ys, orientation='vertical', label=y_label, num_ticks=4)
        scat = self.scat = bq.Lines(x=list(self.xdq), y=list(self.ydq),
                                      scales={'x': xs, 'y': ys},
                                      min_aspect_ratio=min_aspect_ratio,
                                      max_aspect_ratio=max_aspect_ratio,
                                      default_size = 3,
                                      colors=['#15b01a'])
        self.fig = bq.Figure(marks=[scat], axes=[xax, yax], title=title)
        # https://stackoverflow.com/a/20256491/3870917 tweaked:
        dictfilt = lambda x, y: dict(((i,x[i]) for i in x if i in set(y)))
        self.fig.layout = dictfilt(kwargs, ('height', 'width'))

    def __call__(self, new_data):
        batch, loss = new_data
        self.xdq.append(batch)
        self.ydq.append(loss)
        scat = self.scat
        with self.fig.hold_sync():
            scat.x = list(self.xdq)
            scat.y = list(self.ydq)
            self.fig.title = "mean = %.4e" % np.mean(list(self.ydq)[-10:])

    ## FIXME
    def _repr_html_(self):
        #return self.fig._repr_html_()
        return r"""<strong>FIXME:</strong> Sorry, this doesn't work yet.
        (<code>'Figure' object has no attribute '_repr_html_'</code>!
        How does JL do it? I think an ipywidgets.Widget is needed.)
            Just reference the <code>.fig</code> member directly instead for now."""

# A stripchart more appropriate for tracking loss
class Losschart():
    def __init__(self, data_len, margin=20,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2,
                 title=None,
                 **kwargs):
        self.data_len = data_len
        self.xdq = deque([], data_len)
        self.ydq = deque([], data_len)
        xs = bq.LinearScale()
        ys = bq.LogScale()
        xax = bq.Axis(scale=xs, label='batch', num_ticks=4)
        yax = bq.Axis(scale=ys, orientation='vertical', label='loss', num_ticks=4)
        scat = self.scat = bq.Lines(x=list(self.xdq), y=list(self.ydq),
                                      scales={'x': xs, 'y': ys},
                                      min_aspect_ratio=min_aspect_ratio,
                                      max_aspect_ratio=max_aspect_ratio,
                                      default_size = 3,
                                      colors=['#0343df'])
        self.fig = bq.Figure(marks=[scat], axes=[xax, yax], title='Loss')
        self.fig.fig_margin = kwargs.get('fig_margin') \
            or dict(top=margin, bottom=margin, left=margin, right=margin)
        # https://stackoverflow.com/a/20256491/3870917 :
        dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
        self.fig.layout = dictfilt(kwargs, ('height', 'width'))

    def __call__(self, new_data):
        batch, loss = new_data
        self.xdq.append(batch)
        self.ydq.append(loss)
        scat = self.scat
        with self.fig.hold_sync():
            scat.x = list(self.xdq)
            scat.y = list(self.ydq)
            self.fig.title = "mean loss = %.4e" % np.mean(list(self.ydq)[-10:])

    ## FIXME
    def _repr_html_(self):
        #return self.fig._repr_html_()
        return r"""<strong>FIXME:</strong> Sorry, this doesn't work yet.
        (<code>'Figure' object has no attribute '_repr_html_'</code>!
        How does JL do it? I think an ipywidgets.Widget is needed.)
            Just reference the <code>.fig</code> member directly instead for now."""

