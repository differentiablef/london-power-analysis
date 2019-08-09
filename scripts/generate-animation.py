
# imports ######################################################################

import os, sys
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import numpy as np
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()

# defs #########################################################################


class UpdateDist(object):
    def __init__(self, ax, start='2013-05-25 00:00', end='2013-06-01 00:00', delta='0.5H'):
        global data

        self.ymin = 0.0
        self.ymax = 0.016
        
        # store instance specific variables
        self.axes = ax
        self.current = start
        self.delta = pd.to_timedelta(delta)
        self.date_range = \
            pd.date_range(start=start, end=end, freq=delta)
        
        #self.bins = bins
        #self.midpoints = \
        #    (bins[:-1] + bins[1:])/2.0
        
        self.data = \
            data.loc[self.date_range, :]

        tmp = self.data.stack().dropna()

        sigma = tmp.std()
        mu = tmp.mean()
        counts = tmp.value_counts().sort_index()
        density = \
            counts/counts.sum()

        del tmp

        # plot stuff
        self.axes.set_xlim(max([mu-4*sigma, min(density.index)]),
                           min([mu+4*sigma, max(density.index)]) )
        
        self.axes.set_ylim(self.ymin, self.ymax)
        
        self.stem1 = ax.scatter(density.index.values,
                                density.values, s=0.5)
        
        self.stem = ax.scatter(density.index.values,
                               density.values,
                               label='local', s=0.7)
        
        self.legend = ax.legend()
        self.vlines = ax.vlines( [  ], self.ymin, self.ymax,
                                 alpha=0.5, linestyles='dashed',
                                 colors=['gray', 'darkgray', 'darkgray' ])
        
        self.mu=mu; self.sigma=sigma; self.density=density;
        pass

    def init(self):
        
        #self.stem.
        
        return self.stem, self.legend,

    def __call__(self, i):
        if i==0:
            return self.init()

        self.current=i
        # cal density for surrounding date/time
        region = pd.Series(
            self.data.loc[
                self.date_range[i], :].values.flatten())
        
        mu_0 = region.mean()
        sigma_0 = region.std()
        
        locald = region.value_counts()
        locald = locald/locald.sum()

        
        offsets = np.zeros((len(self.density.index), 2))
        for ii, val in enumerate(self.density.index):
            if val in locald.index.values:
                offsets[ii,:] = [val, locald[val]]
            else:
                offsets[ii,:] = [val, 0.0]
            pass

        segments = np.array(
            [[ [mu_0, self.ymin], [mu_0, self.ymax ] ],
             [ [mu_0 + sigma_0, self.ymin], [mu_0+sigma_0, self.ymax] ],
             [ [mu_0 - sigma_0, self.ymin], [mu_0-sigma_0, self.ymax] ]])

        self.vlines.set_segments( segments )
        self.vlines.changed()
        self.stem.\
            set_offsets(offsets)

        self.legend.texts[0].\
            set_text( str(self.date_range[i]) )

        print(str(self.date_range[i]))
        
        return self.stem, self.legend, self.vlines,
    

        

# script entry-point ###########################################################

if __name__=='__main__':
    global data
    # load data.
    data = pd.read_pickle('./pickle/2013-01.pkl')
    # restrict to relevant value region and apply np.log
    tmp = data.stack()
    tmp = np.log(tmp[(2 > tmp)&(tmp > 0.01)])
    data = tmp.unstack(0)
    
    fig, ax = plt.subplots()
    uu = UpdateDist(ax, start='2013-01-01', end='2013-01-28', delta='0.5H')
    anim = FuncAnimation(fig, uu,
                         frames=np.arange(289),
                         init_func=uu.init, interval=200, blit=True)
    plt.show()
