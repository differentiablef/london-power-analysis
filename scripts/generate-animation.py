
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
        self.density = \
            counts/counts.sum()


        self.ymin = 0.0
        self.ymax = 2.0 * self.density.max()
        
        #self.bins = pd.qcut( tmp, 50, precision=10, retbins=True )[1]
        
        del tmp

        # plot stuff
        self.axes.set_xlim(max([mu-4*sigma, min(self.density.index)]),
                           min([mu+4*sigma, max(self.density.index)]) )
        
        self.axes.set_ylim(self.ymin, self.ymax)
        
        self.stem1 = ax.scatter(self.density.index.values,
                                self.density.values, s=0.5)
        
        self.stem = ax.scatter(self.density.index.values,
                               self.density.values,
                               label='local', s=0.7)
        
        self.legend = ax.legend()
        self.vlines = ax.vlines( [  ], self.ymin, self.ymax,
                                 alpha=0.5, linestyles='dashed',
                                 colors=['green', 'darkgray', 'darkgray' ])
        
        self.mu=mu; self.sigma=sigma;
        pass

    def init(self):
        
        #self.stem.
        
        return self.stem, self.legend,

    def __call__(self, i):
        if i==0:
            return self.init()

        self.current=i
        
        # extract region for i'th date/time
        region = pd.Series(
            self.data.loc[self.date_range[i], :].values.flatten())

        # compute \mu and \sigma for the region
        mu_0 = region.mean()
        sigma_0 = region.std()

        # compute rel. frequencies
        locald = region.value_counts()
        locald = locald/locald.sum()

        # initialize new offsets array with zeros
        offsets = np.zeros((len(self.density.index), 2))
        
        pos = list(self.density.index)
        for val in locald.index:
            ii = pos.index(val)
            offsets[ii,:] = [val, locald[val]]

        # save new line information for \mu and \mu +/- \sigma segments
        segments = np.array(
            [[ [mu_0, self.ymin], [mu_0, self.ymax ] ],
             [ [mu_0 + sigma_0, self.ymin], [mu_0+sigma_0, self.ymax] ],
             [ [mu_0 - sigma_0, self.ymin], [mu_0-sigma_0, self.ymax] ]])

        # update \mu and \mu +/- \sigma lines
        self.vlines.set_segments( segments )
        self.vlines.changed()

        # update scatter plot of rel. frequencies
        self.stem.set_offsets(offsets)

        # set the legend text to the date being displayed
        self.legend.texts[0].\
            set_text( str(self.date_range[i]) )

        # print the date being displayed to stdout
        print(str(self.date_range[i]))
        
        return self.stem, self.legend, self.vlines,
    

        

# script entry-point ###########################################################

if __name__=='__main__':
    global data
    # load data.
    data = pd.read_pickle('./pickle/2013-12.pkl')
    
    # restrict to relevant value region 
    tmp = data.stack()
    tmp = tmp[(6 >= tmp)&(tmp >= 0.000)]
    data = tmp.unstack(0)
    
    fig, ax = plt.subplots()

    fig.set_size_inches(6, 5)
    
    uu = UpdateDist(ax, start='2013-12-01', end='2013-12-31', delta='0.5H')
    anim = FuncAnimation(fig, uu,
                         frames=np.arange(1441),
                         init_func=uu.init, interval=200, blit=True)
    plt.show()
