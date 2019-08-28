
import scipy as sp
import numpy as np
import pandas as pd

from scipy.stats import norm as normal

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)

# histogram our data with numpy
data = pd.read_pickle('./pickle/2013-10x90-samp.pkl')
data_flat = pd.read_pickle('./pickle/2013-10x90-flat.pkl')
data_flat = np.log(data_flat[ (data_flat > 0.001)].dropna())
density, bins = np.histogram(data_flat, bins=125, density=True)

del data_flat

# date range

date_range = \
    pd.date_range(start='2013-01-01',
                  end='2013-07-30',
                  freq=pd.Timedelta('0.5H'))

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + density
nrects = len(left)

# ##############################################################################

nverts = nrects * (1 + 3 + 1)
verts = np.zeros((nverts, 2))

verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY


# ##############################################################################

patch = None


def animate(i):

    label = str(date_range[i])
    
    # extract
    region = data.loc[date_range[i], :]
    
    # transform
    region = np.log(region[ region > 0.001 ])
    
    n, _ = np.histogram(region.values,
                        bins=bins,
                        density=True)
    top = bottom + n - density
    verts[1::5, 1] = top
    verts[2::5, 1] = top

    legend.texts[0].set_text(label)
    print(label)
    
    return [patch, legend, ]

# ##############################################################################

fig, ax = plt.subplots()

fig.set_size_inches(16, 3)

barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='blue', edgecolor='gray', alpha=0.5)

ax.add_patch(patch)

#ax.add_patch(patches.PathPatch(
#    barpath, facecolor='lightblue', edgecolor='gray', alpha=0.5))

patch.set_label('')
legend = ax.legend((patch,), ('',) )

ax.set_xlim(left[0], right[-1])
ax.set_ylim(-top.max()*1.1, top.max()*1.1)

ani = animation.FuncAnimation(fig, animate,
                              len(date_range),
                              interval=10,
                              repeat=False, blit=True)


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

ani.save('test.mp4', writer=writer)
#plt.show()

