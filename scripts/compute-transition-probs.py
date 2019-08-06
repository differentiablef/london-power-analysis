# ##############################################################################

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ##############################################################################

dt = pd.Timedelta(30, unit='m')

# ##############################################################################

# load the data for 2013
data = pd.read_pickle('./pickle/2013-pivot.pkl').round(6)

print(f' - Computing Transition Probabilities:', end=' ', flush=True)
# compute transition matrix for each sample path
for samp_id in data.columns:
    print(f'{samp_id:06}', end=' ', flush=True)
    
    # determine which states the sample visits
    states = data[samp_id].unique()
    
    # initialize transition dict
    trans=dict()

    # compute transition  frequencies for each state
    for state in states:
        # initialize state's entry in transition matrix
        trans[state] = dict()

        # find every where the state has been observed
        observed = (data[samp_id] == state)

        # get the corresponding observation times
        times = data.loc[observed, samp_id].index

        # update transition frequencies
        for t in times:
            if t+dt in data[samp_id].index:
                new_state = data.loc[t+dt, samp_id]
                trans[state][new_state] = \
                    trans[state].setdefault(new_state, 0.0) + 1

        # normalize frequencies
        for state in trans:
            sigma = sum(trans[state].values())
            for new_state in trans[state]:
                trans[state][new_state] = trans[state][new_state]/sigma

    # store transition probabilities
    pd.to_pickle(trans, f'./pickle/trans-dist/{samp_id:06}.pkl')
    del trans
    pass
