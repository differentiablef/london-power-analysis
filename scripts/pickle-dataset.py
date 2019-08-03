import pandas as pd
import numpy as np
import os, sys

# ##############################################################################

# column names for datafiles
headers=['id', 'pricing', 'datetime',
         'kWh', 'acorn', 'acorn-grouped']

raw_dir = os.path.join('.', 'raw')
out_dir = os.path.join('.', 'pickle')

raw_suffix = 'power-survey-london.csv.bz2'

# ##############################################################################

with os.scandir(raw_dir) as it:
    for entry in it:
        if entry.name.endswith( raw_suffix ):
            print(f' - Converting "{entry.name}" to pickle')
            
            # load file
            df = pd.read_csv( entry.path, names=headers )

            # remove extranious columns
            del df['acorn'], df['acorn-grouped'], df['pricing']

            # generate useful 'date' and 'time' column
            tmp = df['datetime'].apply(lambda x : x.split())
            df['date'] = pd.to_datetime(tmp.apply(lambda x : x[0]))
            df['time'] = pd.to_timedelta(tmp.apply(lambda x : x[1]), unit='m')

            # remove redundent column
            del df['datetime']

            # store file in a compressed pickle file
            outpath = os.path.join(out_dir,
                                   entry.name[:7]+'.pkl')
            df.to_pickle(outpath)


