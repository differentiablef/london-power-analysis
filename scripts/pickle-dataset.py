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
            print(f' - Converting "{entry.name}" to pickle.',
                  end=' ', flush=True)
            
            # load file
            df = pd.read_csv( entry.path, names=headers )

            # remove extranious columns
            del df['acorn'], df['acorn-grouped'], df['pricing']

            # remove null entries
            rng = df['kWh'].apply(lambda x : x != 'Null')
            df = df.loc[rng]

            # shorten id's
            df['id'] = df['id'].apply(lambda x : x[3:])

            # set column types
            df = df.astype({
                'id': 'int16',
                'kWh': 'float32',
                'datetime': 'datetime64'})

            # remove double entries
            df = df.groupby(['id', 'datetime']).mean()
            
            # set column types
            # store file in a compressed pickle file
            outpath = os.path.join(out_dir,
                                   entry.name[:7]+'.pkl')
            df.to_pickle(outpath)

            print('done.')
