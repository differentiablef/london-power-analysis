# imports ######################################################################

import os, sys, json
import matplotlib.pyplot as plt
import pandas as pd

# variables/methods ############################################################
 
nb_dir = os.path.join('.', 'notebooks')
nb_ext = '.ipynb'

import_path = os.path.join(nb_dir, 'pickup_utilities.py')

found_init = False
show_called = False

def do_something_useful():
    global show_called
    show_called = True
    pass

# setup context ################################################################

# pull in notebooks/pickup_utilities.py
with open(import_path, newline='') as infile:
    source = infile.read()
    code = compile( source, 'pickup_utilities', 'exec')
    exec(code)
    pass

# load 'df' from file 
df = pd.read_csv('./data/complete.csv')
df['tstamp'] = df.apply(tstampCalc, axis = 1)

# overwrite show function
plt.show = do_something_useful

# script entry-point ###########################################################

# process notebooks
for filename in os.listdir(nb_dir):    
    filepath = os.path.join(nb_dir, filename)
    if filename.endswith(nb_ext):
        # read in notebook as json, and extract 'cells' entry 
        with open(filepath) as infile:
            nbcells = json.load(infile).get('cells')
            pass

        # begin processing notebook file
        print(f'- Processing notebook {filename}')

        # search for valid initialization cell
        for ii, cell in enumerate(nbcells):
            # concat. all lines of 'source'; generating full cell source-code 
            source = ''.join( cell.get('source', []) )
            
            # search source for slack'd out snippet which ..
            #     .. should be in the notebook somewhere.
            if cell['cell_type'] == 'code' and \
               source.find('from pickup_utilities import')>=0 and \
               source.find('pd.read_csv("../data/complete.csv")')>=0:
                found_init = True
                break # exit this loop


        # check to see if we found a valid initalization cell.
        if not found_init: # if not, ...
            # tell the user we are skipping the notebook
            print(' + No valid init. cell found. ignoring.')
            
        else: # if so, ...
            # tell user we are interested in notebook
            print(' + Found valid init. cell')
            
            # reset switch
            found_init = False 

            # delete found initalization cell
            print(' + Removing init. cell')
            del nbcells[ii]

            # generate basepath for saving plots
            figbasepath = os.path.join(
                '.', 'images', filename[:-6])

            # compile/exec remaining code cells 
            print(f' + Compiling/Running remaining notebook code cells')
            img_id = 1
            for ii, cell in enumerate(nbcells):
                # concat. lines of 'source' together to get full cell source
                source = ''.join( cell.get('source', []) )

                # if 'cell_type' is 'code', then ...
                if cell['cell_type'] == 'code':
                    # compile the source into python byte-code
                    print(f'  - Compiling cell {ii}')
                    code = compile( source, filename[:-6], 'exec')

                    # execute compiled byte-code in the current 'context'
                    print(f'  - Running cell {ii}')
                    
                    os.chdir(nb_dir)
                    exec(code)
                    os.chdir('..')

                    # check to see if any plots have been generated
                    if show_called:
                        # reset show_called switch
                        show_called = False

                        # save any figures generated by running 'code'
                        print(f'  * Saving plot {img_id}')
                        plt.savefig(figbasepath + f'-{img_id}.png')
                        plt.close()
                        img_id += 1
                        pass
