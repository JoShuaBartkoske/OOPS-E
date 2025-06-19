'''
This file will perform the secondary analysis showing the limiting magnitude we can hope to resolve for VERITAS ECM data
Author: Joshua Thomas Bartkoske (mostly taken from Samantha Wong's fakesignal_forlimit.ipynb and using Samantha Wong's oopse2.py functions)
DoC: June 18, 2025
'''

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import scipy as sp
from scipy import stats
from scipy.signal.windows import tukey
from scipy import optimize
import scipy.integrate as integrate
import scipy.stats as st
from astropy.io import fits
import seaborn as sns
from oopse2 import *
import os
sns.set(font="Geneva",style='ticks',context='talk')

def sinusoid(t,a,f,phi):
    return a*(np.cos(2*np.pi*f*t + phi)+ 1j*np.sin(2*np.pi*f*t))

# ChatGPT produced the frame.
def wiki_table_to_dataframe(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    headers = []
    rows = []
    current_row = []

    for line in lines:
        if line.startswith('!'):
            # Each header is its own line
            headers.append(line[1:].strip())
        elif line.startswith('|-'):
            # Start a new row
            if current_row:
                rows.append(current_row)
                current_row = []
        elif line.startswith('|'):
            # Value line: add to the current row
            value = line[1:].strip()
            if value == '}':
                continue
            current_row.append(value if value else np.nan)

    # Catch the last row if not followed by a |- at the end
    if current_row:
        rows.append(current_row)

    print(rows)
    # Check if column count matches
    for i, row in enumerate(rows):
        if len(row) != len(headers):
            print(row)
            raise ValueError(f"Row {i} has {len(row)} values but {len(headers)} headers")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


# Example usage
df = wiki_table_to_dataframe('wikitable.txt')
print(df)

# set up the directory for the data
data_directory = '../data/psrj2229_fits/'

# print every fits file in the data directory
for file in os.listdir(data_directory):
    if file.endswith('fits'):
        print(file)

# Now we use the Run name as the 'date' and loop


