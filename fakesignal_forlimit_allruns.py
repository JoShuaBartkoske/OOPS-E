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
from tqdm.auto import tqdm
import logging
import time
sns.set(font="Geneva",style='ticks',context='talk')

# functions

# VERITAS ECM empirical relations
## convert VERITAS ECM voltage to an integrated magnitude of stars in field of view of central pixel
def get_mag(signal):
    return ((np.log10(np.abs(signal)) - 2.27458167)/-0.40355447)

## integrated magnitude of stars in central pixel to a voltage in the VERITAS ECM
def get_V(mag):
    return 10**(-0.4*mag+2.27458167)

# simple sinusoid model for fake pulsar signal
def sinusoid(t,a,f,phi):
    return a*(np.cos(2*np.pi*f*t + phi)+ 1j*np.sin(2*np.pi*f*t))

# ChatGPT produced the frame for this function to convert the wikitable format to a panda dataframe.
# I made sure it works with the specific information I want from the wikitable.
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

df.drop([15])

# SET-UP
# set up the directory for the data
data_directory = '../data/psrj2229_fits/'

# set up file for logging
logfilename = "psrj2229_allfiles.log"
logging.basicConfig(filename=logfilename, level=logging.INFO)

logging.info(f"time started: {time.localtime()}")

# print every fits file in the data directory
for file in os.listdir(data_directory):
    if file.endswith('fits'):
        print(file)

# voltages for the injected sinusoidal signals from magnitudes 19, 20, 21, 22, and 23
mags = np.arange(start=19,stop=24)
amps = get_V(mags)
logging.info(f"Magnitudes tested: {mags}")
logging.info(f"Amplitudes corresponds to above magnitudes: {amps}")

# pulse frequency - doesn't really need to match actual frequency exactly
p = 19
logging.info(f"Pulse frequency: {p}")

        
# re-digitization step require defined digitization step and all values between -1 and 1.
step = 1.22e-5 #G100
edges = np.arange(-1,1,step)

all_pvals = []

# Now we use the Run name as the 'date' and loop
dates = df["Run name"].values
print(dates)

a = input("Pause before entering the abyss and state your cause.")

for j,rundate in enumerate(dates):
    print(f"Start analysis of run: {rundate}")
    print("---------------------------------------")
    logging.info(f"Start analysis of run: {rundate}")
    logging.info("---------------------------------------")
    
    # open files for each date
    # make sure we have the total number of telescopes for each date:
    telescopes = ["T1", "T2", "T3", "T4"]
    hduls = {}

    for tel in telescopes:
        path = f"../data/psrj2229_fits/j2229_{rundate}_{tel}.fits"
        if os.path.exists(path):
            hduls[tel] = fits.open(path)
            logging.info(f"   * Opened file: {path}")
        else:
            logging.warning(f"========File not found: {path}=======")
            continue

    if not hduls:
        logging.error(f"No valid FITS files found for run {rundate} skipping...")
        continue

    # DEFINE PARAMETERS USED ACROSS ALL FOUR TELESCOPES
    # declare number of harmonics
    harmonics = 1

    # on window - get from wiki page
    hz = float(df["Window size [Hz]"][j])
    logging.info(f"   * window size: {hz} Hz")

    # loop through amplitudes for each magnitude and test recovery and calculate significance
    for amp in tqdm(amps):
        logging.info(f"   * amplitude: {amp:.2e} V")
         # define array for p-values for each amplitude to combine all telescopes into one p-value
        p_array=[]

        # for each amplitude, we use all the available telescope data
        for tel, hdul in hduls.items():
            # define the flux and times from the data
            flux, times = hdul[1].data['signal'], hdul[1].data['time']
            
            # calculate the sampling rate
            samp = int(1/(times[1]-times[0]))

            # undigitize the data by randomly adding noise below the digitization limit
            on = flux + np.random.uniform((-1.22e-5)/2,(1.22e-5)/2,len(flux))

            # create periodic signal
            sine = np.real(sinusoid(times,amp,p,0))

            # add the periodic signal to the undigitized data
            signal = on + sine

            # re-digitize the data with the periodic signal
            dig = np.zeros(len(signal))
            dig_bins = np.digitize(signal,edges,right=True)
            for i,b in enumerate(dig_bins):
                dig[i] = edges[dig_bins[i]-1]
            spacing = get_spacing(samp,len(dig))


            # define number of points used in data
            pts = 2*hz/spacing
            # caluclate p-value using gumbel distribution
            pval = calc_p_gumball(times,dig,p,tel,rundate,spacing,amp, samp=samp,numpoints=pts,plot=True,showplots=False)

            p_array.append(pval)

        # calculate the total significance of the signal recovery from all telescopes with data
        sig = calc_sigma(p_array)
        print(amp,sig,p_array)
        logging.info(f"   * gumball_p_array: {p_array}")
        logging.info(f"   * total significance: {sig} \u03C3")


logging.info(f"time finished: {time.localtime()}")
