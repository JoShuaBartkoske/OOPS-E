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
    logging.info(f"Start analysis of run: {rundate}")
    logging.info("---------------------------------------")
    
    # open four files for each date
    hdul1 = fits.open(f"../data/psrj2229_fits/j2229_{rundate}_T1.fits")
    hdul2 = fits.open(f"../data/psrj2229_fits/j2229_{rundate}_T2.fits")
    hdul3 = fits.open(f"../data/psrj2229_fits/j2229_{rundate}_T3.fits")
    hdul4 = fits.open(f"../data/psrj2229_fits/j2229_{rundate}_T4.fits")

    # define signal and times from data for each telescope
    t1, time1 = hdul1[1].data['signal'], hdul1[1].data['time']
    t2, time2 = hdul2[1].data['signal'], hdul2[1].data['time']
    t3, time3 = hdul3[1].data['signal'], hdul3[1].data['time']
    t4, time4 = hdul4[1].data['signal'], hdul4[1].data['time']

    # calculate sampling rate for T1 and the other three telescope
    samp1 = int(1/(time1[1]-time1[0]))
    samp = int(1/(time2[1]-time2[0]))

    # undigitize the data by randomly adding noise below the digitization limit
    on1 = t1 + np.random.uniform((-1.22e-5)/2,(1.22e-5)/2,len(t1))
    on2 = t2 + np.random.uniform((-1.22e-5)/2,(1.22e-5)/2,len(t2))
    on3 = t3 + np.random.uniform((-1.22e-5)/2,(1.22e-5)/2,len(t3))
    on4 = t4 + np.random.uniform((-1.22e-5)/2,(1.22e-5)/2,len(t4))

    for amp in tqdm(amps):
        logging.info(f"   * amplitude of injected signal: {amp} V")
        sin1 = np.real(sinusoid(time2,amp,p,0))
        sin2 = np.real(sinusoid(time2,amp,p,0))
        sin3 = np.real(sinusoid(time3,amp,p,0))
        sin4 = np.real(sinusoid(time4,amp,p,0))
        
        signal1 = on1 + sin1
        signal2 = on2 + sin2
        signal3 = on3 + sin3
        signal4 = on4 + sin4

        # re-digitization
        dig1 = np.zeros(len(signal1))
        dig2 = np.zeros(len(signal2))
        dig3 = np.zeros(len(signal3))
        dig4 = np.zeros(len(signal4))

        dig_bins1 = np.digitize(signal1,edges,right=True)
        dig_bins2 = np.digitize(signal2,edges,right=True)
        dig_bins3 = np.digitize(signal3,edges,right=True)
        dig_bins4 = np.digitize(signal4,edges,right=True)

        for i,b in enumerate(dig_bins2):
            dig1[i] = edges[dig_bins1[i]-1]
            dig2[i] = edges[dig_bins2[i]-1]
            dig3[i] = edges[dig_bins3[i]-1]
            dig4[i] = edges[dig_bins4[i]-1]
        
        spacing1 = get_spacing(samp,len(dig2))
        spacing2 = get_spacing(samp,len(dig2))
        spacing3 = get_spacing(samp,len(dig3))
        spacing4 = get_spacing(samp,len(dig4))
        
        
        harmonics = 1
        p_array=[]
        
        # on window - get from wiki page
        hz = float(df["Window size [Hz]"][j])
        logging.info(f"   * window size: {hz} Hz")
        
        pts1 = 2*hz/spacing1
        pts2 = 2*hz/spacing2 
        pts3 = 2*hz/spacing3
        pts4 = 2*hz/spacing4

        p1 = calc_p_gumball(time1,dig1,p,1,rundate,spacing1,amp, samp=samp1,numpoints=pts1,plot=True,showplots=False)
        p2 = calc_p_gumball(time2,dig2,p,2,rundate,spacing2,amp, samp=samp, numpoints=pts2,plot=True,showplots=False)
        p3 = calc_p_gumball(time3,dig3,p,3,rundate,spacing3,amp, samp=samp, numpoints=pts3,plot=True,showplots=False)
        p4 = calc_p_gumball(time4,dig4,p,4,rundate,spacing4,amp, samp=samp, numpoints=pts4,plot=True,showplots=False)
        
        p_array.append(p1)
        p_array.append(p2)
        p_array.append(p3)
        p_array.append(p4)
        
        all_pvals.append(p_array)
        sig = calc_sigma(p_array)
        print(amp,sig,p_array)
        logging.info(f"   * gumball_p_array: {p_array}")
        logging.info(rf"   * total significance: {sig} $\sigma$")

logging.info(f"time finished: {time.localtime()}")
