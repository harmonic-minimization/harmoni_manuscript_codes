"""
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, ..., Vadim V. Nikulin
(c) insert the future preprint and ms link here


** Code to generate Figure 1: Sawtooth signal and its the fundamental component and the harmonics **

-----------------------------------------------------------------------

(c) Mina jamshidi Idaji @ Neurolgy Dept, MPI CBS
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210615 by \Mina

"""

import numpy as np
import matplotlib.pyplot as plt
from harmoni_tools_signal import plot_fft, hilbert_
from scipy.signal import butter, filtfilt, sawtooth
from harmoni_tools_connectivity import compute_phase_connectivity
from tools_general import *



# --------------------------------------
# general settings
# --------------------------------------
sfreq = 512  # sampling rate
dt = 1 / sfreq
t_len = 60 * 2  # total simulation time
t = np.arange(dt, t_len, dt) # time vector
n_samp = sfreq * t_len
times = np.arange(0, n_samp) / sfreq


# --------------------------------------
# generate signal
# --------------------------------------

# sawtooth generation
t = np.linspace(0, t_len, n_samp)
f0 = 6
sig_sawtooth = sawtooth(2 * np.pi * f0 * t, .1)
plot_fft(sig_sawtooth, sfreq)


# build the filters for fundamental and harmonic frequencies
b1, a1 = butter(N=2, Wn=np.array([f0-1, f0+1])/sfreq*2, btype='bandpass')
b7, a7 = butter(N=2, Wn=np.array([7*f0 - 1, 7*f0 + 1])/sfreq*2, btype='bandpass')
b72, a72 = butter(N=2, Wn=np.array([6*f0 - 1, 8*f0 + 1])/sfreq*2, btype='bandpass')

# filter, zero-phase
sig1 = filtfilt(b1, a1, sig_sawtooth)
sig7 = filtfilt(b7, a7, sig_sawtooth)
sig72 = filtfilt(b72, a72, sig_sawtooth)

# complex signal for phase and amplitude extraction
sig1_h = hilbert_(sig1)
sig7_h = hilbert_(sig7)
sig72_h = hilbert_(sig72)

# compute PAC as the synchronization of the lower-frequency signal and the amplitude of the higher-frequency  signal
pac17 = compute_phase_connectivity(sig1_h, np.abs(sig7_h), 1, 1, 'coh', type1='abs')
pac172 = compute_phase_connectivity(sig1_h, np.abs(sig72_h), 1, 1, 'coh', type1='abs')

coh17 = compute_phase_connectivity(sig1_h, sig7_h, 1, 7, 'coh', type1='abs')
coh172 = compute_phase_connectivity(sig1_h, sig72_h, 1, 7, 'coh', type1='abs')

print('narrow-band filter at the harmonic frequency: coh17=', str(coh17), ', pac17=', str(pac17))
print('wider filter at the harmonic frequency: coh172=', str(coh172), ', pac172=', str(pac172))

# plot the one-second segment between 40s and 41 s
ind1 = np.argmin(np.abs(t - 40.045))
ind2 = np.argmin(np.abs(t - 41.045))

plt.figure()
plt.plot(t[ind1:ind2], 0.25 * sig_sawtooth[ind1:ind2], label='sawtooth signal')
plt.plot(t[ind1:ind2], 0.25 * sig1[ind1:ind2], label='fundamental component')
plt.plot(t[ind1:ind2], 2 * sig7[ind1:ind2], label='harmonic component')
plt.plot(t[ind1:ind2], 2 * sig72[ind1:ind2], label='wide harmonic component')
plt.legend()

plt.figure()
plt.plot(t[ind1:ind2], sig1[ind1:ind2], label='harmonic component')
plt.plot(t[ind1:ind2], np.angle(sig1_h.ravel())[ind1:ind2], label='harmonic component')
plt.legend()

plt.figure()
plot_fft(sig_sawtooth, sfreq)
plot_fft(sig1, sfreq)
plot_fft(sig7, sfreq)
plot_fft(sig72, sfreq)
plt.legend(['sawtooth', '1st harmonic', '7th harmonic-narrow', '7th harmonic-wide'])

