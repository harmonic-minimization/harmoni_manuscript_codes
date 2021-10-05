
import numpy as np
from scipy.signal import butter, filtfilt, sawtooth
import matplotlib.pyplot as plt

from tools_signal import plot_fft, hilbert_
from tools_connectivity import compute_phase_connectivity


# ToDo: use MI index. you dont need filtering the env in that case
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
sig7 = filtfilt(b72, a72, sig_sawtooth)


# complex signal for phase and amplitude extraction
sig1_h = hilbert_(sig1)
sig7_h = hilbert_(sig7)

sig73 = filtfilt(b72, a72, np.random.randn(n_samp))
sig73_h = hilbert_(sig73)

# sig12_amp = filtfilt(b1, a1, np.random.randn(n_samp))
# sig12_amp = np.abs(hilbert_(sig12_amp))
# sig12_phi = filtfilt(b1, a1, np.abs(sig73_h))
# sig12_phi = np.abs(sig73_h)
# sig12_phi = np.angle(hilbert_(sig12_phi))
# sig12 = np.real(sig12_amp * np.exp(1j * sig12_phi))
sig12 = demean(np.abs(sig73_h))
sig12_h = hilbert_(sig12)
compute_phase_connectivity(demean(np.abs(sig72_h)), sig12_h, 1, 1, 'coh', type1='abs')

sig74 = sig73 + np.sqrt(40) * sig7
sig74_2 = filtfilt(b72, a72, np.real(sig74))
sig74_env = filtfilt(b1, a1, np.abs(hilbert_(sig74_2)))

compute_phase_connectivity(sig74_env, sig12_h, 1, 1, 'coh', type1='abs')

y2 = remove_harmonic(sig1, sig74_env, sfreq, n=1)

compute_phase_connectivity(demean(y2), sig12_h, 1, 1, 'coh', type1='abs')
