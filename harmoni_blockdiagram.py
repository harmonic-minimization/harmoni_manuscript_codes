"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
script for:
** Harmoni block diagram **

-----------------------------------------------------------------------

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210927 by \Mina

"""

import numpy as np
from numpy import pi
from scipy.signal import butter, filtfilt
from tools_signal import hilbert_
from tools_connectivity import compute_phase_connectivity
from tools_harmonic_removal import optimize_1_gridsearch
from tools_simulations import data_fun_pink_noise
from tools_general import *
from tools_signal import psd


def produce_nonsin_sig(x1):
    x_h = hilbert_(x1)

    n = 2
    sigma2 = np.random.random(1) * 2 * pi - pi
    y2_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma2)
    y2 = np.real(y2_h)

    n = 3
    sigma3 = np.random.random(1) * 2 * pi - pi
    y3_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma3)
    y3 = np.real(y3_h)

    n = 4
    sigma4 = np.random.random(1) * 2 * pi - pi
    y4_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma4)
    y4 = np.real(y4_h)

    x_nonsin = x + 0.25 * y2 + 0.06 * y3 + 0.025 * y4
    return x_nonsin


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# --------------------------------------
# general settings
# --------------------------------------

sfreq = 256  # sampling rate
snr_desired = 10
dt = 1 / sfreq
T = 60 * 2 # total simulation time
t = np.arange(dt, T, dt) # time vector
n_samp = sfreq * T
times = np.arange(0, n_samp) / sfreq
b20, a20 = butter(N=2, Wn=np.array([16, 24]) / sfreq * 2, btype='bandpass')
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / sfreq * 2, btype='bandpass')


# --------------------------------------
# generate the non-sin signal
# --------------------------------------
seed = 3100819795
np.random.seed(seed)
x = np.random.randn(n_samp)
x = filtfilt(b10, a10, x)
x_nonsin1_ = produce_nonsin_sig(x)

# filter the original non-sin signal within the fundamental and 2nd harmonic frequency bands,
# we need this as the bench-mark
x1_orig = filtfilt(b10, a10, x_nonsin1_)
y1_orig = filtfilt(b20, a20, x_nonsin1_)
# ----------------------------------------------------------------------------
# generate the pink noise, tune the SNR, and add the noise to the non-sin signal
# ----------------------------------------------------------------------------
# generate the noise signal
pink_noise_1 = data_fun_pink_noise(times)[np.newaxis, :]

# filter the noise within the alpha band to compute the noise power
pn11 = filtfilt(b10, a10, pink_noise_1)
# pn12 = filtfilt(b20, a20, pink_noise_1)

# compute the current SNR, i.e. before tuning to the desired value
r1 = np.var(x_nonsin1_) / np.var(pn11)

# rescale the noise signal to reach the desired noise power
pink_noise_1 = pink_noise_1 * np.sqrt(r1 / snr_desired)

# filter noise again and compute SNR
pn2 = filtfilt(b10, a10, pink_noise_1)
pn2y = filtfilt(b20, a20, pink_noise_1)
r2 = np.var(x_nonsin1_) / np.var(pn2)  # should be equal to the SNR_desired

# add noise to the signal
x_nonsin1 = x_nonsin1_ + pink_noise_1

# ----------------------------------------------------------------------------
# plot psd of the clean and noisy non-sinusoidal signal
# ----------------------------------------------------------------------------
fig = plt.figure()
ax = plt.subplot(121)
psd(x_nonsin1_, sfreq, f_max=50, freq_res=1, fig=(fig, ax))
plt.title('original')
ax = plt.subplot(122)
psd(x_nonsin1, sfreq, f_max=50, freq_res=1, fig=(fig, ax))
plt.title('noisy')


"""
from here on, we pretend that we have the noisy non-sin signal and we wanna use Harmoni to suppress the
harmonic info
"""

# *****************************************
# *****************************************
# HARMONI    ******************************
# *****************************************
# *****************************************

# filter the noisy, non-sin signal within the bands of interest
x1_ = filtfilt(b10, a10, x_nonsin1)
y1_ = filtfilt(b20, a20, x_nonsin1)

# compute the coherence of the 1st and 2nd harmonic components of the noisy signal
# for the original signal this is almost 1
conn_x1_y1 = compute_phase_connectivity(x1_, y1_, 1, 2, 'coh', type1='abs')[0]

# accelerate the fundamental component by a factor of n=2
ts1_ = np.abs(hilbert_(x1_)) * np.exp(1j * 2 * np.angle(hilbert_(x1_)))  # phase warping

# normalize the power of the two components at the 1st and 2nd harmonic frequency bands
ts1_ = hilbert_(ts1_) / np.std(np.real(ts1_))
ts2_ = hilbert_(y1_) / np.std(y1_)

# remove harmonic using the grid-search minimization
plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
ts2_corr = ts2_ - c_opt * np.exp(1j*phi_opt) * ts1_
ts2_corr = ts2_corr * np.std(y1_)

# compute the coherence of the fundamental component and the residual of the 2nd harmonic
conn_ts1_ts2corr = compute_phase_connectivity(ts1_, ts2_corr, 1, 1, 'coh', type1='abs')[0]

print('coh_before=', str(conn_x1_y1), ', coh_after=', str(conn_ts1_ts2corr))


# -----------------------------------
# plot the 3d optimization function
# -----------------------------------
c_range = np.arange(-1, 1 + 0.01, 0.01)
phi_range = np.arange(-pi/2, pi/2, pi / 10)
plot_3d(c_range, phi_range, plv_sigx_yres_c_phi_all)


# -----------------------------------
# plot the signals
# -----------------------------------
plt.figure()
plt.plot(times, x_nonsin1_.ravel()+2.5, label='nonsin orig')
plt.plot(times, y1_.ravel()+1.5, label='beta noisy')
plt.plot(times, x1_.ravel()+0.8, label='alpha noisy')
plt.plot(times, ts2_corr.ravel(), label='beta corr noisy')
plt.legend()
