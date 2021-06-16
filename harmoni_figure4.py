"""
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, ..., Vadim V. Nikulin
(c) insert the future preprint and ms link here


** Code to generate Figure  4: harmoni block diagram**

-----------------------------------------------------------------------

(c) Mina jamshidi Idaji @ Neurolgy Dept, MPI CBS
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210615 by \Mina

"""

import numpy as np
from scipy.signal import butter, filtfilt
from harmoni_tools_signal import hilbert_
from harmoni_tools_connectivity import compute_phase_connectivity
from numpy import pi
from tools_harmonic_removal import optimize_1_gridsearch
from harmoni_tools_general import *
from harmoni_tools_signal import psd

def produce_nonsin_sig(x):
    x_h = hilbert_(x)

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


def _data_fun_pink_noise(times):
    import colorednoise as cn
    n_sample = len(times)
    data = cn.powerlaw_psd_gaussian(1, n_sample)
    data /= np.linalg.norm(data)
    return data
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

sfreq = 256  # sampling rate
dt = 1 / sfreq
T = 60 * 2 # total simulation time
t = np.arange(dt, T, dt) # time vector
n_samp = sfreq * T
times = np.arange(0, n_samp) / sfreq
b20, a20 = butter(N=2, Wn=np.array([19, 21]) / sfreq * 2, btype='bandpass')
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / sfreq * 2, btype='bandpass')


seed = 3100819795
np.random.seed(seed)

x = np.random.randn(n_samp)
x = filtfilt(b10, a10, x)
# y = signal.filtfilt(b20, a20, np.random.randn(n_samp))
x_nonsin1_ = produce_nonsin_sig(x)

pink_noise_1 = _data_fun_pink_noise(times)[np.newaxis, :]
pn11 = filtfilt(b10, a10, pink_noise_1)
pn12 = filtfilt(b20, a20, pink_noise_1)
r1 = np.var(x_nonsin1_) / np.var(pn11)
pink_noise_1 = pink_noise_1 * np.sqrt(r1 / 10)
pn2 = filtfilt(b10, a10, pink_noise_1)
pn2y = filtfilt(b20, a20, pink_noise_1)

r2 = np.var(x_nonsin1_) / (np.var(pn2) + np.var(pn2y))
x_nonsin1 = x_nonsin1_ + pink_noise_1

x1_orig = filtfilt(b10, a10, x_nonsin1_)
y1_orig = filtfilt(b20, a20, x_nonsin1_)

x1_ = filtfilt(b10, a10, x_nonsin1)
y1_ = filtfilt(b20, a20, x_nonsin1)

conn_x1_y1 = compute_phase_connectivity(x1_, y1_, 1, 2, 'coh', type1='abs')[0]
# conn_x1_y1_n = compute_phase_connectivity(x1_, pn2y, 1, 2, 'coh', type1='abs')[0]

ts1_ = np.abs(hilbert_(x1_)) * np.exp(1j * 2 * np.angle(hilbert_(x1_))) #sin(2*phase1)[np.newaxis, :]
ts1_ = hilbert_(ts1_) / np.std(np.real(ts1_))
ts2_ = hilbert_(y1_) / np.std(y1_)

# remove harmonic using harmoni
plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
ts2_corr = ts2_ - c_opt * np.exp(1j*phi_opt) * ts1_
ts2_corr = ts2_corr * np.std(y1_)

conn_ts1_ts2corr = compute_phase_connectivity(ts1_, ts2_corr, 1, 1, 'coh', type1='abs')[0]
# conn_ts1_ts2corr_n = compute_phase_connectivity(ts2_, ts2_corr, 1, 1, 'coh', type1='abs')[0]

print('coh_before=', str(conn_x1_y1), ', coh_after=', str(conn_ts1_ts2corr))

# plot the 3d optimization function
c_range = np.arange(-1, 1 + 0.01, 0.01)
phi_range = np.arange(-pi/2, pi/2, pi / 10)
plot_3d(c_range, phi_range, plv_sigx_yres_c_phi_all)

# plot the signals
plt.figure()
plt.plot(times, x_nonsin1_.ravel(), label='nonsin orig')
plt.plot(times, y1_.ravel(), label='beta noisy')
plt.plot(times, x1_.ravel(), label='alpha noisy')
plt.plot(times, ts2_corr.ravel(), label='beta corr noisy')
plt.plot(times, np.zeros(times.shape), label='zero')
plt.legend()

# plot psd
fig = plt.figure()
ax = plt.subplot(121)
psd(x_nonsin1_, sfreq, f_max=50, freq_res=1, fig=(fig, ax))
plt.title('original')
ax = plt.subplot(122)
psd(x_nonsin1, sfreq, f_max=50, freq_res=1, fig=(fig, ax))
plt.title('noisy')
