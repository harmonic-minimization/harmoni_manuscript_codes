
import numpy as np
from harmoni.harmonitools import  harmonic_removal_simple, optimize_1_gridsearch
from scipy.signal import butter, filtfilt
from tools_signal import *
from numpy import pi

fs = 256  # sampling frequency
n_samples = int(20*60*fs)  # number of time samples
times = np.arange(0, n_samples)/fs  # the time points - used for plotting purpose
max_iter = 50

b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')


c_opt = np.zeros((max_iter,))
for n_iter in range(max_iter):
    print(n_iter)
    z = np.random.randn(1, n_samples)
    x = filtfilt(b10, a10, z)
    y = filtfilt(b20, a20, z)

    ts1_h = hilbert_(x)
    ts1_ = np.abs(ts1_h) * np.exp(1j * 2 * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(y) / np.std(np.real(y))

    # plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, fs, True, return_all=True)

    _, c_opt[n_iter], _ = y_sig1_res = harmonic_removal_simple(x, y, fs, return_all=True)
    print(c_opt[n_iter])



c_range = np.arange(-1, 1 + 0.01, 0.01)
phi_range = np.arange(-pi/2, pi/2, pi / 10)
plot_3d(c_range, phi_range, plv_sigx_yres_c_phi_all)
