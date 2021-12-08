
import numpy as np
from harmoni.harmonitools import  harmonic_removal_simple, optimize_1_gridsearch
from scipy.signal import butter, filtfilt
from tools_signal import *
from numpy import pi
from tools_connectivity import compute_phase_connectivity, compute_phaseconn_with_permtest
from tools_general import *


def optimize_1_gridsearch2(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-.04, 0.04 + 0.01, 0.01)
    phi_range = np.arange(-pi/2, pi/2, pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_phaseconn_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return plv_sigx_yres_c_phi, c_range[ind_temp[0]], phi_range[ind_temp[1]]
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]



fs = 256  # sampling frequency
n_samples = int(5*60*60*fs)  # number of time samples
times = np.arange(0, n_samples)/fs  # the time points - used for plotting purpose
max_iter = 50

b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')


c_opt = np.zeros((max_iter,))
coh_before = np.zeros((max_iter,))
coh_after = np.zeros((max_iter,))

for n_iter in range(max_iter):
    print(n_iter)
    # n_iter = 0
    z = np.random.randn(1, n_samples)
    x = filtfilt(b10, a10, z)
    y = filtfilt(b20, a20, z)

    coh_before[n_iter] = compute_phase_connectivity(x, y, 1, 2, 'coh', type1='abs')

    ts1_h = hilbert_(x)
    ts1_ = np.abs(ts1_h) * np.exp(1j * 2 * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(y) / np.std(np.real(y))

    plv_sigx_yres_c_phi_all,  c_opt[n_iter], phi_opt = optimize_1_gridsearch2(ts2_, ts1_, fs, True, return_all=True)
    ts2_corr = ts2_ - c_opt[n_iter] * np.exp(1j * phi_opt) * ts1_
    # ts2_corr, c_opt[n_iter], _ = y_sig1_res = harmonic_removal_simple(x, y, fs, return_all=True)
    coh_after[n_iter] = compute_phase_connectivity(x, ts2_corr, 1, 2, 'coh', type1='abs')


plt.boxplot(np.concatenate((coh_before[:, np.newaxis], coh_after[:, np.newaxis]), axis=1), notch=True)
plt.boxplot(c_opt, notch=True)

plt.figure()
plt.subplot(121)
plt.scatter(np.abs(c_opt), coh_before)
# plt.plot(np.abs(c_opt), coh_after, '.')
plt.subplot(122)
plt.scatter(coh_after, coh_before)


c_range = np.arange(-0.04, .04 + 0.01, 0.01)
phi_range = np.arange(-pi/2, pi/2, pi / 10)
plot_3d(c_range, phi_range, plv_sigx_yres_c_phi_all)


