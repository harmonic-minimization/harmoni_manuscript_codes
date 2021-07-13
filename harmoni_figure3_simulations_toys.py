"""
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, ..., Vadim V. Nikulin
(c) insert the future preprint and ms link here


** Code to generate Figure 3:toy examples**
panel A: scenario 1
panel B: scenario 3
panel C: scenario 4
panel D:  scenario 5
-----------------------------------------------------------------------

(c) Mina jamshidi Idaji @ Neurolgy Dept, MPI CBS
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------
last modified: 20210512 \Mina
20200414 \Mina

-----------------------------------------------------------------------
-----------------------------------------------------------------------

the two-signal scenario without noise
±±±±±±±±±±± ±±±±±±±±±±± ±±±±±±±±±±±

Scenario 1:
------------
x1  -R-  x3
|        |
y1  -S-  y3



scenario 2:
--------------
x1  -R-  x3
|        |
y1  -S-  y3

y2       y4


scenario 3:
--------------
x1  -R-  x3
|        |
y1  -S-  y3

y2  -R-  y4


scenario 4:
--------------
x1       x3
|        |
y1       y3

y2  -R-  y4

scenario 5:
--------------

x1  -R-  y4
|
y1       x3
         |
y2       y3

scenario 6:
--------------
x1 <--> x3
x1 <--> y4

x1  -R-  y4
|   ..
y1    ...x3
         |
y2       y3

±±±±±±±±±±± ±±±±±±±±±±± ±±±±±±±±±±±

"""

import numpy as np
import os.path as op
from numpy import pi
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter
from tools_general import *
from tools_signal import *
from scipy import stats
from tools_connectivity import compute_coherency, compute_plv, compute_plv_windowed, compute_plv_with_permtest
from tools_simulation import _data_fun_pink_noise, filtered_randn, produce_nm_phase_locked_sig
from tools_harmonic_removal import optimize_1
from tools_harmonic_removal import *


def adjust_snr(sig, noise, snr, freq):
    b2, a2 = butter(2, freq, btype='bandpass')
    noise_nb = filtfilt(b2, a2, noise)
    noise_var = np.mean(noise_nb**2)
    sig_var = np.mean(sig**2)
    snr_current = sig_var / noise_var
    factor1 = np.sqrt(snr_current / snr)
    return factor1


def remove_harmonic(ts1, ts2, sfreq):
    ts1_h = hilbert_(ts1)
    ts1_ = np.abs(ts1_h) * np.exp(1j * 2 * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(ts2) / np.std(np.real(ts2))

    plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
    ts2_corr = ts2_ - c_opt * np.exp(1j * phi_opt) * ts1_
    return ts2_corr


# --------------------
# parameters
# --------------------
path_save_fig = '/NOBACKUP/Results/results_HR/Simulation/toyexp/fig/'
fs = 256
n_samples = int(1 * 60 * fs)
times = np.arange(0, n_samples)/fs
max_iter = 50
c_y2 = 1
c_y4 = 1
kappa = None
noisy = 1
SNR_alpha = dBinv(5)
SNR_beta = dBinv(-5)
scenario = 7
coh = True

if scenario == 1:
    x1_x3_coupling = 1
    y2_y4_exist = 0
elif scenario == 2:
    x1_x3_coupling = 1
    y2_y4_exist = 1
    y2_y4_coupling = 0
elif scenario == 3:
    x1_x3_coupling = 1
    y2_y4_exist = 1
    y2_y4_coupling = 1
elif scenario == 4:
    x1_x3_coupling = 0
    y2_y4_exist = 1
    y2_y4_coupling = 1
elif scenario == 5:
    x1_x3_coupling = 0
    y2_y4_exist = 1
    y2_y4_coupling = 0
elif scenario == 6:
    x1_x3_coupling = 1
    y2_y4_exist = 1
    y2_y4_coupling = 0
elif scenario == 7:
    x1_x3_coupling = 1
    y2_y4_exist = 1
    y2_y4_coupling = 0

b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')

c_abs_opt_1 = np.empty((max_iter,))
c_phi_opt_1 = np.empty((max_iter,))
c_abs_opt_2 = np.empty((max_iter,))
c_phi_opt_2 = np.empty((max_iter,))

plv_sig1x_sig1y = np.empty((max_iter,))
plv_sig1x_yres1 = np.empty((max_iter,))
plv_sig2x_sig2y = np.empty((max_iter,))
plv_sig2x_yres2 = np.empty((max_iter,))
plv_sig1x_sig2y = np.empty((max_iter,))
plv_sig1x_yres2 = np.empty((max_iter,))
plv_sig2x_sig1y = np.empty((max_iter,))
plv_sig2x_yres1 = np.empty((max_iter,))
plv_sig1y_sig2y = np.empty((max_iter,))
plv_yres1_yres2 = np.empty((max_iter,))
seed = np.zeros((max_iter,))
for n_iter in range(max_iter):
    # n_iter = 0
    print(n_iter)
    seed[n_iter] = np.random.randint(low=0, high=2 ** 32, size=(1,))
    np.random.seed(int(seed[n_iter]))
    """    
    dphi_y1 = 0
    dphi_y3 = 0
    dphi_x3 = 0
    dphi_y4 = 0
    """
    dphi_y1 = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift
    dphi_y3 = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift
    dphi_x3 = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift
    dphi_y4 = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift

    # narrow-band components of sig1 and sig2 -------------
    x1 = filtered_randn(8, 12, fs, n_samples)
    if x1_x3_coupling:
        x3 = produce_nm_phase_locked_sig(x1, dphi_x3, 1, 1, [8, 12], fs)
    else:
        x3 = filtered_randn(8, 12, fs, n_samples)

    y1 = produce_nm_phase_locked_sig(sig=x1, phase_lag=dphi_y1, n=1, m=2, wn_base=[8, 12], sfreq=fs)
    y3 = produce_nm_phase_locked_sig(x3, dphi_y3, 1, 2, [8, 12], fs)

    y2 = filtered_randn(16, 24, fs, n_samples)
    if y2_y4_exist:
        if y2_y4_coupling:
            y4 = produce_nm_phase_locked_sig(y2, dphi_y4, 1, 1, [16, 24], fs)
        else:
            if scenario == 5 or scenario == 6 or scenario == 7:
                y4 = produce_nm_phase_locked_sig(sig=x1, phase_lag=dphi_y4, n=1, m=2, wn_base=[8, 12], sfreq=fs)
            else:
                y4 = filtered_randn(16, 24, fs, n_samples)

    # the alpha and beta components of sig1 and sig2 -------------
    x_sig1 = x1
    x_sig2 = x3

    if scenario == 7:
        y_sig1 = y1
        y_sig2 = 0
    else:
        y_sig1 = y1
        y_sig2 = y3

    if y2_y4_exist:
        if scenario == 7:
            y_sig2 = y_sig2 + c_y4 * y4
        else:
            y_sig1 = y_sig1 + c_y2 * y2
            y_sig2 = y_sig2 + c_y4 * y4

    if noisy:
        # noise components ---------------------------------------
        pink_noise_1 = _data_fun_pink_noise(times)[np.newaxis, :]
        pink_noise_2 = _data_fun_pink_noise(times)[np.newaxis, :]

        # SNR adjustment ---------------------------------------------
        factor_x_sig1 = adjust_snr(np.real(x_sig1), pink_noise_1, SNR_alpha, np.array([8, 12]) / fs * 2)
        x_sig1 = x_sig1 / factor_x_sig1

        factor_x_sig2 = adjust_snr(np.real(x_sig2), pink_noise_2, SNR_alpha, np.array([8, 12]) / fs * 2)
        x_sig2 = x_sig2 / factor_x_sig2

        factor_y_sig1 = adjust_snr(np.real(y_sig1), pink_noise_1, SNR_beta, np.array([16, 24]) / fs * 2)
        y_sig1 = y_sig1 / factor_y_sig1

        factor_y_sig2 = adjust_snr(np.real(y_sig2), pink_noise_2, SNR_beta, np.array([16, 24]) / fs * 2)
        y_sig2 = y_sig2 / factor_y_sig2

    # final sig1 and sig1 ---------------------------------------
    sig1 = np.real(x_sig1 + y_sig1)
    sig2 = np.real(x_sig2 + y_sig2)
    # sig2 = y_sig2
    if noisy:
        sig1 += pink_noise_1
        sig2 += pink_noise_2
    # filter sig1 and sig2 in narrow-band ------------------------
    sig1_x = filtfilt(b10, a10, sig1)
    sig1_y = filtfilt(b20, a20, sig1)

    sig2_x = filtfilt(b10, a10, sig2)
    sig2_y = filtfilt(b20, a20, sig2)

    # optimization for sig1 and sig2 -------------
    y_sig1_res = remove_harmonic(sig1_x, sig1_y, fs)

    y_sig2_res = remove_harmonic(sig2_x, sig2_y, fs)


    # PLVs ---------------------------------------
    plv_sig1x_sig1y[n_iter] = compute_plv_with_permtest(sig1_x, sig1_y, 1, 2, fs, plv_type='abs', coh=coh)
    plv_sig1x_yres1[n_iter] = compute_plv_with_permtest(sig1_x, y_sig1_res, 1, 2, fs, plv_type='abs', coh=coh)

    plv_sig2x_sig2y[n_iter] = compute_plv_with_permtest(sig2_x, sig2_y, 1, 2, fs, plv_type='abs', coh=coh)
    plv_sig2x_yres2[n_iter] = compute_plv_with_permtest(sig2_x, y_sig2_res, 1, 2, fs, plv_type='abs', coh=coh)

    plv_sig1x_sig2y[n_iter] = compute_plv_with_permtest(sig1_x, sig2_y, 1, 2, fs, plv_type='abs', coh=coh)
    plv_sig1x_yres2[n_iter] = compute_plv_with_permtest(sig1_x, y_sig2_res, 1, 2, fs, plv_type='abs', coh=coh)

    plv_sig2x_sig1y[n_iter] = compute_plv_with_permtest(sig2_x, sig1_y, 1, 2, fs, plv_type='abs', coh=coh)
    plv_sig2x_yres1[n_iter] = compute_plv_with_permtest(sig2_x, y_sig1_res, 1, 2, fs, plv_type='abs', coh=coh)

    plv_sig1y_sig2y[n_iter] = compute_plv_with_permtest(sig1_y, sig2_y, 1, 1, fs, plv_type='abs', coh=coh)
    plv_yres1_yres2[n_iter] = compute_plv_with_permtest(y_sig1_res, y_sig2_res, 1, 1, fs, plv_type='abs', coh=coh)


# ------------------------------------
# plot
# ------------------------------------
dict = {'seed': seed,
        'plv_sig1x_sig1y': plv_sig1x_sig1y, 'plv_sig1x_yres1': plv_sig1x_yres1,
        'plv_sig2x_sig2y': plv_sig2x_sig2y, 'plv_sig2x_yres2': plv_sig2x_yres2,
        'plv_sig1x_sig2y': plv_sig1x_sig2y, 'plv_sig1x_yres2': plv_sig1x_yres2,
        'plv_sig2x_sig1y': plv_sig2x_sig1y, 'plv_sig2x_yres1': plv_sig2x_yres1,
        'plv_sig1y_sig2y': plv_sig1y_sig2y, 'plv_yres1_yres2': plv_yres1_yres2}
save_pickle('/NOBACKUP/Results/Results_Paper/Toys/toys_' + 'scenario' + str(scenario), dict)

fig = plt.figure()

ax = plt.subplot(231)
plot_boxplot_paired(ax, dict['plv_sig1x_sig1y'], dict['plv_sig1x_yres1'], datapoints=True,
                    labels=['plv(s1_x, s1_y)', 'plv(s1_x, s1_y_res)'])

ax = plt.subplot(232)
plot_boxplot_paired(ax, dict['plv_sig2x_sig2y'], dict['plv_sig2x_yres2'], datapoints=True,
                    labels=['plv(s2_x, s2_y)', 'plv(s2_x, s2_y_res)'])

ax = plt.subplot(233)
plot_boxplot_paired(ax, dict['plv_sig1x_sig2y'], dict['plv_sig1x_yres2'], datapoints=True,
                    labels=['plv(s1_x, s2_y)', 'plv(s1_x, s2_y_res)'])

ax = plt.subplot(234)
plot_boxplot_paired(ax, dict['plv_sig2x_sig1y'], dict['plv_sig2x_yres1'], datapoints=True,
                    labels=['plv(s2_x, s1_y)', 'plv(s2_x, s1_y_res)'])

ax = plt.subplot(235)
plot_boxplot_paired(ax, dict['plv_sig1y_sig2y'], dict['plv_yres1_yres2'], datapoints=True,
                    labels=['plv(s1_y, s2_y)', 'plv(s1_y_res, s2_y_res)'])

fname_fig = op.join(path_save_fig, 'sc' + str(scenario) + '.eps')
fig.savefig(fname_fig, facecolor='white')




scenario = 5
dict = load_pickle('/NOBACKUP/Results/Results_Paper/Toys/toys_scenario' + str(scenario))

data = (dict['plv_sig1x_sig1y'][:, np.newaxis], dict['plv_sig1x_yres1'][:, np.newaxis],
        dict['plv_sig2x_sig2y'][:, np.newaxis], dict['plv_sig2x_yres2'][:, np.newaxis],
        dict['plv_sig1x_sig2y'][:, np.newaxis], dict['plv_sig1x_yres2'][:, np.newaxis],
        dict['plv_sig2x_sig1y'][:, np.newaxis], dict['plv_sig2x_yres1'][:, np.newaxis],
        dict['plv_sig1y_sig2y'][:, np.newaxis], dict['plv_yres1_yres2'][:, np.newaxis])
fig = plt.figure()

plt.boxplot(np.concatenate(data, axis=1), notch=True)

for k in range(max_iter):
    for i1 in range(0, 9, 2):
        plt.plot(np.ones((1, 1)) * (i1+1) + np.random.randn(1, 1) * 0.02, data[i1][k],
                 marker='.', color='lightskyblue', markersize=3)
        plt.plot(np.ones((1, 1)) * (i1+2) + np.random.randn(1, 1) * 0.02, data[i1+1][k],
                 marker='.', color='lightskyblue', markersize=3)
        x = np.array([i1+1, i1+2])
        y = np.array([data[i1][k], data[i1+1][k]])
        plt.plot(x, y, '-', linewidth=.05)

fname_fig = op.join(path_save_fig, 'sc' + str(scenario) + '.eps')
fig.savefig(fname_fig, facecolor='white')


for ii in range(0, 9, 2):
    res = stats.wilcoxon(data[ii].ravel(), data[ii+1].ravel())
    print(res[1], res[1]*5)