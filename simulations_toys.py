"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
script for:
** simulation of toy examples**

In the manuscript figure:
panel A: scenario 1
panel B: scenario 3
panel C: scenario 4
panel D:  scenario 5

In this script more other scenarios are included than the 4 that are presented in teh ms.
-----------------------------------------------------------------------

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210929 \Mina

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
from numpy import pi
import os.path as op
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter

from tools_signal import *
from tools_simulations import data_fun_pink_noise, filtered_randn, produce_nm_phase_locked_sig, adjust_snr
from tools_general import *
from tools_connectivity import *
from scipy import stats
from tools_harmonic_removal import *


def remove_harmonic(ts1, ts2, sfreq, n=2):
    ts1_h = hilbert_(ts1)
    ts1_ = np.abs(ts1_h) * np.exp(1j * n * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(ts2) / np.std(np.real(ts2))

    plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
    ts2_corr = ts2_ - c_opt * np.exp(1j * phi_opt) * ts1_
    return ts2_corr


# --------------------
# Scenario
# --------------------
scenario = 2  # the scenario to be simulated - pls check the header for the scenario descriptions

# in the following we encode the scenario in the parameters identifying which components exist in the signals
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


# --------------------
# general settings
# --------------------

path_save_results = ''  # fill this in, if you wanna save the results. Otherwise leave it as ''
path_save_fig = ''  # fill this in, if you wanna save the figures. Otherwise leave it as ''

# in case you have the seeds for the simulations, fill this in. Otherwise leave it as ''
# path_seeds = ''
path_seeds = ''
# --------------------
# parameters
# --------------------
fs = 256  # sampling frequency
n_samples = int(1 * 60 * fs)  # number of time samples
times = np.arange(0, n_samples)/fs  # the time points - used for plotting purpose
max_iter = 50  # number of interactions
c_y2 = 1  # the weight of y2 in the signal
c_y4 = 1  # the weight of y4
noisy = 1  # if the additive noise should be added to the signals. noisy = 1 --> noisy signals
SNR_alpha = dBinv(5)  # SNR of the alpha band
SNR_beta = dBinv(-5)  # SNR of the beta band
coh = True  # to use coherence or PLV as the connectivity measure

# the filter coefficients
b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')

# the containers for the optimum values of c and phi
c_abs_opt_1 = np.empty((max_iter,))
c_phi_opt_1 = np.empty((max_iter,))
c_abs_opt_2 = np.empty((max_iter,))
c_phi_opt_2 = np.empty((max_iter,))


# the containers for the synchronization values
synch_sig1x_sig1y = np.empty((max_iter,))
synch_sig1x_yres1 = np.empty((max_iter,))
synch_sig2x_sig2y = np.empty((max_iter,))
synch_sig2x_yres2 = np.empty((max_iter,))
synch_sig1x_sig2y = np.empty((max_iter,))
synch_sig1x_yres2 = np.empty((max_iter,))
synch_sig2x_sig1y = np.empty((max_iter,))
synch_sig2x_yres1 = np.empty((max_iter,))
synch_sig1y_sig2y = np.empty((max_iter,))
synch_yres1_yres2 = np.empty((max_iter,))

if path_seeds == '':
    seed = np.random.randint(low=0, high=2 ** 32, size=(max_iter,))
else:
    seed = load_pickle(path_seeds)

for n_iter in range(max_iter):
    # n_iter = 0
    print(n_iter)
    np.random.seed(int(seed[n_iter]))
    """    
    dphi_y1 = 0
    dphi_y3 = 0
    dphi_x3 = 0
    dphi_y4 = 0
    """
    dphi_y1 = pi / 2 * np.random.random(1) + pi / 4  # phase-shift of y1 comparing to the phase warped x1
    dphi_y3 = pi / 2 * np.random.random(1) + pi / 4  # phase-shift of y3 comparing to the phase of warped x3
    dphi_x3 = pi / 2 * np.random.random(1) + pi / 4  # phase-shift of x3 comparing to x1(in case of coupling of x1 & x3)
    dphi_y4 = pi / 2 * np.random.random(1) + pi / 4  # phase-shift of y4 comparing to y2

    # --------------------------------------------------------------
    # generate narrow-band components of sig1 and sig2
    # --------------------------------------------------------------

    # x1 is the alpha component of sig1 - produced by band-pass filtering random noise
    x1 = filtered_randn(8, 12, fs, n_samples)
    if x1_x3_coupling:  # if sig1 and sig2 are coupled, generate x3 by shifting the phase of x1
        x3 = produce_nm_phase_locked_sig(x1, dphi_x3, 1, 1, [8, 12], fs)
    else:  # otherwise, also generate x3 by band-pass filtering random noise
        x3 = filtered_randn(8, 12, fs, n_samples)

    # generate y1 and y3 by phase-warping of x1 and x3, and then adding a phase-shift
    y1 = produce_nm_phase_locked_sig(sig=x1, phase_lag=dphi_y1, n=1, m=2, wn_base=[8, 12], sfreq=fs)
    y3 = produce_nm_phase_locked_sig(x3, dphi_y3, 1, 2, [8, 12], fs)

    # generate a band-pass filtering random noise, it will be saved as y2
    y2 = filtered_randn(16, 24, fs, n_samples)
    if y2_y4_exist:  # if y2 and y4 are contained in sig1 and sig2:
        if y2_y4_coupling:  # if y2 and y4 are coupled, generate y4 by phase-shifting y2
            y4 = produce_nm_phase_locked_sig(y2, dphi_y4, 1, 1, [16, 24], fs)
        else:  # otherwise, if y2 and y4 are not coupled:
            if scenario == 5 or scenario == 6 or scenario == 7:  # if there is a geneuine CFS:
                # use phase warping on x1, to generate y4 cross-frequency coupled to x1
                y4 = produce_nm_phase_locked_sig(sig=x1, phase_lag=dphi_y4, n=1, m=2, wn_base=[8, 12], sfreq=fs)
            else:  # if non of the above cases, generate y4 by band-pass filtering random noise
                y4 = filtered_randn(16, 24, fs, n_samples)

    # the alpha components of sig1 and sig2 ------------
    x_sig1 = x1
    x_sig2 = x3

    # the beta components of sig1 and sig2 ---------------
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

    # --------------------------------------------------------------
    # generate and add the pink noise - SNR is also tuned here
    # --------------------------------------------------------------

    if noisy:
        # generate the noise components ---------
        pink_noise_1 = data_fun_pink_noise(times)[np.newaxis, :]
        pink_noise_2 = data_fun_pink_noise(times)[np.newaxis, :]

        # SNR adjustment ------------
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

    if noisy:  # if noisy add teh pink noise
        sig1 += pink_noise_1
        sig2 += pink_noise_2

    """
    from here on, we pretend that we have the noisy non-sin signal and we wanna use Harmoni to suppress the
    harmonic info
    """
    # --------------------------------------------------------------
    # HARMONI
    # --------------------------------------------------------------

    # filter sig1 and sig2 in narrow-band
    sig1_x = filtfilt(b10, a10, sig1)
    sig1_y = filtfilt(b20, a20, sig1)

    sig2_x = filtfilt(b10, a10, sig2)
    sig2_y = filtfilt(b20, a20, sig2)

    # optimization for sig1 and sig2 -------------
    y_sig1_res = remove_harmonic(sig1_x, sig1_y, fs)

    y_sig2_res = remove_harmonic(sig2_x, sig2_y, fs)

    # compute the synchronization indices
    # we use the absolute coherency as the metric
    synch_sig1x_sig1y[n_iter] = compute_phaseconn_with_permtest(sig1_x, sig1_y, 1, 2, fs, plv_type='abs', coh=coh)
    synch_sig1x_yres1[n_iter] = compute_phaseconn_with_permtest(sig1_x, y_sig1_res, 1, 2, fs, plv_type='abs', coh=coh)

    synch_sig2x_sig2y[n_iter] = compute_phaseconn_with_permtest(sig2_x, sig2_y, 1, 2, fs, plv_type='abs', coh=coh)
    synch_sig2x_yres2[n_iter] = compute_phaseconn_with_permtest(sig2_x, y_sig2_res, 1, 2, fs, plv_type='abs', coh=coh)

    synch_sig1x_sig2y[n_iter] = compute_phaseconn_with_permtest(sig1_x, sig2_y, 1, 2, fs, plv_type='abs', coh=coh)
    synch_sig1x_yres2[n_iter] = compute_phaseconn_with_permtest(sig1_x, y_sig2_res, 1, 2, fs, plv_type='abs', coh=coh)

    synch_sig2x_sig1y[n_iter] = compute_phaseconn_with_permtest(sig2_x, sig1_y, 1, 2, fs, plv_type='abs', coh=coh)
    synch_sig2x_yres1[n_iter] = compute_phaseconn_with_permtest(sig2_x, y_sig1_res, 1, 2, fs, plv_type='abs', coh=coh)

    synch_sig1y_sig2y[n_iter] = compute_phaseconn_with_permtest(sig1_y, sig2_y, 1, 1, fs, plv_type='abs', coh=coh)
    synch_yres1_yres2[n_iter] = compute_phaseconn_with_permtest(y_sig1_res, y_sig2_res, 1, 1, fs, plv_type='abs', coh=coh)


dict1 = {'seed': seed,
         'synch_sig1x_sig1y': synch_sig1x_sig1y, 'synch_sig1x_yres1': synch_sig1x_yres1,
         'synch_sig2x_sig2y': synch_sig2x_sig2y, 'synch_sig2x_yres2': synch_sig2x_yres2,
         'synch_sig1x_sig2y': synch_sig1x_sig2y, 'synch_sig1x_yres2': synch_sig1x_yres2,
         'synch_sig2x_sig1y': synch_sig2x_sig1y, 'synch_sig2x_yres1': synch_sig2x_yres1,
         'synch_sig1y_sig2y': synch_sig1y_sig2y, 'synch_yres1_yres2': synch_yres1_yres2}
if len(path_save_results):
    save_pickle(path_save_results + '/toys_' + 'scenario' + str(scenario), dict1)


# ------------------------------------
# plotting
# ------------------------------------

# fig = plt.figure()
#
# ax = plt.subplot(231)
# plot_boxplot_paired(ax, dict1['plv_sig1x_sig1y'], dic1t['plv_sig1x_yres1'], datapoints=True,
#                     labels=['plv(s1_x, s1_y)', 'plv(s1_x, s1_y_res)'])
#
# ax = plt.subplot(232)
# plot_boxplot_paired(ax, dict1['plv_sig2x_sig2y'], dict1['plv_sig2x_yres2'], datapoints=True,
#                     labels=['plv(s2_x, s2_y)', 'plv(s2_x, s2_y_res)'])
#
# ax = plt.subplot(233)
# plot_boxplot_paired(ax, dict1['plv_sig1x_sig2y'], dict1['plv_sig1x_yres2'], datapoints=True,
#                     labels=['plv(s1_x, s2_y)', 'plv(s1_x, s2_y_res)'])
#
# ax = plt.subplot(234)
# plot_boxplot_paired(ax, dict1['plv_sig2x_sig1y'], dict1['plv_sig2x_yres1'], datapoints=True,
#                     labels=['plv(s2_x, s1_y)', 'plv(s2_x, s1_y_res)'])
#
# ax = plt.subplot(235)
# plot_boxplot_paired(ax, dict1['plv_sig1y_sig2y'], dict1['plv_yres1_yres2'], datapoints=True,
#                     labels=['plv(s1_y, s2_y)', 'plv(s1_y_res, s2_y_res)'])
#
# fname_fig = op.join(path_save_fig, 'sc' + str(scenario) + '.eps')
# fig.savefig(fname_fig, facecolor='white')


# ------------------------------------
# plot by loading your saved results
# ------------------------------------
# if you wanna load your saved results. uncomment the follwoing line:
# dict1 = load_pickle(path_save_results + 'toys_scenario' + str(scenario))

data = (dict1['synch_sig1x_sig1y'][:, np.newaxis], dict1['synch_sig1x_yres1'][:, np.newaxis],
        dict1['synch_sig2x_sig2y'][:, np.newaxis], dict1['synch_sig2x_yres2'][:, np.newaxis],
        dict1['synch_sig1x_sig2y'][:, np.newaxis], dict1['synch_sig1x_yres2'][:, np.newaxis],
        dict1['synch_sig2x_sig1y'][:, np.newaxis], dict1['synch_sig2x_yres1'][:, np.newaxis],
        dict1['synch_sig1y_sig2y'][:, np.newaxis], dict1['synch_yres1_yres2'][:, np.newaxis])

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

if len(path_save_fig):
    fname_fig = op.join(path_save_fig, 'sc' + str(scenario) + '.eps')
    fig.savefig(fname_fig, facecolor='white')

# do the stats
for ii in range(0, 9, 2):
    res = stats.wilcoxon(data[ii].ravel(), data[ii+1].ravel())
    print(res[1], res[1]*5)