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
from harmoni.harmonitools import harmonic_removal_simple


# --------------------
# general settings
# --------------------

path_save_results = '/data/pt_02076/Harmonic_Removal/Simulations/toys/'  # fill this in, if you wanna save the results. Otherwise leave it as ''
path_save_fig = '/data/pt_02076/Harmonic_Removal/Simulations/toys/'  # fill this in, if you wanna save the figures. Otherwise leave it as ''

# in case you have the seeds for the simulations, fill this in. Otherwise leave it as ''
# path_seeds = ''
path_seeds = ''
# --------------------
# parameters
# --------------------
scenario = 2
fs = 256  # sampling frequency
n_samples = int(60*fs)  # number of time samples
times = np.arange(0, n_samples)/fs  # the time points - used for plotting purpose
max_iter = 50  # number of interactions
c_y2 = 1  # the weight of y2 in the signal
c_y4 = 1  # the weight of y4
noisy = 1  # if the additive noise should be added to the signals. noisy = 1 --> noisy signals
SNR_alpha = dBinv(5)  # SNR of the alpha band
SNR_beta = dBinv(0)  # SNR of the beta band
SNR_gamma = dBinv(-3)  # SNR of the beta band

coh = True  # to use coherence or PLV as the connectivity measure

# the filter coefficients
b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')
b30, a30 = butter(N=2, Wn=np.array([24, 36])/fs*2, btype='bandpass')

# the containers for the optimum values of c and phi
c_abs_opt_1 = np.empty((max_iter,))
c_phi_opt_1 = np.empty((max_iter,))
c_abs_opt_2 = np.empty((max_iter,))
c_phi_opt_2 = np.empty((max_iter,))


# the containers for the synchronization values
synch_y1_z2 = np.empty((max_iter,))
synch_y1_zres2 = np.empty((max_iter,))
synch_y2_z1 = np.empty((max_iter,))
synch_y2_zres1 = np.empty((max_iter,))


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
    dphi_z1 = pi / 2 * np.random.random(1) + pi / 4
    dphi_z3 = pi / 2 * np.random.random(1) + pi / 4
    dphi_prime1 = pi / 2 * np.random.random(1) + pi / 4

    # --------------------------------------------------------------
    # generate narrow-band components of sig1 and sig2
    # --------------------------------------------------------------

    # x1 is the alpha component of sig1 - produced by band-pass filtering random noise
    x1 = filtered_randn(8, 12, fs, n_samples)
    y1 = produce_nm_phase_locked_sig(sig=x1, phase_lag=dphi_y1, n=1, m=2, wn_base=[8, 12], sfreq=fs)
    z1 = produce_nm_phase_locked_sig(x1, dphi_z1, 1, 3, [8, 12], fs)

    if scenario == 1:
        x3 = produce_nm_phase_locked_sig(x1, dphi_x3, 1, 1, [8, 12], sfreq=fs, nonsin_mode=1)
        x3 = filtfilt(b10, a10, x3)
    elif scenario == 2:
        x3 = filtered_randn(8, 12, fs, n_samples)
    y3 = produce_nm_phase_locked_sig(x3, dphi_y3, 1, 2, [8, 12], fs)
    z3 = produce_nm_phase_locked_sig(x3, dphi_z3, 1, 3, [8, 12], fs)

    if scenario == 2:
        x2 = filtered_randn(8, 12, fs, n_samples)
        y3_2 = produce_nm_phase_locked_sig(x2, dphi_prime1, 1, 2, [8, 12], fs, nonsin_mode=1)
        z1_2 = produce_nm_phase_locked_sig(x2, dphi_prime1, 1, 3, [8, 12], fs, nonsin_mode=1)

    # x1 = x1 / np.std(np.real(x1))
    # x3 = x3 / np.std(np.real(x3))
    # y1 = y1 / np.std(np.real(y1))
    # y3 = y3 / np.std(np.real(y3))
    # z1 = z1 / np.std(np.real(z1))
    # z3 = z3 / np.std(np.real(z3))

    pink_noise_1 = data_fun_pink_noise(times)[np.newaxis, :]
    pink_noise_2 = data_fun_pink_noise(times)[np.newaxis, :]

    # SNR adjustment ------------
    factor1 = adjust_snr(np.real(x1), pink_noise_1, SNR_alpha, np.array([8, 12]) / fs * 2)
    x1 /= factor1

    factor1 = adjust_snr(np.real(y1), pink_noise_1, SNR_beta, np.array([16, 24]) / fs * 2)
    y1 /= factor1

    factor1 = adjust_snr(np.real(z1), pink_noise_1, SNR_gamma, np.array([24, 36]) / fs * 2)
    z1 /= factor1

    factor1 = adjust_snr(np.real(x3), pink_noise_2, SNR_alpha, np.array([8, 12]) / fs * 2)
    x3 /= factor1

    factor1 = adjust_snr(np.real(y3), pink_noise_2, SNR_beta, np.array([16, 24]) / fs * 2)
    y3 /= factor1

    factor1 = adjust_snr(np.real(z3), pink_noise_2, SNR_gamma, np.array([24, 36]) / fs * 2)
    z3 /= factor1

    if scenario == 2:
        factor1 = adjust_snr(np.real(y3_2), pink_noise_2, SNR_beta, np.array([16, 24]) / fs * 2)
        y3_2 /= factor1

        factor1 = adjust_snr(np.real(z1_2), pink_noise_1, SNR_gamma, np.array([24, 36]) / fs * 2)
        z1_2 /= factor1
    # ------------------
    if scenario == 1:
        sig1 = x1 + y1 + z1 + pink_noise_1
        sig2 = x3 + y3 + z3 + pink_noise_2
    elif scenario == 2:
        sig1 = x1 + y1 + z1 + 2*z1_2 + pink_noise_1
        sig2 = x3 + y3 + z3 + 2*y3_2 + pink_noise_2

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

    sig1_z = filtfilt(b30, a30, sig1)
    sig2_z = filtfilt(b30, a30, sig2)

    # optimization for sig1 and sig2 -------------
    y_sig1_res = harmonic_removal_simple(sig1_x, sig1_y, fs)
    y_sig2_res = harmonic_removal_simple(sig2_x, sig2_y, fs)

    z_sig1_res = harmonic_removal_simple(sig1_x, sig1_z, fs, n=3)
    z_sig2_res = harmonic_removal_simple(sig2_x, sig2_z, fs, n=3)

    # compute the synchronization indices
    # we use the absolute coherency as the metric
    synch_y1_z2[n_iter] = compute_phaseconn_with_permtest(sig1_y, sig2_z, 2, 3, fs, plv_type='abs', coh=coh)
    synch_y1_zres2[n_iter] = compute_phaseconn_with_permtest(sig1_y, z_sig2_res, 2, 3, fs, plv_type='abs', coh=coh)
    synch_y2_z1[n_iter] = compute_phaseconn_with_permtest(sig2_y, sig1_z, 2, 3, fs, plv_type='abs', coh=coh)
    synch_y2_zres1[n_iter] = compute_phaseconn_with_permtest(sig2_y, z_sig1_res, 2, 3, fs, plv_type='abs', coh=coh)


dict1 = {'seed': seed,
         'synch_y1_z2': synch_y1_z2, 'synch_y1_zres2': synch_y1_zres2,
         'synch_y2_z1': synch_y2_z1, 'synch_y2_zres1': synch_y2_zres1}
if len(path_save_results):
    save_pickle(path_save_results + '/toys_2:3_' + 'scenario' + str(scenario), dict1)


# ------------------------------------
# plot by loading your saved results
# ------------------------------------
# if you wanna load your saved results. uncomment the follwoing line:
# dict1 = load_pickle(path_save_results + 'toys_2to3_scenario' + str(scenario))

data = (dict1['synch_y1_z2'][:, np.newaxis], dict1['synch_y1_zres2'][:, np.newaxis],
        dict1['synch_y2_z1'][:, np.newaxis], dict1['synch_y2_zres1'][:, np.newaxis])

fig = plt.figure()

plt.boxplot(np.concatenate(data, axis=1), notch=True)

for k in range(max_iter):
    for i1 in range(0, 4, 2):
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
for ii in range(0, 4, 2):
    res = stats.wilcoxon(data[ii].ravel(), data[ii+1].ravel())
    print(res[1], res[1]*5)