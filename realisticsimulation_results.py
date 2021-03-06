"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------

script for:
** analysing the realistic simulations **

Note: the simulation pipeline is not currently shared with these codes. The simulation pipeline produces artificial
realistic EEG data, and then computes the connectivity graphs from them, The connectivity pipeline is similar to the
one used for real EEG data (lemon_conn_bandpass.py). So, the only part missing in this rep is the function to produce
simulated data.

-----------------------------------------------------------------------

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified:
- 20211005 by \Mina

"""

import numpy as np
import os
from numpy import pi
import itertools

import mne
import pandas as pd
import os.path as op
from scipy import stats

from tools_source_space import *
from tools_connectivity import *
from tools_general import *
from scipy.signal import butter, filtfilt
from tools_source_space import *
from matplotlib import pyplot as plt
from tools_connectivity_plot import *



# subjects_dir = '/NOBACKUP/mne_data/'
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'


subject = 'fsaverage'
_oct = '6'
raw_dir = op.join(subjects_dir, subject, 'bem', subject + '-raw.fif')
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
trans_dir = op.join(subjects_dir, subject, 'bem', subject + '-trans.fif')
bem_sol_dir = op.join(subjects_dir, subject, 'bem', subject + '-5120-5120-5120-bem-sol.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-fwd.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-inv.fif')
# path_save_result = '/data/pt_02076/Harmonic_Removal/Simulations/realistic_sc3_sourcespace/'
# path_save_result_root = '/NOBACKUP/Results/Harmoni/Simulations/'
path_save_result_root = '/data/pt_02076/Harmonic_Removal/Simulations/realistic_sim_18_22_broadband_svd/labels_4_apart/'

# simulated data ----------------------
fs = 256
duration = 1 * 60
n_sample = duration * fs

# -----------------------------------------
# set parameters
# -----------------------------------------

# SNR ------------------------
noisy = 1
snr_alpha_dB = 0
snr_beta_dB = -10
snr = dict(alpha=10 ** (snr_alpha_dB / 10), beta=10 ** (snr_beta_dB / 10))
snr_alpha_str = 'min' + str(np.abs(snr_alpha_dB)) if snr_alpha_dB < 0 else 'pos' + str(np.abs(snr_alpha_dB))
snr_beta_str = 'min' + str(np.abs(snr_beta_dB)) if snr_beta_dB < 0 else 'pos' + str(np.abs(snr_beta_dB))
snr_str = 'snr_alpha_' + str(snr_alpha_str) + '_beta_' + str(snr_beta_str)

# Head ----------------------
#parcellation = dict(name='Schaefer2018_100Parcels_7Networks_order', abb='Schaefer100')
parcellation = dict(name='aparc', abb='DK')
labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation['name'])
labels_med = []  # labels[-2:]
labels = labels[:-1]
labels_sorted, idx_sorted = rearrange_labels(labels)  # rearrange labels

n_parc = len(labels)
n_parc_range_prod = list(itertools.product(np.arange(n_parc), np.arange(n_parc)))
fwd = mne.read_forward_solution(fwd_dir)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
src = fwd_fixed['src']

# calculate fwd-----------------------------------------
montage = mne.channels.make_standard_montage('standard_1005')
montage1 = mne.channels.make_standard_montage('biosemi64')
clab = montage1.ch_names
raw_info = mne.create_info(ch_names=clab, sfreq=fs, ch_types=['eeg'] * len(clab))
data1 = np.zeros((len(clab), 1))
raw_temp = mne.io.RawArray(data1, raw_info)
raw_temp.set_montage(montage)
raw_temp.set_eeg_reference(projection=True)
raw_info = raw_temp.info


# inv operator-----------------------------------------

# simulated data ----------------------
fc = 10
total_iter_n = 200
seed1 = np.random.randint(low=0, high=2 ** 32, size=(total_iter_n, 1))
iir_params = dict(order=2, ftype='butter')
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / fs * 2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([18, 22]) / fs * 2, btype='bandpass')

inv_method = 'eLORETA'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scenario 3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Scenario: connections and number of sources ------------------------
scenario = 3
# n_sources = dict(n_source=4, n_nonsin=2, n_alpha=1, n_noise=50)
# # n_connections = dict(n_alpha_conn=1, n_nonsin_conn=1, n_beta_conn=1)
# n_connections = dict(n_alpha_conn=1, n_nonsin_conn=1, n_beta_conn=0, n_cfc_conn=1)
#
# n_source = n_sources['n_source']
# n_alpha = n_sources['n_alpha']
# n_nonsin = n_sources['n_nonsin']
# n_noise = n_sources['n_noise']
# n_beta = n_source - n_alpha - n_nonsin

path_save_result = op.join(path_save_result_root, 'realistic_sc' + str(scenario) + '_sourcespace')
IDs_all = os.listdir(path_save_result)

total_iter_n = len(IDs_all)

# total_iter_n = 200
# first check!----------------------------------------------------------------------
bad_iters = []
for n_iter in range(total_iter_n):  # total_iter_n
    print(n_iter)
    # name_file = combine_names('_', 'scenario', scenario, snr_str, inv_method, 'iter', n_iter)
    name_file = IDs_all[n_iter]
    file_dir = op.join(path_save_result, name_file)
    dict1 = load_pickle(file_dir)
    conn_alpha_orig1 = dict1['conn_alpha_orig1']
    conn_cfc_orig1 = dict1['conn_cfc_orig1']
    alpha_orig1 = np.abs(conn_alpha_orig1[idx_sorted, :][:, idx_sorted])
    cfc_orig1 = np.abs(conn_cfc_orig1[idx_sorted, :][:, idx_sorted])

    ind_lbl_alpha = np.where(conn_alpha_orig1)[0]
    ind_lbl_cfc = np.where(conn_cfc_orig1)
    ind_lbl_alpha_cfc = np.concatenate((ind_lbl_alpha, ind_lbl_cfc[0], ind_lbl_cfc[1]))
    if 63 in ind_lbl_alpha_cfc or 66 in ind_lbl_alpha_cfc:
        bad_iters.append(n_iter)
    # bad_lbl1 = 10 in ind_lbl_alpha_cfc or 11 in ind_lbl_alpha_cfc or 38 in ind_lbl_alpha_cfc or 39 in ind_lbl_alpha_cfc
    # bad_lbl2 = 18 in ind_lbl_alpha_cfc or 19 in ind_lbl_alpha_cfc or 40 in ind_lbl_alpha_cfc or 41 in ind_lbl_alpha_cfc
    # bad_lbl3 = 0 in ind_lbl_alpha_cfc or 1 in ind_lbl_alpha_cfc or 24 in ind_lbl_alpha_cfc or 25 in ind_lbl_alpha_cfc
    # bad_lbl4 = 16 in ind_lbl_alpha_cfc or 17 in ind_lbl_alpha_cfc
    # bad_lbl = bad_lbl1 or bad_lbl2 or bad_lbl3 or bad_lbl4
    # # dist --------------------------------------------------------
    # ind1_cfc = np.where(cfc_orig1)
    # ind1_diff_cfc = np.abs(ind1_cfc[0][0] - ind1_cfc[1][0])
    # ind1_alpha = np.where(alpha_orig1)
    # ind1_diff_alpha = np.abs(ind1_alpha[0][0] - ind1_alpha[1][0])
    # ind_diff = ind1_diff_cfc < 4 or ind1_diff_alpha < 4
    # if ind_diff or bad_lbl:
    #     bad_iters.append(n_iter)

# read all if you need them !!----------------------------------------------------------------------
auc_beta = np.zeros((total_iter_n, 2))
auc_cfc = np.zeros((total_iter_n, 2))
fp_cfc = np.zeros((total_iter_n, 2))
fp_beta = np.zeros((total_iter_n, 2))
auc_alpha_beta = np.zeros((total_iter_n, 2))
auc_cfc_nonsin = np.zeros((total_iter_n, 2))
dist_nonsin_beta_mat = np.zeros((total_iter_n, 2))
dist_nonsin_cfc_mat = np.zeros((total_iter_n, 2))
pattern_distance = np.zeros((total_iter_n, 4))


for n_iter in range(total_iter_n):  # total_iter_n
    print(n_iter)
    # name_file = combine_names('_', 'scenario', scenario, snr_str, inv_method, 'iter', n_iter)
    name_file = IDs_all[n_iter]
    file_dir = op.join(path_save_result, name_file)
    dict1 = load_pickle(file_dir)

    # extract the adjacency matrices ---------------------------------------------
    seed_iter = dict1['seed']
    vert_sigs_ind = dict1['vert_sigs_ind']
    # pattern = dict1['pattern']
    conn_mat_beta_orig = dict1['conn_mat_beta_orig']
    conn_mat_beta_corr = dict1['conn_mat_beta_corr']
    conn_mat_alpha_orig = dict1['conn_mat_alpha_orig']
    conn_alpha_orig1 = dict1['conn_alpha_orig1']
    conn_beta_orig1 = dict1['conn_beta_orig1']
    conn_cfc_orig1 = dict1['conn_cfc_orig1']
    conn_mat_cfc_orig = dict1['conn_mat_cfc_orig']
    conn_mat_cfc_corr = dict1['conn_mat_cfc_corr']
    # conn_cfc_gen_orig1 = dict1['conn_cfc_gen_orig1']
    # conn_beta_gen_orig1 = dict1['conn_beta_gen_orig1']

    # rearrange the labels ------------------------------------------------------
    beta_corr = np.abs(conn_mat_beta_corr[idx_sorted, :][:, idx_sorted])
    beta_orig = np.abs(conn_mat_beta_orig[idx_sorted, :][:, idx_sorted])
    alpha_orig = np.abs(conn_mat_alpha_orig[idx_sorted, :][:, idx_sorted])
    cfc_orig = conn_mat_cfc_orig[idx_sorted, :][:, idx_sorted]
    cfc_corr = conn_mat_cfc_corr[idx_sorted, :][:, idx_sorted]

    beta_orig1 = np.abs(conn_beta_orig1[idx_sorted, :][:, idx_sorted])
    alpha_orig1 = np.abs(conn_alpha_orig1[idx_sorted, :][:, idx_sorted])
    cfc_orig1 = np.abs(conn_cfc_orig1[idx_sorted, :][:, idx_sorted])

    # cfc_gen_orig1 = conn_cfc_gen_orig1[idx_sorted, :][:, idx_sorted]
    # beta_gen_orig1 = conn_beta_gen_orig1[idx_sorted, :][:, idx_sorted]

    # ROC analysis --------------------------------------------------------
    g1 = beta_orig1 - alpha_orig1  # pure beta
    g1[g1 < 0] = 0
    g2 = beta_orig1 - g1  # nonsin conn

    # for scenario 3 -----------
    if scenario == 3 or scenario == 4 or scenario == 5:
        fp1, _ = graph_roc(g1, beta_orig)
        fp2, _ = graph_roc(g1, beta_corr)
        max_auc = np.max(np.append(fp1, fp2)) * 1
        fp_beta[n_iter, 0] = np.sum(0.01 * fp1) / max_auc
        fp_beta[n_iter, 1] = np.sum(0.01 * fp2) / max_auc

        cfc_orig1 = blur_matrix(cfc_orig1, 1)
        fp1, tp1 = graph_roc(cfc_orig1, cfc_orig)
        fp2, tp2 = graph_roc(cfc_orig1, cfc_corr)
        max_auc = np.max(np.append(fp1, fp2)) * np.max(np.append(tp1, tp2))
        auc_cfc[n_iter, 0] = np.sum(np.diff(fp1) * tp1[:-1]) / max_auc
        auc_cfc[n_iter, 1] = np.sum(np.diff(fp2) * tp2[:-1]) / max_auc
    if scenario == 6:
        fp1, tp1 = graph_roc(cfc_gen_orig1, cfc_orig)
        fp2, tp2 = graph_roc(cfc_gen_orig1, cfc_corr)
        max_auc = np.max(np.append(fp1, fp2)) * np.max(np.append(tp1, tp2))
        auc_cfc[n_iter, 0] = np.sum(np.diff(fp1) * tp1[:-1]) / max_auc
        auc_cfc[n_iter, 1] = np.sum(np.diff(fp2) * tp2[:-1]) / max_auc

        fp1, tp1 = graph_roc(beta_gen_orig1, beta_orig)
        fp2, tp2 = graph_roc(beta_gen_orig1, beta_corr)
        max_auc = np.max(np.append(fp1, fp2)) * np.max(np.append(tp1, tp2))
        auc_beta[n_iter, 0] = np.sum(np.diff(fp1) * tp1[:-1]) / max_auc
        auc_beta[n_iter, 1] = np.sum(np.diff(fp2) * tp2[:-1]) / max_auc

        fp1, _ = graph_roc(beta_gen_orig1, beta_orig)
        fp2, _ = graph_roc(beta_gen_orig1, beta_corr)
        max_auc = np.max(np.append(fp1, fp2)) * 1
        fp_beta[n_iter, 0] = np.sum(0.01 * fp1) / max_auc
        fp_beta[n_iter, 1] = np.sum(0.01 * fp2) / max_auc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dict_save_sc3 = {'auc_cfc': auc_cfc, 'fp_beta': fp_beta, 'auc_cfc_nonsin': auc_cfc_nonsin}

file_dir = op.join(path_save_result_root, 'sc3_all_results_200')
# save_pickle(file_dir, dict_save_sc3)
dict_save_sc3 = load_pickle(file_dir)

auc_cfc = dict_save_sc3['auc_cfc']
fp_beta = dict_save_sc3['fp_beta']
auc_cfc_nonsin = dict_save_sc3['auc_cfc_nonsin']

# plot, scenario 3 =================
bad_iters = [] # [22, 30, 32, 37, 50, 71, 97]
plt.figure()
x, y = auc_cfc[:, 0], auc_cfc[:, 1]
iter_inds = np.arange(auc_cfc.shape[0])
x = np.delete(x, bad_iters)
y = np.delete(y, bad_iters)
iter_inds = np.delete(iter_inds, bad_iters)
ax = plt.subplot(221)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('AUC cfc, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))

p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(222)
ax.plot(
    x, (y - x)/ x * 100, "o", color="#b9cfe7", markersize=8,
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
)
plt.grid(True)

# plot_scatterplot_linearReg_bootstrap(x, (y - x)/ x * 100, ax, xlabel='initial value', ylabel='percentage change',
#                                      title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
#                                            'pval = ' + str(np.round(p_value, 3)))

x, y = fp_beta[:, 0], fp_beta[:, 1]
ax = plt.subplot(223)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('fp beta, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))
p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(224)
# plot_scatterplot_linearReg_bootstrap(x, (y - x) / x * 100, ax, xlabel='initial value', ylabel='percentage change',
#                                      title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
#                                            'pval = ' + str(np.round(p_value, 3)))

ax.plot(
    x, (y - x)/ x * 100, "o", color="#b9cfe7", markersize=8,
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
)
plt.grid(True)


# plot, scenario 6 =================
plt.figure()
x, y = auc_cfc[:, 0], auc_cfc[:, 1]
ax = plt.subplot(221)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('AUC cfc, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))

p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(222)
plot_scatterplot_linearReg_bootstrap(x, (y - x)/ x * 100, ax, xlabel='initial value', ylabel='percentage change',
                                     title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
                                           'pval = ' + str(np.round(p_value, 3)))

x, y = auc_beta[:, 0], auc_beta[:, 1]
ax = plt.subplot(223)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('auc beta, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))
p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(224)
plot_scatterplot_linearReg_bootstrap(x, (y - x) / x * 100, ax, xlabel='initial value', ylabel='percentage change',
                                     title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
                                           'pval = ' + str(np.round(p_value, 3)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scenario 2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Scenario: connections and number of sources ------------------------
scenario = 2

n_sources = dict(n_source=6, n_nonsin=2, n_alpha=2, n_noise=50)
n_connections = dict(n_alpha_conn=1, n_nonsin_conn=1, n_beta_conn=1)

n_source = n_sources['n_source']
n_alpha = n_sources['n_alpha']
n_nonsin = n_sources['n_nonsin']
n_noise = n_sources['n_noise']
n_beta = n_source - n_alpha - n_nonsin

path_save_result = op.join(path_save_result_root, 'realistic_sc' + str(scenario) + '_sourcespace')

# ------------------------------------------------------------------------------------------------------------
# total_iter_n = 200

IDs_all = os.listdir(path_save_result)
total_iter_n = len(IDs_all)
auc_beta = np.zeros((total_iter_n, 2))
auc_cfc = np.zeros((total_iter_n, 2))
fp_cfc = np.zeros((total_iter_n, 2))
fp_beta = np.zeros((total_iter_n, 2))
auc_alpha_beta = np.zeros((total_iter_n, 2))
# auc_cfc_nonsin = np.zeros((total_iter_n, 2))
dist_nonsin_beta_mat = np.zeros((total_iter_n, 2))
dist_nonsin_cfc_mat = np.zeros((total_iter_n, 2))
pattern_distance = np.zeros((total_iter_n, 4))
for n_iter in range(total_iter_n):  # total_iter_n
    print(n_iter)
    # name_file = combine_names('_', 'scenario', scenario, snr_str, inv_method, 'iter', n_iter)
    name_file = IDs_all[n_iter]
    file_dir = op.join(path_save_result, name_file)
    dict1 = load_pickle(file_dir)

    # extract the adjacency matrices ---------------------------------------------
    seed_iter = dict1['seed']
    vert_sigs_ind = dict1['vert_sigs_ind']
    # pattern = dict1['pattern']
    conn_mat_beta_orig = dict1['conn_mat_beta_orig']
    conn_mat_beta_corr = dict1['conn_mat_beta_corr']
    conn_mat_alpha_orig = dict1['conn_mat_alpha_orig']
    conn_alpha_orig1 = dict1['conn_alpha_orig1']
    conn_beta_orig1 = dict1['conn_beta_orig1']
    conn_cfc_orig1 = dict1['conn_cfc_orig1']
    conn_mat_cfc_orig = dict1['conn_mat_cfc_orig']
    conn_mat_cfc_corr = dict1['conn_mat_cfc_corr']

    # rearrange the labels ------------------------------------------------------
    beta_corr = np.abs(conn_mat_beta_corr[idx_sorted, :][:, idx_sorted])
    beta_orig = np.abs(conn_mat_beta_orig[idx_sorted, :][:, idx_sorted])
    alpha_orig = np.abs(conn_mat_alpha_orig[idx_sorted, :][:, idx_sorted])
    cfc_orig = conn_mat_cfc_orig[idx_sorted, :][:, idx_sorted]
    cfc_corr = conn_mat_cfc_corr[idx_sorted, :][:, idx_sorted]

    beta_orig1 = np.abs(conn_beta_orig1[idx_sorted, :][:, idx_sorted])
    alpha_orig1 = np.abs(conn_alpha_orig1[idx_sorted, :][:, idx_sorted])
    cfc_orig1 = np.abs(conn_cfc_orig1[idx_sorted, :][:, idx_sorted])

    # ROC analysis --------------------------------------------------------
    g1 = beta_orig1 - alpha_orig1  # pure beta
    g1[g1 < 0] = 0
    g2 = beta_orig1 - g1  # nonsin conn - spurious beta

    # for scenario 2 -----------
    if scenario == 2:
        g1 = blur_matrix(g1, 1)
        fp1, tp1 = graph_roc(g1, beta_orig)
        fp2, tp2 = graph_roc(g1, beta_corr)
        max_auc = np.max(np.append(fp1, fp2)) * np.max(np.append(tp1, tp2))
        auc_beta[n_iter, 0] = np.sum(np.diff(fp1) * tp1[:-1]) / max_auc
        auc_beta[n_iter, 1] = np.sum(np.diff(fp2) * tp2[:-1]) / max_auc

        # fp1, tp1 = graph_roc(g2, beta_orig)
        # fp2, tp2 = graph_roc(g2, beta_corr)
        # max_auc = np.max(np.append(fp1, fp2)) * np.max(np.append(tp1, tp2))
        # auc_alpha_beta[n_iter, 0] = np.sum(np.diff(fp1) * tp1[:-1]) / max_auc
        # auc_alpha_beta[n_iter, 1] = np.sum(np.diff(fp2) * tp2[:-1]) / max_auc

        cfc_groundtruth_mat = np.zeros((n_parc, n_parc))
        fp1, _ = graph_roc(cfc_groundtruth_mat, cfc_orig)
        fp2, _ = graph_roc(cfc_groundtruth_mat, cfc_corr)
        max_auc = np.max(np.append(fp1, fp2)) * 1
        fp_cfc[n_iter, 0] = np.sum(.01 * fp1) / max_auc
        fp_cfc[n_iter, 1] = np.sum(.01 * fp2[:-1]) / max_auc


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dict_save_sc2 = {'fp_cfc': fp_cfc, 'auc_beta': auc_beta, 'auc_alpha_beta': auc_alpha_beta}

file_dir = op.join(path_save_result_root, 'sc2_all_results_200')
# save_pickle(file_dir, dict_save_sc2)

dict_save_sc2 = load_pickle(file_dir)

fp_cfc = dict_save_sc2['fp_cfc']
auc_beta = dict_save_sc2['auc_beta']
auc_alpha_beta = dict_save_sc2['auc_alpha_beta']

# Plot , scenario 2 =======================

plt.figure()
x, y = auc_beta[:, 0], auc_beta[:, 1]
# x = np.delete(x, 20)
# y = np.delete(y, 20)
ax = plt.subplot(221)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('AUC beta, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))

p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(222)
# plot_scatterplot_linearReg_bootstrap(x, (y - x)/ x * 100, ax, xlabel='initial value', ylabel='percentage change',
#                                      title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
#                                            'pval = ' + str(np.round(p_value, 3)))
ax.plot(
    x, (y - x)/ x * 100, "o", color="#b9cfe7", markersize=8,
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
)
plt.grid(True)

x, y = fp_cfc[:, 0], fp_cfc[:, 1]
ax = plt.subplot(223)
plot_boxplot_paired(ax, x, y, ['before', 'after'],
                    paired=True,  violin=False, notch=True, datapoints=True)
res_wilcoxon = stats.wilcoxon(x, y)
plt.title('fp cfc, wilcoxon rank-sum pvalue=' + str(np.round(res_wilcoxon[1], 3)))
p_value, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(224)
# plot_scatterplot_linearReg_bootstrap(x, (y - x)/ x * 100, ax, xlabel='initial value', ylabel='percentage change',
#                                      title='corr=' + str(np.round(r_obs, 3)) + '(' + str(np.round(r0, 3)) + ')' +
#                                            'pval = ' + str(np.round(p_value, 3)))
ax.plot(
    x, (y - x)/ x * 100, "o", color="#b9cfe7", markersize=8,
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
)
plt.grid(True)
