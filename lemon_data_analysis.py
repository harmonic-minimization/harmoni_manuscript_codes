"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Jaunli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
INSERT THE DOIs
-----------------------------------------------------------------------
script for:
** proof of concept example **

-----------------------------------------------------------------------

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210930 by \Mina

-----------------------------------------------------------------------
-----------------------------------------------------------------------
"""

import os.path as op
import os
import itertools
from operator import itemgetter
import multiprocessing
from functools import partial
import time
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import pi

import scipy.stats as stats



from scipy.signal import butter


from tools_general import *
from tools_source_space import *
from tools_connectivity import *
from tools_connectivity_plot import *
from tools_harmonic_removal import *

# directories and settings -----------------------------------------------------
# subjects_dir = '/NOBACKUP/mne_data/'
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'
subject = 'fsaverage'
_oct = '6'
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
inv_method = 'eLORETA'
dir_graphs = '/NOBACKUP/Results/lemon_processed_data/graphs/'
condition = 'EC'
dir_adjmat = op.join(dir_graphs, 'AdjacencyMatrices_coh_with_permtest', condition)
dir_raw_set = '/data/pt_nro109/Share/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/'
    # '/NOBACKUP/Data/lemon/LEMON_prep/'
# -----------------------------------------------------
# read the parcellation
# -----------------------------------------------------
# parcellation = dict(name='aparc', abb='DK')  # Desikan-Killiany
parcellation = dict(name='Schaefer2018_100Parcels_7Networks_order', abb='Schaefer100')
labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation['name'])
labels = labels[:-2]
# labels = labels[:-1]

labels_sorted, idx_sorted = rearrange_labels(labels)  # rearrange labels
labels_sorted2, idx_sorted2 = rearrange_labels_network(labels)  # rearrange labels
labels_network_sorted, idx_lbl_sort = rearrange_labels_network(labels_sorted)
n_parc = len(labels)
n_parc_range_prod = list(itertools.product(np.arange(n_parc), np.arange(n_parc)))

# read forward solution ---------------------------------------------------
fwd = mne.read_forward_solution(fwd_dir)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
n_vox = leadfield.shape[1]
src = fwd_fixed['src']
sfreq = 250
vertices = [src[0]['vertno'], src[1]['vertno']]
iir_params = dict(order=2, ftype='butter')
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / sfreq * 2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24]) / sfreq * 2, btype='bandpass')
# -----------------------------------------------------
# read the graphs
# -----------------------------------------------------
# ids1 = select_subjects('young', 'male', 'right', meta_file_path)
list_ids = listdir_restricted(dir_adjmat, 'sub-')
ids = [list_ids1[:10] for list_ids1 in list_ids]
ids = np.unique(np.sort(ids))
n_subj = len(ids)

# -----------------------------------------------------
# FIGURE 8-A
# 1:2 coh all subjects source-space

# This part is commented because it takes a lot of time  - just uncomment it if you wanna run it
# -----------------------------------------------------
# plv_src = np.zeros((n_vox, n_subj))
# for i_subj, subj in enumerate(ids):
#     print(i_subj, '**************')
#     raw_name = op.join(dir_raw_set, subj + '_EC.set')
#     raw = read_eeglab_standard_chanloc(raw_name)
#     data_raw = raw.get_data()
#     inv_sol, inv_op, inv = extract_inv_sol(data_raw.shape, fwd, raw.info)
#     fwd_ch = fwd_fixed.ch_names
#     raw_ch = raw.info['ch_names']
#     ind = [fwd_ch.index(ch) for ch in raw_ch]
#     leadfield_raw = leadfield[ind, :]
#     sfreq = raw.info['sfreq']
#
#     # alpha sources --------
#     raw_alpha = raw.copy()
#     raw_alpha.load_data()
#     raw_alpha.filter(l_freq=8, h_freq=12, method='iir', iir_params=iir_params)
#     raw_alpha.set_eeg_reference(projection=True)
#     stc_alpha_raw = mne.minimum_norm.apply_inverse_raw(raw_alpha, inverse_operator=inv,
#                                                        lambda2=0.05, method=inv_method, pick_ori='normal')
#     # beta sources --------
#     raw_beta = raw.copy()
#     raw_beta.load_data()
#     raw_beta.filter(l_freq=16, h_freq=24, method='iir', iir_params=iir_params)
#     raw_beta.set_eeg_reference(projection=True)
#     stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv,
#                                                       lambda2=0.1, method=inv_method, pick_ori='normal')
#
#     for i_parc, label1 in enumerate(labels):
#         print(i_parc)
#         parc_idx, _ = label_idx_whole_brain(src, label1)
#         data1 = stc_alpha_raw.data[parc_idx, :]
#         data2 = stc_beta_raw.data[parc_idx]
#         plv_src[parc_idx, i_subj] = compute_phase_connectivity(data1, data2, 1, 2, measure='coh', axis=1, type1='abs')
#
# save_json_from_numpy('/NOBACKUP/Results/lemon_processed_data/parcels/plv_vertices_all-subj', plv_src)
#
# stc_new = mne.SourceEstimate(np.mean(plv_src, axis=-1, keepdims=True), vertices, tmin=0, tstep=0.01, subject='fsaverage')
# stc_new.plot(subject='fsaverage', subjects_dir=subjects_dir, time_viewer=True, hemi='split', background='white',
#                  surface='pial')

# containers for the graphs  -----------------------------------

# thresholded at 95%
conn12_thresh_all = np.zeros((n_parc, n_parc, n_subj))
conn12_corr_thresh_all = np.zeros((n_parc, n_parc, n_subj))
conn1_thresh_all = np.zeros((n_parc, n_parc, n_subj))
conn2_thresh_all = np.zeros((n_parc, n_parc, n_subj))
conn2_corr_thresh_all = np.zeros((n_parc, n_parc, n_subj))

# not thresholded
conn1_all = np.zeros((n_parc, n_parc, n_subj))
conn2_all = np.zeros((n_parc, n_parc, n_subj))
conn2_corr_all = np.zeros((n_parc, n_parc, n_subj))
conn12_all = np.zeros((n_parc, n_parc, n_subj))
conn12_corr_all = np.zeros((n_parc, n_parc, n_subj))

# containers for centrality measures  -----------------------------------
conn12_symm_idx = np.zeros((n_subj, 2))  # asymmetry index container

ind_triu = np.triu_indices(n_parc, k=1)
ind_diag = np.diag_indices(n_parc)

# ************************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BIG NOTE: graph adjacency matrices are rearranged here --> labels_sorted
# ************************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for i_subj, subj in enumerate(ids):
    print(i_subj)
    # pickle_name = op.join(dir_adjmat, subj + '-alpha-alpha-' + condition)
    pickle_name = op.join(dir_adjmat, subj + '-alpha-alpha')
    conn1, pval1, pval1_ = load_pickle(pickle_name)
    # pickle_name = op.join(dir_adjmat, subj + '-beta-beta-' + condition)
    pickle_name = op.join(dir_adjmat, subj + '-beta-beta')
    conn2, pval2, _ = load_pickle(pickle_name)
    # pickle_name = op.join(dir_adjmat, subj + '-beta-beta-corr-' + condition)
    pickle_name = op.join(dir_adjmat, subj + '-beta-beta-corr')
    conn2_corr, pval2_corr, _ = load_pickle(pickle_name)
    # pickle_name = op.join(dir_adjmat, subj + '-alpha-beta-' + condition)
    pickle_name = op.join(dir_adjmat, subj + '-alpha-beta')
    conn12, pval12, _ = load_pickle(pickle_name)
    # pickle_name = op.join(dir_adjmat, subj + '-alpha-beta-corr-' + condition)
    pickle_name = op.join(dir_adjmat, subj + '-alpha-beta-corr')
    conn12_corr, pval12_corr, _ = load_pickle(pickle_name)

    # save the original graphs
    conn1_all[:, :, i_subj] = conn1[idx_sorted, :][:, idx_sorted]
    conn2_all[:, :, i_subj] = conn2[idx_sorted, :][:, idx_sorted]
    conn2_corr_all[:, :, i_subj] = conn2_corr[idx_sorted, :][:, idx_sorted]
    conn12_all[:, :, i_subj] = conn12[idx_sorted, :][:, idx_sorted]
    conn12_corr_all[:, :, i_subj] = conn12_corr[idx_sorted, :][:, idx_sorted]

    # asymmetry index from original graphs
    conn12_symm_idx[i_subj, 0] = np.linalg.norm((conn12 - conn12.T)) / np.sqrt(2) / np.linalg.norm(conn12)
    conn12_symm_idx[i_subj, 1] = np.linalg.norm((conn12_corr - conn12_corr.T)) / np.sqrt(2) / np.linalg.norm(conn12_corr)




# -----------------------------------------------------
# FIGURE 8-B & C & D & E
# means
# -----------------------------------------------------
net_mean_before = np.mean(conn12_all, axis=-1)
net_mean_after = np.mean(conn12_corr_all, axis=-1)

# zscore all -------------------------
conn12_all_z = np.zeros_like(conn12_all)
conn12_corr_all_z = np.zeros_like(conn12_corr_all)

for i_subj in range(n_subj):
    conn12_all_z[:, :, i_subj] = zscore_matrix(conn12_all[:, :, i_subj])
    conn12_corr_all_z[:, :, i_subj] = zscore_matrix(conn12_corr_all[:, :, i_subj])

# difference -------------------------
conn12_diff_z = conn12_corr_all_z - conn12_all_z
conn12_diff_z_mean = np.mean(conn12_diff_z, axis=-1)
conn12_diff_z_mean_pos = conn12_diff_z_mean.copy()
conn12_diff_z_mean_pos[conn12_diff_z_mean < 0] = 0
conn12_diff_z_mean_neg = conn12_diff_z_mean.copy()
conn12_diff_z_mean_neg[conn12_diff_z_mean > 0] = 0
conn12_diff_z_mean_neg = np.abs(conn12_diff_z_mean_neg)

# test the significance of change in each connection -------------------------
pvalue_zscores = np.zeros((n_parc, n_parc))
statistics_all = np.zeros((n_parc, n_parc))
for i1 in range(n_parc):
    for i2 in range(n_parc):
        statistics_all[i1, i2], pvalue_zscores[i1, i2] = stats.ttest_rel(conn12_all_z[i1, i2, :],
                                                                         conn12_corr_all_z[i1, i2, :])
ind_nonsig = pvalue_zscores > 0.05 / n_parc ** 2  # bonferroni correction
pvalue2_zscores = np.ones((n_parc, n_parc))
pvalue2_zscores[ind_nonsig] = 0


# plot the networks -------------------------
# the mean before Harmoni - FIGURE 8 - B
_, _ = plot_connectivity_bipartite_2_prime(net_mean_before,
                                           labels_sorted, 0, edge_cmp='Blues',
                                           fig_title='mean before',
                                           only_lbl=None, arrange='network')
# the mean after Harmoni -  FIGURE 8 - C
_, _ = plot_connectivity_bipartite_2_prime(net_mean_after,
                                           labels_sorted, 0, edge_cmp='Blues',
                                           fig_title='mean after',
                                           only_lbl=None, arrange='network')

# the positive difference - significant connections  - FIGURE 8 - D
_, _ = plot_connectivity_bipartite_2_prime(conn12_diff_z_mean_pos * pvalue2_zscores,
                                           labels_sorted, 0, edge_cmp='Purples',
                                           fig_title='pos difference',
                                           only_lbl=None, arrange='network')

# the negative difference - significant connections -  FIGURE 8 - E
 _, _ = plot_connectivity_bipartite_2_prime(conn12_diff_z_mean_neg * pvalue2_zscores,
                                           labels_sorted, 0, edge_cmp='Greens',
                                           fig_title='pos difference',
                                           only_lbl=None, arrange='network')

# -----------------------------------
# FIGURE 9
# test the decrease
# -----------------------------------
pvalue_all = np.zeros((n_parc, n_parc))
tvalue_all = np.zeros((n_parc, n_parc))

for i1 in range(n_parc):
    for i2 in range(n_parc):
        tvalue_all[i1, i2], pvalue_all[i1, i2] = stats.ttest_rel(conn12_corr_all[i1, i2, :],
                                                                  conn12_all[i1, i2, :])

ind_nonsig = pvalue_all > 0.05 / n_parc**2  # bonferroni correction
pvalue2 = np.ones((n_parc, n_parc))
pvalue2[ind_nonsig] = 0

print('min t=', np.min(tvalue_all[pvalue2 == 1]), '- max t=', np.max(tvalue_all[pvalue2 == 1]))
print('min pval=', np.min(pvalue_all[pvalue2 == 1] * n_parc**2), '- max pval=', np.max(pvalue_all[pvalue2 == 1] * n_parc**2))

cmap_blues = plt.get_cmap('Blues')
cmap_blues = cmap_blues.reversed()

cmap_oranges = plt.get_cmap('Oranges')
cmap_oranges = cmap_oranges.reversed()

matplotlib.rcParams.update({'font.size': 20})

plt.figure()
net = np.mean(conn12_corr_all - conn12_all, axis=-1)
ax = plt.subplot(121)
cax = ax.matshow(pvalue2 * net, cmap=cmap_blues)
plt.colorbar(cax)
ax = plt.subplot(122)
cax = ax.matshow((pvalue2 * pvalue_all * n_parc**2), cmap=cmap_oranges, norm=matplotlib.colors.LogNorm())
plt.colorbar(cax)




# -------------------------------
# FIGURE 10
# Plot change in asymmetry CFC
# --------------------------------

plt.figure()
ax = plt.subplot(121)
plot_boxplot_paired(ax, conn12_symm_idx[:, 0], conn12_symm_idx[:, 1],
                    labels=['before Harmoni', 'after Harmoni'], datapoints=True)
plt.ylabel('asymmetryness of CFC connectivity')
t_asymm, p_asymm = stats.ttest_rel(conn12_symm_idx[:, 1],  conn12_symm_idx[:, 0])
plt.title('ttest, ' + strround(p_asymm))

x, y = conn12_symm_idx[:, 0] , conn12_symm_idx[:, 1]
perc_change = np.diff(conn12_symm_idx, axis=1).ravel() / conn12_symm_idx[:, 0]*100
pval_perc_change, zscore, r_obs, r0, pval_init, k = significance_perc_increase(x, y)
ax = plt.subplot(122)
plot_scatterplot_linearReg_bootstrap(x, perc_change, ax, xlabel='initial value', ylabel='percentage change',
                                     title='corr=' + strround(r_obs, 3) + '(' + strround(r0) + ')' +
                                           'pval = ' + strround(pval_perc_change))

print('t-asymm=', t_asymm, 'p-asymm=', p_asymm)
print('pval-perc-change=', pval_perc_change, 'r-obs=', r_obs, 'H0=', r0)