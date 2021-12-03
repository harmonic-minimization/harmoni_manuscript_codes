"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
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

from matplotlib import pyplot as plt

import numpy as np
from numpy import pi

import mne
from mne.minimum_norm import read_inverse_operator


from tools_connectivity_plot import *
from tools_connectivity import *
from tools_meeg import *
from tools_source_space import *
from tools_general import *
from tools_signal import *
from harmoni.harmonitools import harmonic_removal


# -----------------------------------------
# paths
# -----------------------------------------

# subjects_dir = '/NOBACKUP/mne_data/'
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'


subject = 'fsaverage'
_oct = '6'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-fwd.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-inv.fif')
simulated_data_dir = '../harmoni-supplementary-data/simulated_data/proof_of_concept_data'
raw_dir = '../harmoni-supplementary-data/simulated_data/proofconcept_simulated_sesorspace-raw.fif'

# -----------------------------------------
# set parameters
# -----------------------------------------
iir_params = dict(order=2, ftype='butter')

# Head ----------------------
parcellation = dict(name='aparc', abb='DK')
labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation['name'])
labels_med = []  # labels[-2:]
labels = labels[:-1]

labels_sorted, idx_sorted = rearrange_labels(labels, order='anterior_posterior')  # rearrange labels
n_parc = len(labels)
fwd = mne.read_forward_solution(fwd_dir)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
src = fwd_fixed['src']


# inv operator-----------------------------------------
inv_method = 'eLORETA'
inv_op = read_inverse_operator(inv_op_dir)

# ----------------------------------------
# load simulated data
# -----------------------------------------
simulated_raw = mne.io.read_raw_fif(raw_dir)
fs = simulated_raw.info['sfreq']
dict_simulated_data = load_pickle(simulated_data_dir)
conn_alpha_orig1 = dict_simulated_data['conn_alpha_orig1']
conn_beta_orig1 = dict_simulated_data['conn_beta_orig1']
conn_cfc_orig1 = dict_simulated_data['conn_cfc_orig1']

# --------------------------------------------------------------------
# Compute the source-space data from simulated raw
# --------------------------------------------------------------------
data_alpha = simulated_raw.get_data().copy()
data_alpha, freqs_alpha = morlet_filter(data_alpha, fs, 8, 12, freq_res=0.5)
data_alpha = np.real(data_alpha)
n_freq_alpha = freqs_alpha.shape[0]
parcel_series_alpha_all = [None] * n_freq_alpha

data_beta = simulated_raw.get_data().copy()
data_beta, freqs_beta = morlet_filter(data_beta, fs, 16, 24, freq_res=0.5)
data_beta = np.real(data_beta)
n_freq_beta = freqs_beta.shape[0]
parcel_series_beta_all = [None] * n_freq_beta

# alpha --------------
for n_freq, freq in enumerate(freqs_alpha):
    raw_alpha = mne.io.RawArray(data_alpha[:, n_freq, :], simulated_raw.info)
    raw_alpha.set_eeg_reference(projection=True)
    stc_alpha_raw = mne.minimum_norm.apply_inverse_raw(raw_alpha, inverse_operator=inv_op,
                                                       lambda2=0.05, method=inv_method, pick_ori='normal')
    print(' ******** parcel collapse, alpha, freq %d/%d ************' % (n_freq+1, n_freq_alpha))
    parcel_series_alpha_all[n_freq] = \
        extract_parcel_time_series(stc_alpha_raw.data, labels, src, mode='svd', n_select=1)

# beta --------------
for n_freq, freq in enumerate(freqs_beta):
    raw_beta = mne.io.RawArray(data_beta[:, n_freq, :], simulated_raw.info)
    raw_beta.set_eeg_reference(projection=True)
    stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv_op,
                                                      lambda2=0.05, method=inv_method, pick_ori='normal')
    print(' ******** parcel collapse, beta, freq %d/%d ************' % (n_freq+1, n_freq_beta))
    parcel_series_beta_all[n_freq] = extract_parcel_time_series(stc_beta_raw.data, labels, src, mode='svd', n_select=1)


# --------------------------------------------------------------------
# Harmoni: regress out alpha from beta in each ROI
# --------------------------------------------------------------------
parcel_series_beta_corr_all = parcel_series_beta_all.copy()
for ind_freq_alpha, freq_alpha in enumerate(freqs_alpha):
    print(' ******** correction, freq %d/%d ************' % (ind_freq_alpha+1, n_freq_alpha))
    freq_beta = freq_alpha * 2
    ind_freq_beta = np.where(freqs_beta == freq_beta)[0][0]
    parcel_series_alpha = parcel_series_alpha_all[ind_freq_alpha]
    parcel_series_beta = parcel_series_beta_all[ind_freq_beta]
    parcel_series_beta_corr_all[ind_freq_beta] = \
        harmonic_removal(parcel_series_alpha, parcel_series_beta, int(fs), n=2, mp=True)


ind_freq_beta_regressed = np.empty((n_freq_alpha,))
for ind_freq_alpha, freq_alpha in enumerate(freqs_alpha):
    freq_beta = freq_alpha * 2
    ind_freq_beta_regressed[ind_freq_alpha] = np.where(freqs_beta == freq_beta)[0][0]
ind_freq_beta_regressed = ind_freq_beta_regressed.astype('int')

# --------------------------------------------------------------------
# Compute connectivity
# --------------------------------------------------------------------

# alpha --------------
conn_alpha_orig = np.zeros((n_parc, n_parc, n_freq_alpha))
for ind_freq_alpha, freq_alpha in enumerate(freqs_alpha):
    print(' ******** alpha, conn computation, freq %d/%d ************' % (ind_freq_alpha + 1, n_freq_alpha))
    parcel_series_alpha = parcel_series_alpha_all[ind_freq_alpha]
    conn_alpha_orig[:, :, ind_freq_alpha] = \
        compute_conn_2D_parallel(parcel_series_alpha, parcel_series_alpha, 1, 1, fs, 'imag')
conn_mat_alpha_orig = np.mean(np.abs(conn_alpha_orig), axis=2)
print('--------alpha within-freq computed--------')

# beta -------------
conn_beta_corr = np.zeros((n_parc, n_parc, n_freq_beta))
conn_beta_orig = np.zeros((n_parc, n_parc, n_freq_beta))
for ind_freq_beta, freq_beta in enumerate(freqs_beta):
    print(' ******** beta, conn computation, freq %d/%d ************' % (ind_freq_beta + 1, n_freq_beta))
    parcel_series_beta = parcel_series_beta_all[ind_freq_beta]
    parcel_series_beta_corr = parcel_series_beta_corr1_all[ind_freq_beta]
    conn_beta_corr[:, :, ind_freq_beta] = \
        compute_conn_2D_parallel(parcel_series_beta_corr, parcel_series_beta_corr, 1, 1, fs, 'imag')
    conn_beta_orig[:, :, ind_freq_beta] = \
        compute_conn_2D_parallel(parcel_series_beta, parcel_series_beta, 1, 1, fs, 'imag')

conn_mat_beta_corr = np.mean(np.abs(conn_beta_corr[:, :, ind_freq_beta_regressed]), axis=2)
conn_mat_beta_orig = np.mean(np.abs(conn_beta_orig), axis=2)
print('--------beta within-freq computed--------')

# CFC ---------------
conn_cfc_orig = np.zeros((n_parc, n_parc, n_freq_alpha))
conn_cfc_corr_orig = np.zeros((n_parc, n_parc, n_freq_alpha))
for ind_freq_alpha, freq_alpha in enumerate(freqs_alpha):
    print(' ******** alpha, conn computation, freq %d/%d ************' % (ind_freq_alpha + 1, n_freq_alpha))
    parcel_series_alpha = parcel_series_alpha_all[ind_freq_alpha]
    for ind_freq_beta in ind_freq_beta_regressed:
        parcel_series_beta = parcel_series_beta_all[ind_freq_beta]
        parcel_series_beta_corr = parcel_series_beta_corr_all[ind_freq_beta]
        conn_cfc_orig[:, :, ind_freq_alpha] += \
            compute_conn_2D_parallel(parcel_series_alpha, parcel_series_beta, 1, 2, fs, 'abs') / n_freq_alpha
        conn_cfc_corr_orig[:, :, ind_freq_alpha] += \
            compute_conn_2D_parallel(parcel_series_alpha, parcel_series_beta_corr, 1, 2, fs, 'abs') / n_freq_alpha

conn_mat_cfc_orig = np.mean(conn_cfc_orig, axis=-1)
conn_mat_cfc_corr_orig = np.mean(conn_cfc_corr_orig, axis=-1)
print('--------cfc  within-freq computed--------')

# --------------------------------------------------------------------
# rearrange label
# --------------------------------------------------------------------

beta_orig1 = np.abs(conn_beta_orig1[idx_sorted, :][:, idx_sorted])
alpha_orig1 = np.abs(conn_alpha_orig1[idx_sorted, :][:, idx_sorted])
cfc_orig1 = np.abs(conn_cfc_orig1[idx_sorted, :][:, idx_sorted])

beta_orig = np.abs(conn_mat_beta_orig[idx_sorted, :][:, idx_sorted])
alpha_orig = np.abs(conn_mat_alpha_orig[idx_sorted, :][:, idx_sorted])

cfc_orig = conn_mat_cfc_orig[idx_sorted, :][:, idx_sorted]

cfc_corr = conn_mat_cfc_corr_orig[idx_sorted, :][:, idx_sorted]
beta_corr = np.abs(conn_mat_beta_corr[idx_sorted, :][:, idx_sorted])


# --------------------------------------------------------------------
# plot networks
# --------------------------------------------------------------------
fig = plt.figure()
plot_mne_circular_connectivity(alpha_orig1, labels_sorted, perc_conn=0.01, fig=fig, cfc=False, subplot=111,
                               fig_title='alpha', node_name=True, vmax=1, vmin=0, colormap='Blues',
                               facecolor='white', textcolor='black')
plot_connectivity_bipartite_2_prime(cfc_orig1, labels_sorted, 99, 'Blues', fig_title='CFC original', vmin=0, vmax=1,
                                    only_lbl=None, arrange='other', edge_thresh=None)

fig = plt.figure(num=None, figsize=(8, 4), facecolor='white')
plot_mne_circular_connectivity(alpha_orig, labels_sorted, perc_conn=0.01, fig=fig, cfc=False, subplot=131,
                               fig_title='alpha', node_name=False, vmax=1, vmin=0, colormap='Blues',
                               facecolor='white', textcolor='black')
plot_mne_circular_connectivity(beta_orig, labels_sorted, perc_conn=0.01, fig=fig, cfc=False, subplot=132,
                               fig_title='beta', node_name=False, vmax=1, vmin=0, colormap='Blues',
                               facecolor='white', textcolor='black')
plot_mne_circular_connectivity(beta_corr, labels_sorted, perc_conn=0.01, fig=fig, cfc=False, subplot=133,
                               fig_title='beta corr', node_name=False, vmax=1, vmin=0, colormap='Blues',
                               facecolor='white', textcolor='black')

plot_connectivity_bipartite_2_prime(cfc_orig, labels_sorted, 99, 'Blues', fig_title='CFC before', vmin=0, vmax=1,
                                    only_lbl=None, arrange='other', edge_thresh=None)

plot_connectivity_bipartite_2_prime(cfc_corr, labels_sorted, 99, 'Blues', fig_title='CFC after', vmin=0, vmax=1,
                                    only_lbl=None, arrange='other', edge_thresh=None)