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

last modified: 20210929 by \Mina

-----------------------------------------------------------------------
-----------------------------------------------------------------------
"""

import mne
import pandas as pd
import os.path as op

import numpy as np
from numpy import pi
from scipy.signal import butter, filtfilt
from scipy import signal, stats
from matplotlib import pyplot as plt

from operator import itemgetter
import time

from mne.minimum_norm import read_inverse_operator

from tools_connectivity import *
from tools_general import *
from tools_simulation import *
from tools_source_space import *
from tools_signal import *
from tools_harmonic_removal import *
from tools_multivariate import *
from tools_connectivity_plot import *

# -----------------------------------------
# functions
# -----------------------------------------


def compute_conn(n, m, fs, plv_type, signals):
    x = signals[0]
    y = signals[1]
    conn = np.mean(compute_plv_with_permtest(x, y, n, m, fs, plv_type=plv_type))
    return conn


def compute_conn_2D_parallel(ts_list1, ts_list2, n, m, fs, plv_type):
    list_prod = list(itertools.product(ts_list1, ts_list2))
    pool = multiprocessing.Pool()
    func = partial(compute_conn, n, m, fs, plv_type)
    conn_mat_beta_corr_list = pool.map(func, list_prod)
    pool.close()
    pool.join()
    conn_mat = np.asarray(conn_mat_beta_corr_list).reshape((len(ts_list1), len(ts_list2)))
    return conn_mat


def compute_cfc_reg(ts_l, ts_h, n, m):
    n1 = len(ts_l)
    conn = np.empty((n1, n1))
    for i1 in range(n1):
        l1 = ts_l[i1]
        h1 = ts_h[i1]
        for i2 in range(n1):
            h2 = ts_h[i2]
            c1 = optimize_c_gridsearch(h2, h1, fs, coh=True, return_all=False)
            h2res = h2 - c1 * h1
            conn[i1][i2] = compute_plv_with_permtest(l1, h2res, n, m, fs, plv_type='abs')[0][0]
    return conn

def compute_original_conn(n_parc, source_stc_orig, vert_sigs_ind):
    conn_beta_orig1 = np.zeros((n_parc, n_parc))
    conn_alpha_orig1 = np.zeros((n_parc, n_parc))
    vert_source_stc_orig = np.concatenate(source_stc_orig.vertices)
    vert_beta = vert_source_stc_orig[vert_sigs_ind['beta']]
    vert_alpha = vert_source_stc_orig[vert_sigs_ind['alpha']]
    vert_nonsin = vert_source_stc_orig[vert_sigs_ind['nonsin']]
    for edge_beta in vert_sigs_ind['conn_ind_beta']:
        vert1 = vert_beta[edge_beta[0]]
        vert2 = vert_beta[edge_beta[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_beta_orig1[n1, n2], conn_beta_orig1[n2, n1] = 1, 1

    for edge_alpha in vert_sigs_ind['conn_ind_alpha']:
        vert1 = vert_alpha[edge_alpha[0]]
        vert2 = vert_alpha[edge_alpha[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_alpha_orig1[n1, n2], conn_alpha_orig1[n2, n1] = 1, 1

    for edge_nonsin in vert_sigs_ind['conn_ind_nonsin']:
        vert1 = vert_nonsin[edge_nonsin[0]]
        vert2 = vert_nonsin[edge_nonsin[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_beta_orig1[n1, n2], conn_beta_orig1[n2, n1] = 1, 1
        conn_alpha_orig1[n1, n2], conn_alpha_orig1[n2, n1] = 1, 1
    return conn_alpha_orig1, conn_beta_orig1


def compute_original_conn_cfc(n_parc, source_stc_orig, vert_sigs_ind, labels):
    conn_beta_orig1 = np.zeros((n_parc, n_parc))
    conn_alpha_orig1 = np.zeros((n_parc, n_parc))
    conn_cfc_orig1 = np.zeros((n_parc, n_parc))
    vert_source_stc_orig = np.concatenate(source_stc_orig.vertices)
    vert_beta = vert_source_stc_orig[vert_sigs_ind['beta']]
    vert_alpha = vert_source_stc_orig[vert_sigs_ind['alpha']]
    vert_nonsin = vert_source_stc_orig[vert_sigs_ind['nonsin']]
    vert_alpha_nonsin = np.append(vert_alpha, vert_nonsin)
    for edge_beta in vert_sigs_ind['conn_ind_beta']:
        vert1 = vert_beta[edge_beta[0]]
        vert2 = vert_beta[edge_beta[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_beta_orig1[n1, n2], conn_beta_orig1[n2, n1] = 1, 1

    for edge_alpha in vert_sigs_ind['conn_ind_alpha']:
        vert1 = vert_alpha[edge_alpha[0]]
        vert2 = vert_alpha[edge_alpha[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_alpha_orig1[n1, n2], conn_alpha_orig1[n2, n1] = 1, 1

    for edge_nonsin in vert_sigs_ind['conn_ind_nonsin']:
        vert1 = vert_nonsin[edge_nonsin[0]]
        vert2 = vert_nonsin[edge_nonsin[1]]
        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_beta_orig1[n1, n2], conn_beta_orig1[n2, n1] = 1, 1
        conn_alpha_orig1[n1, n2], conn_alpha_orig1[n2, n1] = 1, 1

    for edge_cfc in vert_sigs_ind['conn_ind_cfc']:
        vert1 = vert_alpha_nonsin[edge_cfc[0]]
        vert2 = vert_beta[edge_cfc[1]]

        h1 = np.intersect1d(source_stc_orig.vertices[0], vert1)
        h2 = np.intersect1d(source_stc_orig.vertices[0], vert2)
        hemi1 = 'lh' if len(h1) else 'rh'
        hemi2 = 'lh' if len(h2) else 'rh'
        n1 = find_label_of_a_vertex_hemi(labels, vert1, hemi1)
        n2 = find_label_of_a_vertex_hemi(labels, vert2, hemi2)
        # print(n1, n2)
        conn_cfc_orig1[n1, n2] = 1

    return conn_alpha_orig1, conn_beta_orig1, conn_cfc_orig1


# -----------------------------------------
# paths
# -----------------------------------------

# subjects_dir = '/NOBACKUP/mne_data/'
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'


subject = 'fsaverage'
_oct = '6'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-fwd.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-64ch-inv.fif')


# simulated data ----------------------
fs = 256
duration = 1 * 60
n_sample = duration * fs

# -----------------------------------------
# set parameters
# -----------------------------------------

# Scenario: connections and number of sources ------------------------
scenario = 0
n_sources = dict(n_source=4, n_nonsin=2, n_alpha=1, n_noise=50)
n_connections = dict(n_alpha_conn=1, n_nonsin_conn=1, n_beta_conn=0, n_cfc_conn=1)


n_source = n_sources['n_source']
n_alpha = n_sources['n_alpha']
n_nonsin = n_sources['n_nonsin']
n_noise = n_sources['n_noise']
n_beta = n_source - n_alpha - n_nonsin

# SNR ------------------------
noisy = 1
snr_alpha_dB = 0
snr_beta_dB = -10
snr = dict(alpha=10 ** (snr_alpha_dB / 10), beta=10 ** (snr_beta_dB / 10))
snr_alpha_str = 'min' + str(np.abs(snr_alpha_dB)) if snr_alpha_dB < 0 else 'pos' + str(np.abs(snr_alpha_dB))
snr_beta_str = 'min' + str(np.abs(snr_beta_dB)) if snr_beta_dB < 0 else 'pos' + str(np.abs(snr_beta_dB))
snr_str = 'snr_alpha_' + str(snr_alpha_str) + '_beta_' + str(snr_beta_str)

# Head ----------------------
parcellation = dict(name='aparc', abb='DK')
labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation['name'])
labels_med = []  # labels[-2:]
labels = labels[:-1]

labels_sorted, idx_sorted = rearrange_labels(labels, order='anterior_posterior')  # rearrange labels
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
inv_method = 'eLORETA'
inv_op = read_inverse_operator(inv_op_dir)

# simulated data ----------------------
fc = 10
total_iter_n = 200
iir_params = dict(order=2, ftype='butter')
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / fs * 2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24]) / fs * 2, btype='bandpass')

# -----------------------------------------
# main loop
# -----------------------------------------

t_start_iter = time.time()

seed_iter = 1939353649 # np.random.randint(low=0, high=2 ** 32)
# 2325241275, 2935603725, 1939353649, 3508868409
np.random.seed(seed=seed_iter)
label_source_ind = [4, 5, 14, 15]
labels_source = itemgetter(*label_source_ind)(labels)

simulated_raw, source_stc, source_stc_orig, vert_sigs_ind = \
    simulate_stc_conn_v1(n_sources, n_connections, n_sample, src, fs, fwd_fixed, raw_info, leadfield, snr,
                         labels_med, labels_=labels_source, random_state=None, location='center',
                         subject=subject, subjects_dir=subjects_dir, surf='sphere')

# --------------------------------------------------------------------
# Compute the ground-truth conn matrices
# --------------------------------------------------------------------
conn_alpha_orig1, conn_beta_orig1, conn_cfc_orig1 = \
    compute_original_conn_cfc(n_parc, source_stc_orig, vert_sigs_ind, labels)
print(np.where(conn_alpha_orig1))
# --------------------------------------------------------------------
# Compute the source-space data from simulated raw
# --------------------------------------------------------------------
# alpha sources --------
raw_alpha = simulated_raw.copy()
raw_alpha.load_data()
raw_alpha.filter(l_freq=8, h_freq=12, method='iir', iir_params=iir_params)
raw_alpha.set_eeg_reference(projection=True)
stc_alpha_raw = mne.minimum_norm.apply_inverse_raw(raw_alpha, inverse_operator=inv_op,
                                                   lambda2=0.05, method=inv_method, pick_ori='normal')
parcel_series_alpha = extract_parcel_time_series(stc_alpha_raw.data, labels, src, mode='svd', n_select=1)
# beta sources --------
raw_beta = simulated_raw.copy()
raw_beta.load_data()
raw_beta.filter(l_freq=16, h_freq=24, method='iir', iir_params=iir_params)
raw_beta.set_eeg_reference(projection=True)
stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv_op,
                                                  lambda2=0.1, method=inv_method, pick_ori='normal')
parcel_series_beta = extract_parcel_time_series(stc_beta_raw.data, labels, src, mode='svd', n_select=1)
# --------------------------------------------------------------------
# regress out alpha from beta in each ROI
# --------------------------------------------------------------------

t_start = time.time()
parcel_series_beta_corr = regress_out(parcel_series_alpha, parcel_series_beta, int(fs), n=2, mp=True)
t_end = time.time() - t_start
print('*******roi correction took ', t_end / 60, ' min*******')

# --------------------------------------------------------------------
# regress out alpha from beta pair-wise
# --------------------------------------------------------------------
# Compute Connectivity ------------------------
# cross-frequency connectivity ..................
t_start = time.time()
conn_mat_cfc_orig = compute_conn_2D_parallel(parcel_series_alpha, parcel_series_beta, 1, 2, fs, 'abs')
conn_mat_beta_orig = compute_conn_2D_parallel(parcel_series_beta, parcel_series_beta, 1, 1, fs, 'imag')
conn_mat_alpha_orig = compute_conn_2D_parallel(parcel_series_alpha, parcel_series_alpha, 1, 1, fs, 'imag')
t_end = time.time() - t_start
print('*******computing cfc conn matrices took ', t_end / 60, ' min*******')

# within-frequency connectivity ..................
t_start = time.time()
conn_mat_beta_corr = \
    compute_conn_2D_parallel(parcel_series_beta_corr, parcel_series_beta_corr, 1, 1, fs, 'imag')
conn_mat_cfc_corr = compute_conn_2D_parallel(parcel_series_alpha, parcel_series_beta_corr, 1, 2, fs, 'abs')
print('--------beta corr within-freq computed--------')



t_end_iter = time.time() - t_start_iter
print('*********the whole iteration took ', t_end_iter / 60, ' min************')
print('**********************************************************************')


# --------------------------------------------------------------------
# rearrange label
# --------------------------------------------------------------------

beta_orig1 = np.abs(conn_beta_orig1[idx_sorted, :][:, idx_sorted])
alpha_orig1 = np.abs(conn_alpha_orig1[idx_sorted, :][:, idx_sorted])
cfc_orig1 = np.abs(conn_cfc_orig1[idx_sorted, :][:, idx_sorted])

beta_orig = np.abs(conn_mat_beta_orig[idx_sorted, :][:, idx_sorted])
alpha_orig = np.abs(conn_mat_alpha_orig[idx_sorted, :][:, idx_sorted])
cfc_orig = conn_mat_cfc_orig[idx_sorted, :][:, idx_sorted]

cfc_corr = conn_mat_cfc_corr[idx_sorted, :][:, idx_sorted]
beta_corr = np.abs(conn_mat_beta_corr[idx_sorted, :][:, idx_sorted])


# --------------------------------------------------------------------
# plot networks
# --------------------------------------------------------------------
plot_mne_circular_connectivity(alpha_orig1, labels_sorted, perc_conn=0.01, fig=fig, cfc=False, subplot=131,
                               fig_title='alpha', node_name=True, vmax=1, vmin=0, colormap='Blues',
                               facecolor='white', textcolor='black')


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