"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Jaunli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
INSERT THE DOIs
-----------------------------------------------------------------------
script for:
** computing LEMON connectivity **

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
import itertools
from operator import itemgetter
import multiprocessing
from functools import partial
import time
from matplotlib import pyplot as plt


import numpy as np
from numpy import pi

import scipy.signal as signal
import scipy.stats as stats
import mne
import mne.minimum_norm as minnorm
from mne.beamformer import make_lcmv, apply_lcmv_raw

from tools_general import *
from tools_signal import *
from tools_meeg import *
from tools_source_space import *
from tools_connectivity import *
from tools_connectivity_plot import *
from tools_lemon_dataset import *
from tools_harmonic_removal import *

# directories and settings -----------------------------------------------------
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'
# '/NOBACKUP/mne_data/'
subject = 'fsaverage'
_oct = '6'
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
raw_set_dir = '/data/pt_nro109/EEG_LEMON/BIDS_IDS/EEG_Preprocessed_BIDS/'
#'/data/p_02076/Data/LEMON/Juanli-LEMON/EEG_preprocessedData_EC/'
#'/NOBACKUP/Data/lemon/Juanli_data/'
inv_method = 'eLORETA'
save_dir_graphs = '/data/pt_02076/LEMON/lemon_processed_data/networks_bandpass/Beamformer/EC/'
# '/NOBACKUP/Results/lemon_processed_data/graphs/'
meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'


# -----------------------------------------------------
# read the parcellation
# -----------------------------------------------------
# parcellation = dict(name='aparc', abb='DK')  # Desikan-Killiany
parcellation = dict(name='Schaefer2018_100Parcels_7Networks_order', abb='Schaefer100')
labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation['name'])
labels = labels[:-2]
labels_sorted, idx_sorted = rearrange_labels(labels, order='anterior_posterior')  # rearrange labels
n_parc = len(labels)
n_parc_range_prod = list(itertools.product(np.arange(n_parc), np.arange(n_parc)))

# -----------------------------------------------------
# settings
# -----------------------------------------------------
sfreq = 250
iir_params = dict(order=2, ftype='butter')
b10, a10 = signal.butter(N=2, Wn=np.array([8, 12]) / sfreq * 2, btype='bandpass')
b20, a20 = signal.butter(N=2, Wn=np.array([16, 24]) / sfreq * 2, btype='bandpass')

# -----------------------------------------
# the head
# -----------------------------------------
# read forward solution ---------------------------------------------------
fwd = mne.read_forward_solution(fwd_dir)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
src = fwd_fixed['src']

# -----------------------------------------------------
# read raw from set
# ---------------------------------------------------
ids1 = select_subjects('young', 'male', 'right', meta_file_path)
#IDs = listdir_restricted(raw_set_dir, '-EC-pruned with ICA.set')
IDs = listdir_restricted(raw_set_dir, '_EC.set')
# IDs = [id[:-23] for id in IDs]
IDs = [id[:-7] for id in IDs]
IDs = np.sort(np.intersect1d(IDs, ids1))
tstart = time.time()
for i_subj, subj in enumerate(IDs[30:60]):
    print(' ******** subject %d/%d ************' % (i_subj + 1, len(IDs)))
    # raw_name = op.join(raw_set_dir, subj + '-EC-pruned with ICA.set')
    raw_name = op.join(raw_set_dir, subj + '_EC.set')
    raw = read_eeglab_standard_chanloc(raw_name) # , bads=['VEOG']
    assert (sfreq == raw.info['sfreq'])
    raw_data = raw.get_data()
    raw_info = raw.info
    clab = raw_info['ch_names']
    n_chan = len(clab)
    # inv_op = inverse_operator(raw_data.shape, fwd, raw_info)

    data_cov = mne.compute_raw_covariance(raw, tmin=0)
    filters = make_lcmv(raw_info, fwd, data_cov, reg=0.05,
                        noise_cov=None, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)

    # alpha sources --------
    raw_alpha = raw.copy()
    raw_alpha.load_data()
    raw_alpha.filter(l_freq=8, h_freq=12, method='iir', iir_params=iir_params)
    raw_alpha.set_eeg_reference(projection=True)
    # stc_alpha_raw = mne.minimum_norm.apply_inverse_raw(raw_alpha, inverse_operator=inv_op,
    #                                                  lambda2=0.05, method=inv_method, pick_ori='normal')
    stc_alpha_raw = apply_lcmv_raw(raw_alpha, filters)
    parcel_series_alpha = extract_parcel_time_series(stc_alpha_raw.data, labels, src,
                                                     mode='svd', n_select=1, n_jobs=1)
    parcel_alpha = np.squeeze(np.asarray(parcel_series_alpha))
    # beta sources --------
    raw_beta = raw.copy()
    raw_beta.load_data()
    raw_beta.filter(l_freq=16, h_freq=24, method='iir', iir_params=iir_params)
    raw_beta.set_eeg_reference(projection=True)
    stc_beta_raw = apply_lcmv_raw(raw_beta, filters)
    # stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv_op,
    #                                                  lambda2=0.1, method=inv_method, pick_ori='normal')
    # plot_stc_power(stc_beta_raw, subjects_dir, hemi='both', clim='whole')
    parcel_series_beta = extract_parcel_time_series(stc_beta_raw.data, labels, src,
                                                    mode='svd', n_select=1, n_jobs=1)
    parcel_beta = np.squeeze(np.asarray(parcel_series_beta))

    # regress out --------
    parcel_series_beta_corr = regress_out(parcel_series_alpha, parcel_series_beta, int(sfreq), mp=True)
    parcel_beta_corr = np.squeeze(np.asarray(parcel_series_beta_corr))

    # conn matrices --------
    t1 = time.time()
    print('alpha-alpha computation started-----------')
    conn_mat_alpha = compute_synch_permtest_parallel(parcel_alpha[:5], parcel_alpha[:5], 1, 1, sfreq,
                                                     ts1_ts2_eq=True, type1='imag', measure='coh',
                                                     seg_len=None, iter_num=0)
    save_pickle(save_dir_graphs + subj + '-alpha-alpha', conn_mat_alpha)
    t_alpha = time.time() - t1
    print('alpha-alpha computation ended in', t_alpha/60, ' minutes-----------')

    t1 = time.time()
    print('beta-beta computation started-----------')
    conn_mat_beta = compute_synch_permtest_parallel(parcel_beta, parcel_beta, 1, 1, sfreq,
                                                    ts1_ts2_eq=True, type1='imag', measure='coh',
                                                    seg_len=None, iter_num=500)
    save_pickle(save_dir_graphs + subj + '-beta-beta', conn_mat_beta)
    t_beta = time.time() - t1
    print('beta-beta computation ended in', t_beta / 60, ' minutes-----------')


    t1 = time.time()
    print('alpha-beta computation started-----------')
    conn_mat_alpha_beta = compute_synch_permtest_parallel(parcel_alpha, parcel_beta, 1, 2, sfreq,
                                                          ts1_ts2_eq=False, type1='abs', measure='coh',
                                                          seg_len=None, iter_num=500)
    save_pickle(save_dir_graphs + subj + '-alpha-beta', conn_mat_alpha_beta)
    t_alpha_beta = time.time() - t1
    print('alpha-beta computation ended in', t_alpha_beta / 60, ' minutes-----------')

    t1 = time.time()
    print('beta-beta-corr computation started-----------')
    conn_mat_beta_corr = compute_synch_permtest_parallel(parcel_beta_corr, parcel_beta_corr, 1, 1, sfreq,
                                                         ts1_ts2_eq=True, type1='imag', measure='coh',
                                                         seg_len=None, iter_num=500)
    save_pickle(save_dir_graphs + subj + '-beta-beta-corr', conn_mat_beta_corr)
    t_beta = time.time() - t1
    print('beta-beta-corr computation ended in', t_beta / 60, ' minutes-----------')

    t1 = time.time()
    print('alpha-beta-corr computation started-----------')
    conn_mat_alpha_beta_corr = compute_synch_permtest_parallel(parcel_alpha, parcel_beta_corr, 1, 2, sfreq,
                                                               ts1_ts2_eq=False, type1='abs', measure='coh',
                                                               seg_len=None, iter_num=500)
    save_pickle(save_dir_graphs + subj + '-alpha-beta-corr', conn_mat_alpha_beta_corr)
    t_alpha_beta = time.time() - t1
    print('alpha-beta-corr computation ended in', t_alpha_beta / 60, ' minutes-----------')

t_stop = time.time() - tstart


