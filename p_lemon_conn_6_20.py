# compute 1:4 6:6 Hz fundamental


import os.path as op
import itertools
from operator import itemgetter
import multiprocessing
from functools import partial
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy import pi

import scipy.signal as signal
import scipy.stats as stats
from scipy.signal import butter, filtfilt
import mne
import mne.minimum_norm as minnorm

from tools_general import *
from tools_signal import *
from tools_meeg import *
from tools_source_space import *
from tools_connectivity import *
from tools_connectivity_plot import *
from tools_lemon_dataset import *
from tools_signal import *


# directories and settings -----------------------------------------------------
subject = 'fsaverage'
condition = 'EC'
_oct = '6'
inv_method = 'eLORETA'

subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'
raw_set_dir = '/data/pt_nro109/EEG_LEMON/BIDS_IDS/EEG_Preprocessed_BIDS/'
meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
path_save = '/data/pt_02076/LEMON/Code_Outputs/coh_6_20/'


# subjects_dir =  '/NOBACKUP/mne_data/'
# raw_set_dir = '/NOBACKUP/Data/lemon/LEMON_prep/'
# meta_file_path = '/NOBACKUP/Data/lemon/behaviour/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
# path_save = '/NOBACKUP/HarmoRemo_Paper/code_outputs/svd-broad-narrow/'

src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
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
b6, a6 = signal.butter(N=2, Wn=np.array([5.6, 7.6]) / sfreq * 2, btype='bandpass')
b20, a20 = signal.butter(N=2, Wn=np.array([19, 21]) / sfreq * 2, btype='bandpass')
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
n_subj = len(IDs)

# -----------------------------------------------------
# subjects
# ---------------------------------------------------

for i_subj, subj in enumerate(IDs):
    print(subj, '-subj' + str(i_subj) + ' of ' + str(n_subj) + ' *******************')
    print('***************************************')
    raw_name = op.join(raw_set_dir, subj + '_EC.set')
    raw = read_eeglab_standard_chanloc(raw_name)  # , bads=['VEOG']
    assert (sfreq == raw.info['sfreq'])
    raw_data = raw.get_data()
    raw_info = raw.info
    clab = raw_info['ch_names']
    n_chan = len(clab)
    inv_op = inverse_operator(raw_data.shape, fwd, raw_info)

    # SVD broad band ---------------------
    raw.set_eeg_reference(projection=True)
    stc_raw = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator=inv_op,
                                                 lambda2=0.05, method=inv_method, pick_ori='normal')
    parcel_series_broad = extract_parcel_time_series(stc_raw.data, labels, src,
                                                     mode='svd', n_select=1, n_jobs=1)

    parcel_series_6_b = [filtfilt(b6, a6, sig1) for sig1 in parcel_series_broad]
    parcel_series_20_b = [filtfilt(b20, a20, sig1) for sig1 in parcel_series_broad]

    # SVD narrow band ----------------
    # 6Hz sources --------
    # raw_6Hz = raw.copy()
    # raw_6Hz.load_data()
    # raw_6Hz.filter(l_freq=5.6, h_freq=7.6, method='iir', iir_params=iir_params)
    # raw_6Hz.set_eeg_reference(projection=True)
    # stc_6Hz_raw = mne.minimum_norm.apply_inverse_raw(raw_6Hz, inverse_operator=inv_op,
    #                                                    lambda2=0.05, method=inv_method, pick_ori='normal')
    # parcel_series_6 = extract_parcel_time_series(stc_6Hz_raw.data, labels, src,
    #                                                  mode='svd', n_select=1, n_jobs=1)

    # beta sources --------
    # raw_beta = raw.copy()
    # raw_beta.load_data()
    # raw_beta.filter(l_freq=19, h_freq=21, method='iir', iir_params=iir_params)
    # raw_beta.set_eeg_reference(projection=True)
    # stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv_op,
    #                                                   lambda2=0.1, method=inv_method, pick_ori='normal')
    # # plot_stc_power(stc_beta_raw, subjects_dir, hemi='both', clim='whole')
    # parcel_series_20 = extract_parcel_time_series(stc_beta_raw.data, labels, src,
    #                                                 mode='svd', n_select=1, n_jobs=1)
    #
    coh_6_20 = [compute_phase_connectivity(parcel_series_6_b[i], parcel_series_20_b[i], 1, 3, type1='abs') for i in
                range(n_parc)]
    save_file = op.join(path_save, subj + '-coh-6-20')
    save_pickle(save_file, coh_6_20)