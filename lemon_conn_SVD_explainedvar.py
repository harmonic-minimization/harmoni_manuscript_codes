
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
from sklearn.decomposition import TruncatedSVD

from tools_general import *
from tools_signal import *
from tools_meeg import *
from tools_source_space import *
from tools_connectivity import *
from tools_multivariate import *
from tools_connectivity_plot import *
from tools_lemon_dataset import *
from tools_harmonic_removal import *
from tools_psd_peak import *


# directories and settings -----------------------------------------------------
subject = 'fsaverage'
condition = 'EC'
_oct = '6'
inv_method = 'eLORETA'

subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'
raw_set_dir = '/data/pt_nro109/EEG_LEMON/BIDS_IDS/EEG_Preprocessed_BIDS/'
meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
error_file = '/data/pt_02076/LEMON/log_files/connectivities_2022_3.txt'
path_peaks = '/data/pt_02076/LEMON/Products/peaks/EC/find_peaks/'
save_dir_graphs = '/data/pt_02076/LEMON/lemon_processed_data/networks_coh_peak_detection_no_perm/'
path_save_error_peaks ='/data/pt_02076/LEMON/Code_Outputs/error_peaks_alpha'
path_save_explained_var = '/data/pt_02076/LEMON/Code_Outputs/explained_var/'

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
IDs = listdir_restricted(raw_set_dir, '_EC.set')
IDs = [id[:-7] for id in IDs]
IDs = np.sort(np.intersect1d(IDs, ids1))
# IDs_error = ['sub-010056', 'sub-010070', 'sub-010207', 'sub-010218', 'sub-010238', 'sub-010304', 'sub-010308', 'sub-010314', 'sub-010241']
dict_alphapeaks = load_pickle(path_save_error_peaks)
IDs_error = list(dict_alphapeaks.keys())

tstart = time.time()
for i_subj, subj in enumerate(IDs):
    print(' ******** subject %d/%d ************' % (i_subj + 1, len(IDs)))
    # raw_name = op.join(raw_set_dir, subj + '-EC-pruned with ICA.set')
    raw_name = op.join(raw_set_dir, subj + '_EC.set')
    raw = read_eeglab_standard_chanloc(raw_name)  # , bads=['VEOG']
    assert (sfreq == raw.info['sfreq'])
    raw_data = raw.get_data()
    raw_info = raw.info
    clab = raw_info['ch_names']
    n_chan = len(clab)
    inv_op = inverse_operator(raw_data.shape, fwd, raw_info)

    if subj in IDs_error:
        peak_alpha = dict_alphapeaks[subj]
    else:
        peaks_file = op.join(path_peaks, subj + '-peaks.npz')
        peaks = np.load(peaks_file)
        f_psd, psd_data = psd(raw_data, sfreq, 45, plot=False)
        psd_mean = np.mean(psd_data, axis=0)
        peak_alpha = find_narrowband_peaks(peaks['peaks'], peaks['peaks_ind'], peaks['pass_freq'],
                                           np.array([8, 12]), 6, f_psd, psd_mean)

    width = np.round(np.diff(peak_alpha.pass_band.ravel()).item() / 2, 2)
    peak_beta = peak_alpha.peak.item() * 2
    beta_band = [np.round(peak_beta - width, 2), np.round(peak_beta + width, 2)]
    b10, a10 = signal.butter(N=2, Wn=peak_alpha.pass_band[:, :, 0] / sfreq * 2, btype='bandpass')
    b20, a20 = signal.butter(N=2, Wn=np.asarray(beta_band) / sfreq * 2, btype='bandpass')

    # SVD broad band ---------------------
    raw.set_eeg_reference(projection=True)
    stc_raw = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator=inv_op,
                                                 lambda2=0.05, method=inv_method, pick_ori='normal')
    explained_variance = np.zeros((n_parc, 5))
    for i_lbl, label in enumerate(labels):
        lbl_idx, _ = label_idx_whole_brain(src, label)
        data_lbl = stc_raw.data[lbl_idx, :]
        svd = TruncatedSVD(n_components=5)
        svd.fit(data_lbl)
        explained_variance[i_lbl, :] = svd.singular_values_**2 / np.sum(svd.singular_values_**2)
    file_name = op.join(path_save_explained_var, subj + '_svd_explainedvar')
    save_pickle(file_name, explained_variance)

t_stop = time.time() - tstart


# -----------------------------------------------------
# read explained vars
# ---------------------------------------------------
all_vars = np.zeros((len(IDs), n_parc))
for i_subj, subj in enumerate(IDs):
    file_name = op.join(path_save_explained_var, subj + '_svd_explainedvar')
    explained_variance = load_pickle(file_name)
    all_vars[i_subj, :] = explained_variance[:, 0]