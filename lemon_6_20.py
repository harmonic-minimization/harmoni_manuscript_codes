

import os.path as op
import itertools
from operator import itemgetter
import multiprocessing
from functools import partial
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
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
path_outs = '/data/pt_02076/LEMON/Code_Outputs/'
raw_set_dir = '/data/pt_nro109/EEG_LEMON/BIDS_IDS/EEG_Preprocessed_BIDS/'

# subjects_dir = '/NOBACKUP/mne_data/'
# path_outs = '/NOBACKUP/HarmoRemo_Paper/code_outputs/'


path_coh_6 = op.join(path_outs, 'coh_6_20')
path_svd = op.join(path_outs, 'svd_broad_vs_narrow')
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')

# -----------------------------------------
# the head
# -----------------------------------------
# read forward solution ---------------------------------------------------
fwd = mne.read_forward_solution(fwd_dir)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
src = fwd_fixed['src']

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
b10, a10 = signal.butter(N=2, Wn=np.array([9, 11]) / sfreq * 2, btype='bandpass')
b13, a13 = signal.butter(N=2, Wn=np.array([12.2, 14.2]) / sfreq * 2, btype='bandpass')
b3, a3 = signal.butter(N=2, Wn=np.array([2.3, 4.3]) / sfreq * 2, btype='bandpass')
b5, a5 = signal.butter(N=2, Wn=np.array([4, 6]) / sfreq * 2, btype='bandpass')
# -----------------------------------------------------
# coherence
# ---------------------------------------------------
IDs = listdir_restricted(path_coh_6, '-coh-6-20')
IDs = [id[:-9] for id in IDs]
n_subj = len(IDs)

coh_mat = np.zeros((n_subj, 100))
for i_subj, subj in enumerate(IDs):
    print(i_subj)
    file_name = op.join(path_coh_6, subj + '-coh-6-20')
    coh_mat_subj = load_pickle(file_name)
    coh_mat[i_subj, :] = np.squeeze(np.asanyarray(coh_mat_subj))

# plt.figure()
# sns.histplot(data=coh_mat.ravel())
# plt.xlabel('1:3 ROI coherence between 6.6Hz and 20 Hz')


subj_inds, rois = np.where(coh_mat > 0.3)

# -----------------------------------------------------
# check subject 17, ROIs 6 and 7
# ---------------------------------------------------
subj = IDs[17]

raw_name = op.join(raw_set_dir, subj + '_EC.set')
raw = read_eeglab_standard_chanloc(raw_name)  # , bads=['VEOG']
raw_data = raw.get_data()
raw_info = raw.info
psd(raw_data, sfreq, 45)
inv_op = inverse_operator(raw_data.shape, fwd, raw_info)

# SVD broad band ---------------------
raw.set_eeg_reference(projection=True)
stc_raw = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator=inv_op,
                                             lambda2=0.05, method=inv_method, pick_ori='normal')
parcel_series_broad = extract_parcel_time_series(stc_raw.data, labels, src,
                                                 mode='svd', n_select=1, n_jobs=1)

parcel_series_6 = [filtfilt(b6, a6, sig1) for sig1 in parcel_series_broad]
parcel_series_13 = [filtfilt(b13, a13, sig1) for sig1 in parcel_series_broad]
parcel_series_20 = [filtfilt(b20, a20, sig1) for sig1 in parcel_series_broad]
parcel_series_10 = [filtfilt(b10, a10, sig1) for sig1 in parcel_series_broad]
parcel_series_3 = [filtfilt(b3, a3, sig1) for sig1 in parcel_series_broad]
parcel_series_5 = [filtfilt(b5, a5, sig1) for sig1 in parcel_series_broad]

compute_phase_connectivity(parcel_series_6[6], parcel_series_20[6], 1, 3, type1='abs')
compute_phase_connectivity(parcel_series_6[7], parcel_series_20[7], 1, 3, type1='abs')

coh_6_10 = [compute_phase_connectivity(parcel_series_6[6], parcel_series_10[i], 2, 3, type1='abs') for i in
                range(n_parc)]
parc_ind = np.where(np.asarray(coh_6_10) > 0.35)

coh_20_10 = [compute_phase_connectivity(parcel_series_20[6], parcel_series_10[i], 2, 1, type1='abs') for i in
                range(n_parc)]


save_parcels = [parcel_series_broad[i] for i in [6, 7, 70]]
save_pickle('/data/p_02076/CODES/Codes_CurrentlyWorking/EEG_Networks/build_nets_python36/harmoni/harmoni-supplementary-data/parcels_R2C2', save_parcels)
# -----------------------------------------------------
# ROIs 6 vs 7 and 50
# ---------------------------------------------------
i_parc1 = 6
i_parc2 = 70

parc1_coh_3_6 = compute_phase_connectivity(parcel_series_3[i_parc1], parcel_series_6[i_parc1], 1, 2, type1='abs')
parc1_coh_5_20 = compute_phase_connectivity(parcel_series_5[i_parc1], parcel_series_20[i_parc1], 1, 4, type1='abs')
parc1_coh_6_10 = compute_phase_connectivity(parcel_series_6[i_parc1], parcel_series_10[i_parc1], 2, 3, type1='abs')
parc1_coh_6_20 = compute_phase_connectivity(parcel_series_6[i_parc1], parcel_series_20[i_parc1], 1, 3, type1='abs')
parc1_coh_10_20 = compute_phase_connectivity(parcel_series_10[i_parc1], parcel_series_20[i_parc1], 1, 2, type1='abs')


coh_parc1_6_parc2_10 = compute_phase_connectivity(parcel_series_6[i_parc1], parcel_series_10[i_parc2], 2, 3, type1='abs')
coh_parc1_20_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc2], parcel_series_20[i_parc1], 1, 2, type1='abs')
coh_parc1_3_parc2_3 = compute_phase_connectivity(parcel_series_3[i_parc1], parcel_series_3[i_parc2], 1, 1, type1='complex')
coh_parc1_6_parc2_6 = compute_phase_connectivity(parcel_series_6[i_parc1], parcel_series_6[i_parc2], 1, 1, type1='complex')
coh_parc1_10_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc1], parcel_series_10[i_parc2], 1, 1, type1='complex')
coh_parc1_20_parc2_20 = compute_phase_connectivity(parcel_series_20[i_parc1], parcel_series_20[i_parc2], 1, 1, type1='complex')

# correct parc1_6.6Hz and parc2_10Hz by 3.3hz each ---------------------------------
parc1_6_corr_3 = harmonic_removal_simple(parcel_series_3[i_parc1], parcel_series_6[i_parc1], sfreq, n=2)

parc2_10hz_corr_5 = harmonic_removal_simple(parcel_series_5[i_parc2], parcel_series_10[i_parc2], sfreq, n=2)
parc2_10_corr_5_3 = harmonic_removal_simple(parcel_series_3[i_parc2], parc2_10hz_corr_5, sfreq, n=3)

coh_parc1_6corr3_parc2_10corr5 = compute_phase_connectivity(parc1_6_corr_3, parc2_10hz_corr_5, 2, 3, type1='abs')
coh_parc1_6corr3_parc2_10corr5corr3 = compute_phase_connectivity(parc1_6_corr_3, parc2_10_corr_5_3, 2, 3, type1='abs')

# correct parc1_20hz by 6.6hz and 10Hz  ---------------------------------
parc1_20hz_corr_6 = harmonic_removal_simple(parcel_series_6[i_parc1], parcel_series_20[i_parc1], sfreq, n=3)
parc1_20hz_corr_6_10 = harmonic_removal_simple(parcel_series_10[i_parc1], parc1_20hz_corr_6, sfreq, n=2)

parc1_20hz_corr_10 = harmonic_removal_simple(parcel_series_10[i_parc1], parcel_series_20[i_parc1], sfreq, n=2)
parc1_20hz_corr_10_6 = harmonic_removal_simple(parcel_series_6[i_parc1], parc1_20hz_corr_10, sfreq, n=3)
parc1_20hz_corr_10_6_5 = harmonic_removal_simple(parcel_series_5[i_parc1], parc1_20hz_corr_10_6, sfreq, n=4)

coh_parc1_20_corr6_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc2], parc1_20hz_corr_6, 1, 2, type1='abs')
coh_parc1_20_corr6corr10_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc2], parc1_20hz_corr_6_10, 1, 2, type1='abs')

coh_parc1_20_corr10_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc2], parc1_20hz_corr_10, 1, 2, type1='abs')
coh_parc1_20_corr10corr6_parc2_10 = compute_phase_connectivity(parcel_series_10[i_parc2], parc1_20hz_corr_10_6, 1, 2, type1='abs')


parc2_10hz_corr_5 = harmonic_removal_simple(parcel_series_5[i_parc2], parcel_series_10[i_parc2], sfreq, n=2)
coh_parc1_20_corr10corr6corr5_parc2_10 = compute_phase_connectivity(parc2_10hz_corr_5, parc1_20hz_corr_10_6_5, 1, 2, type1='abs')







