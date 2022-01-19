
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
from tools_meeg import *
from tools_source_space import *
from tools_connectivity import *
from tools_connectivity_plot import *
from tools_lemon_dataset import *
from tools_signal import *
from tools_peaks import *
from tools_psd_peak import *


# directories and settings -----------------------------------------------------
subject = 'fsaverage'
condition = 'EC'
_oct = '6'
inv_method = 'eLORETA'

subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'
raw_set_dir = '/data/pt_nro109/EEG_LEMON/BIDS_IDS/EEG_Preprocessed_BIDS/'
meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
path_save = '/data/pt_02076/LEMON/Code_Outputs/svd_broad_vs_narrow/'
path_pdf = '/data/pt_02076/LEMON/Code_Outputs/pdfs/peaks/peaks_alpha_2022.pdf'
error_file = '/data/pt_02076/LEMON/log_files/alpha_peak_error_2022_3.txt'
path_peaks = '/data/pt_02076/LEMON/Products/peaks/EC/find_peaks/'
path_save_error_peaks ='/data/pt_02076/LEMON/Code_Outputs/error_peaks_alpha'


# subjects_dir =  '/NOBACKUP/mne_data/'
# raw_set_dir = '/NOBACKUP/Data/lemon/LEMON_prep/'
# meta_file_path = '/NOBACKUP/Data/lemon/behaviour/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
# path_save = '/NOBACKUP/HarmoRemo_Paper/code_outputs/svd-broad-narrow/'

src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')


# pdf = PdfPages(path_pdf)
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
# ids1 = select_subjects('young', 'male', 'right', meta_file_path)
# #IDs = listdir_restricted(raw_set_dir, '-EC-pruned with ICA.set')
# IDs = listdir_restricted(raw_set_dir, '_EC.set')
# # IDs = [id[:-23] for id in IDs]
# IDs = [id[:-7] for id in IDs]
# IDs = np.sort(np.intersect1d(IDs, ids1))
# n_subj = len(IDs)
IDs_check = ['sub-010052', 'sub-010053', 'sub-010224', 'sub-010230', 'sub-010298']
IDs_error = ['sub-010056', 'sub-010070', 'sub-010207', 'sub-010218', 'sub-010238', 'sub-010304', 'sub-010308', 'sub-010314']
IDs = IDs_check + IDs_error
dict_alphapeaks = load_pickle(path_save_error_peaks)

# -----------------------------------------------------
# subjects
# ---------------------------------------------------

for i_subj, subj in enumerate(IDs_error):
    print(subj, ' *******************')
    try:
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
        # title = subj + ' [' + str(np.around(peak_alpha.pass_band[0, 0, 0], 1)) + ', ' + str(
        #     np.around(peak_alpha.pass_band[1, 0, 0], 1)) + ']'
        # fig = plot_peak(f_psd, psd_mean, [peak_alpha], title)
        # pdf.savefig(fig)
        # plt.close(fig)

        width = np.round(np.diff(peak_alpha.pass_band.ravel()).item() / 2, 2)
        peak_beta = peak_alpha.peak.item() * 2
        beta_band = [np.round(peak_beta - width, 2), np.round(peak_beta + width, 2)]
        b10, a10 = signal.butter(N=2, Wn=peak_alpha.pass_band[:, :, 0] / sfreq * 2, btype='bandpass')
        b20, a20 = signal.butter(N=2, Wn=np.asarray(beta_band) / sfreq * 2, btype='bandpass')

        # SVD broad band ---------------------
        raw.set_eeg_reference(projection=True)
        stc_raw = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator=inv_op,
                                                     lambda2=0.05, method=inv_method, pick_ori='normal')
        parcel_series_broad = extract_parcel_time_series(stc_raw.data, labels, src,
                                                         mode='svd', n_select=1, n_jobs=1)

        parcel_series_alpha_b = [filtfilt(b10, a10, sig1) for sig1 in parcel_series_broad]
        parcel_series_beta_b = [filtfilt(b20, a20, sig1) for sig1 in parcel_series_broad]

        # alpha sources --------
        raw_alpha = raw.copy()
        raw_alpha.load_data()
        raw_alpha.filter(l_freq=peak_alpha.pass_band[0].item(), h_freq=peak_alpha.pass_band[1].item(),
                         method='iir', iir_params=iir_params)
        raw_alpha.set_eeg_reference(projection=True)
        stc_alpha_raw = mne.minimum_norm.apply_inverse_raw(raw_alpha, inverse_operator=inv_op,
                                                           lambda2=0.05, method=inv_method, pick_ori='normal')
        parcel_series_alpha = extract_parcel_time_series(stc_alpha_raw.data, labels, src,
                                                         mode='svd', n_select=1, n_jobs=1)
        parcel_alpha = np.squeeze(np.asarray(parcel_series_alpha))
        # beta sources --------
        raw_beta = raw.copy()
        raw_beta.load_data()
        raw_beta.filter(l_freq=beta_band[0], h_freq=beta_band[1],
                        method='iir', iir_params=iir_params)
        raw_beta.set_eeg_reference(projection=True)
        stc_beta_raw = mne.minimum_norm.apply_inverse_raw(raw_beta, inverse_operator=inv_op,
                                                          lambda2=0.1, method=inv_method, pick_ori='normal')
        parcel_series_beta = extract_parcel_time_series(stc_beta_raw.data, labels, src,
                                                        mode='svd', n_select=1, n_jobs=1)
        parcel_beta = np.squeeze(np.asarray(parcel_series_beta))

        coh_alpha = [compute_phase_connectivity(parcel_series_alpha[i], parcel_series_alpha_b[i], 1, 1, type1='abs') for
                     i in range(n_parc)]
        coh_beta = [compute_phase_connectivity(parcel_series_beta[i], parcel_series_beta_b[i], 1, 1, type1='abs') for i
                    in range(n_parc)]
        coh_alpha = np.squeeze(np.array(coh_alpha))
        coh_beta = np.squeeze(np.array(coh_beta))
        dict_save = {'coh_beta': coh_beta, 'coh_alpha': coh_alpha}
        save_file = op.join(path_save, subj + 'coh-svd-broad-narrow')
        save_pickle(save_file, dict_save)
    except Exception as e:
        write_in_txt(error_file, subj + ': ' + str(e))
# pdf.close()
