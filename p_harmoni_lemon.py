

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

# subjects_dir = '/NOBACKUP/mne_data/'
# path_outs = '/NOBACKUP/HarmoRemo_Paper/code_outputs/'


path_coh_6 = op.join(path_outs, 'coh_6_20')
path_svd = op.join(path_outs, 'svd_broad_vs_narrow')
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
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

plt.figure()
sns.histplot(data=coh_mat.ravel())
plt.xlabel('1:3 ROI coherence between 6.6Hz and 20 Hz')

# -----------------------------------------------------
# SVD 30hz
# ---------------------------------------------------
IDs = listdir_restricted(path_svd, 'coh-svd-broad-narrow-30Hz')
IDs = [id[:-25] for id in IDs]
n_subj = len(IDs)

perc_range = np.arange(5, 95, 5)
perc_30Hz = np.zeros((n_subj, len(perc_range)))
coh_30Hz = np.zeros((n_subj, 100))


for i_subj, subj in enumerate(IDs):
    print(i_subj)
    file_name = op.join(path_svd, subj + 'coh-svd-broad-narrow-30Hz')
    coh_30 = load_pickle(file_name)
    coh_30 = np.squeeze(np.asarray(coh_30))
    for i_perc, perc in enumerate(perc_range):
        perc_30Hz[i_subj, i_perc] = np.percentile(coh_30, perc)

plt.figure()
ax = plt.subplot(111)
ax.errorbar(perc_range, np.mean(perc_30Hz, axis=0), yerr=np.std(perc_30Hz, axis=0), fmt='-o')
plt.ylabel('percentile')
plt.xlabel('percentage')
plt.xticks(perc_range)

# -----------------------------------------------------
# SVD
# ---------------------------------------------------
IDs = listdir_restricted(path_svd, 'coh-svd-broad-narrow')
IDs = [id[:-20] for id in IDs]
n_subj = len(IDs)

perc_range = np.arange(5, 95, 5)
coh_alpha = np.zeros((n_subj, 100))
coh_beta = np.zeros((n_subj, 100))
perc_beta = np.zeros((n_subj, len(perc_range)))
perc_alpha = np.zeros((n_subj, len(perc_range)))

for i_subj, subj in enumerate(IDs):
    print(i_subj)
    file_name = op.join(path_svd, subj + 'coh-svd-broad-narrow')
    dict_subj = load_pickle(file_name)
    coh_alpha[i_subj, :] = dict_subj['coh_alpha']
    coh_beta[i_subj, :] = dict_subj['coh_beta']
    for i_perc, perc in enumerate(perc_range):
        perc_alpha[i_subj, i_perc] = np.percentile(coh_alpha[i_subj, :], perc)
        perc_beta[i_subj, i_perc] = np.percentile(coh_beta[i_subj, :], perc)

plt.figure()
plt.subplot(121)
sns.histplot(data=coh_alpha.ravel(), bins=20)
plt.title('alpha band')
plt.xlabel('absolute coherence ')
plt.subplot(122)
sns.histplot(data=coh_beta.ravel(), bins=20)
plt.title('beta band')
plt.xlabel('absolute coherence ')


plt.figure()
ax = plt.subplot(121)
ax.errorbar(perc_range, np.mean(perc_alpha, axis=0), yerr=np.std(perc_alpha, axis=0), fmt='-o')
plt.ylabel('percentile')
plt.xlabel('percentage')
plt.xticks(perc_range)
plt.title('alpha band')

ax = plt.subplot(122)
ax.errorbar(perc_range, np.mean(perc_beta, axis=0), yerr=np.std(perc_beta, axis=0), fmt='-o')
plt.ylabel('percentile')
plt.xlabel('percentage')
plt.xticks(perc_range)
plt.title('beta band')
plt.show()