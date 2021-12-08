"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
script for:
**
-----------------------------------------------------------------------

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20211004 by \Mina

"""

import os.path as op
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import mne
from tools_meeg import read_eeglab_standard_chanloc, plot_topomap_
from tools_signal import psd, hilbert_
from tools_connectivity import compute_phase_connectivity
from tools_general import circular_hist
from harmoni.harmonitools import harmonic_removal_simple
# -----------------------------------------
# directory
# -----------------------------------------
"""
NOTE ABOUT DATA
You have to download the data of eyes-closed rsEEG of subject sub-010017 from 
https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/sub-010017/RSEEG/
and put it in the data_dir you specify here.
"""

condition = 'EC'
data_dir = '/data/pt_nro109/Share/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/'

# subjects_dir = '/NOBACKUP/mne_data/'
subjects_dir = '/data/pt_02076/mne_data/MNE-fsaverage-data/'

subject = 'fsaverage'
_oct = '6'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')


# -----------------------------------------
# read data
# -----------------------------------------
subj = 'sub-010017'
set_name = data_dir + subj + '_' + condition + '.set'
raw = read_eeglab_standard_chanloc(set_name)
sfreq = raw.info['sfreq']
rawdata = raw.get_data()

# -----------------------------------------
# filters settings
# -----------------------------------------
iir_params = dict(order=2, ftype='butter')
b10, a10 = butter(N=2, Wn=np.array([9, 11]) / sfreq * 2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([18, 22]) / sfreq * 2, btype='bandpass')
b30, a30 = butter(N=2, Wn=np.array([27, 33]) / sfreq * 2, btype='bandpass')

# -----------------------------------------
# ICA
# -----------------------------------------
np.random.seed(3100819795)  # for reproducibility

n_components = 32
method = 'fastica'
ica = ICA(n_components=n_components, method=method)

picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
ica.fit(raw, picks=picks, reject_by_annotation=True)
src = ica.get_sources(raw)
src = src.get_data()
mixing_mat = ica.get_components()

i_cmp = 11  # this component is selected previously manualy
src_i = src[i_cmp, :]
topo_i = mixing_mat[:, i_cmp]

# filter in the harmonic frequency ranges
src_10 = filtfilt(b10, a10, src_i)
src_20 = filtfilt(b20, a20, src_i)
src_30 = filtfilt(b30, a30, src_i)

src_10 = src_10 / np.std(src_10)
src_20 = src_20 / np.std(src_20)
src_20_res, c_opt, phi_opt = harmonic_removal_simple(src_10, src_20, sfreq, return_all=True)

scr10_h = hilbert_(src_10).ravel()
scr20_h = hilbert_(src_20).ravel()


phase_diff_all = np.mod(np.angle(scr20_h) - 2 * np.angle(scr10_h), 2*np.pi)
fig = plt.figure()
circular_hist(phase_diff_all, alpha=1, bins=100, plot_mean=True)
plt.title('histogram of the phase difference of the whole signals')

print('mean angle al==', stats.circmean(phase_diff_all))

win_len = int(30 * sfreq)
overlap_perc = 0.8
start_step = int((1 - overlap_perc) * win_len)
n_win = 10
cc = []
rr = []
win_no = -1
phase_mean = []
power_ratio = []
pow_alpha = []
pow_beta = []
while 1:
    # win_no = 3
    win_no += 1
    win_start = win_no * start_step
    win_end = win_start + win_len
    if win_end > len(scr10_h):
        break
    s1 = scr10_h[win_start:win_end]
    s2 = scr20_h[win_start:win_end]
    phase_diff = np.mod(np.angle(s2) - 2 * np.angle(s1), 2*np.pi)
    # cc1, rr1 = np.histogram(phase_diff)
    # cc.append(cc1)
    # rr.append(rr1)
    phase_mean.append(stats.circmean(phase_diff))
    # circular_hist(phase_diff, alpha=0.2, bins=100)

    # pow_ratio = np.mean(np.real(s1)**2) / np.mean(np.real(s2)**2)
    # pow_ratio = np.mean(np.abs(s2)) / np.mean(np.abs(s1))
    # plt.plot(np.abs(s1) / np.abs(s2))
    pow_ratio = np.mean(np.abs(s2) / np.abs(s1))
    power_ratio.append(pow_ratio)
    pow_alpha.append(np.mean(np.abs(s1)))
    pow_beta.append(np.mean(np.abs(s2)))
    # circular_hist(phase_diff, alpha=0.2)


fig = plt.figure()
circular_hist(phase_mean, alpha=1, bins=100, plot_mean=True)
plt.title('histogram of the mean phase difference of the sliding windows')
print('mean angle al==', stats.circmean(phase_mean))




