"""
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, ..., Vadim V. Nikulin
(c) insert the future preprint and ms link here


** Code to generate Figure  2: non-sin source signal from rsEEG of LEMON data **

-----------------------------------------------------------------------

(c) Mina jamshidi Idaji @ Neurolgy Dept, MPI CBS
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------

last modified: 20210615 by \Mina

"""
import os.path as op
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import mne
from harmoni_tools_meeg import read_eeglab_standard_chanloc, plot_topomap_
from harmoni_tools_signal import psd
from harmoni_tools_connectivity import compute_phase_connectivity
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
data_dir = '/NOBACKUP/Data/lemon/LEMON_prep/'

subjects_dir = '/NOBACKUP/mne_data/'
subject = 'fsaverage'
_oct = '6'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
# -----------------------------------------
# settings
# -----------------------------------------
iir_params = dict(order=2, ftype='butter')
b10, a10 = butter(N=2, Wn=np.array([9, 11]) / sfreq * 2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([18, 22]) / sfreq * 2, btype='bandpass')
b30, a30 = butter(N=2, Wn=np.array([27, 33]) / sfreq * 2, btype='bandpass')

# -----------------------------------------
# read data
# -----------------------------------------
subj = 'sub-010017'
set_name = data_dir + subj + '_' + condition + '.set'
raw = read_eeglab_standard_chanloc(set_name)
sfreq = raw.info['sfreq']
rawdata = raw.get_data()

np.random.seed(3100819795)  # for reproducibility

n_components = 32
method = 'fastica'
ica = ICA(n_components=n_components, method=method)

picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
ica.fit(raw, picks=picks, reject_by_annotation=True)
src = ica.get_sources(raw)
src = src.get_data()
mixing_mat = ica.get_components()

i_cmp = 8
src_i = src[i_cmp, :]
topo_i = mixing_mat[:, i_cmp]

# filter in the harmonic frequency ranges
src_10 = filtfilt(b10, a10, src_i)
src_20 = filtfilt(b20, a20, src_i)
src_30 = filtfilt(b30, a30, src_i)

# plot the source topography
plot_topomap_(topo_i, raw.info)

# plot the psd
psd(src_i, sfreq, 45)

# plot the source signal
times = np.arange(0, len(src_i)) / sfreq
ind1 = np.argmin(np.abs(times - 197.389))
ind2 = np.argmin(np.abs(times - 201.029))

# plot the signals
plt.figure()
plt.plot(times[ind1:ind2], src_i[ind1:ind2], label='orig signal')
plt.plot(times[ind1:ind2], src_10[ind1:ind2], label='fundamental')
plt.plot(times[ind1:ind2], src_20[ind1:ind2], label='2st harmonic')
plt.plot(times[ind1:ind2], src_30[ind1:ind2], label='3rd harmonic')
plt.legend()

# coherence of the components
plv12 = compute_phase_connectivity(src_10, src_20, 1, 2, type1='abs')[0]
plv13 = compute_phase_connectivity(src_10, src_30, 1, 3, type1='abs')[0]
plv23 = compute_phase_connectivity(src_20, src_30, 2, 3, type1='abs')[0]

print('plv12=', str(plv12), ', plv13=', str(plv13), ', plv23=', str(plv23))

