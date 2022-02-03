"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
script for:
** non-sin source signal from rsEEG of LEMON data **

here what we do is to use ICA to extract a source with non-sinusoidal shape.
the component is selected manualy from the ICA components.
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
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import mne
from tools_meeg import read_eeglab_standard_chanloc, plot_topomap_
from tools_signal import psd
from tools_connectivity import compute_phase_connectivity
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
save_dir = '/data/p_02076/CODES/Codes_CurrentlyWorking/EEG_Networks/build_nets_python36/harmoni/harmoni-supplementary-data/'
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

i_cmp = 8  # this component is selected previously manualy
src_i = src[i_cmp, :]
topo_i = mixing_mat[:, i_cmp]

# filter in the harmonic frequency ranges
src_10 = filtfilt(b10, a10, src_i)
src_20 = filtfilt(b20, a20, src_i)
src_30 = filtfilt(b30, a30, src_i)
dict_save = {'srcsig': src_i, 'topo': topo_i, 'src10': src_10, 'src20': src_20}
save_pickle(save_dir + 'lemon_exp', dict_save)
# plot the source topography
plot_topomap_(topo_i, raw.info)

# plot the psd --------------------
psd(src_i, sfreq, 45)

# plot the source signal
times = np.arange(0, len(src_i)) / sfreq
ind1 = np.argmin(np.abs(times - 197.389))
ind2 = np.argmin(np.abs(times - 201.029))

# plot the signals- ---------------------------
plt.figure()
plt.plot(times[ind1:ind2], src_i[ind1:ind2], label='orig signal')
plt.plot(times[ind1:ind2], src_10[ind1:ind2], label='fundamental')
plt.plot(times[ind1:ind2], src_20[ind1:ind2], label='2st harmonic')
plt.plot(times[ind1:ind2], src_30[ind1:ind2], label='3rd harmonic')
plt.legend()

# coherence of the components -----------------
plv12 = compute_phase_connectivity(src_10, src_20, 1, 2, type1='abs')[0]
plv13 = compute_phase_connectivity(src_10, src_30, 1, 3, type1='abs')[0]
plv23 = compute_phase_connectivity(src_20, src_30, 2, 3, type1='abs')[0]

print('plv12=', str(plv12), ', plv13=', str(plv13), ', plv23=', str(plv23))

# apply harmoni to plot the opt surface-----------------
n = 2
c_range = np.arange(-1, 1 + 0.01, 0.001)
phi_range = np.arange(-np.pi/2, np.pi/2, np.pi / 50)
ts1, ts2 = src_10, src_20
ts1_h = hilbert_(ts1)
ts1_ = np.abs(ts1_h) * np.exp(1j * n * np.angle(ts1_h))
ts1_ = ts1_ / np.std(np.real(ts1_))
ts2_ = hilbert_(ts2) / np.std(np.real(ts2))

ts20_corr1, c_opt1, phi_opt1, plv_sigx_yres_c_phi1 = harmonic_removal_simple(src_10, src_20, sfreq, n=2, return_all=True)

c, phi = optimize_gradient(ts2_, ts1_, alpha=0.1, error_thresh=[10e-5, pi/1000], dphi=1e-5, dc=1e-5)

x, y = ts1_, ts2_
alpha = 0.01
dphi = 1e-5
dc = 1e-5

error_thresh = [10e-10, 10e-10]
synch_func1 = lambda z1, z2: compute_coherency_1(z1, z2, 1, 1)
c1 = np.random.random(1)[0] * 2 - 1
phi1 = np.random.random(1)[0] * pi - pi / 2
print(c1, phi1)
y_res = y - c1 * np.exp(1j * phi1) * x
diff_error_c, diff_error_phi = 1, 1
# while diff_error_c > error_thresh[0] or diff_error_phi > error_thresh[1]:
value = [synch_func1(x,  y - c1 * np.exp(1j * phi1) * x)]
for iter_n in range(5000):
    factor = value[-1] / 0.1
    alpha = 0.01 * np.max([factor, 1])
    grad_c, grad_phi = _compute_grad_2(x, y, c1, phi1, dc, dphi, synch_func1)
    c1_old, phi1_old = c1, phi1
    # if np.abs(grad_c) > 0.02:
    c1 -= alpha * grad_c
    # if np.abs(grad_phi) > 0.02:
    phi1 -= alpha * grad_phi

    # diff_error_c, diff_error_phi = np.abs( alpha * grad_c), np.abs(alpha * grad_phi)
    value.append(synch_func1(x,  y - c1 * np.exp(1j * phi1) * x))
    print(iter_n, '......', grad_c, grad_phi, '.....', c1, phi1, '.....', value[-1])
    if value[-1] - value[-2] > 0:
        break
    # if diff_error_c <= error_thresh[0] and diff_error_phi <= error_thresh[1]:
    #     break
c_opt, phi_opt = c1_old, phi1_old

print(c1, phi1)
print(c_opt, phi_opt)
