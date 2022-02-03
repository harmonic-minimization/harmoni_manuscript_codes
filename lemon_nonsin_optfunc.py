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
# from harmoni.harmonitools import harmonic_removal_simple
from tools_harmonic_removal import *
from tools_general import *
# -----------------------------------------
# directory
# -----------------------------------------
"""
NOTE ABOUT DATA
You have to download the data of eyes-closed rsEEG of subject sub-010017 from 
https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/sub-010017/RSEEG/
and put it in the data_dir you specify here.
"""

save_dir = '/data/p_02076/CODES/Codes_CurrentlyWorking/EEG_Networks/build_nets_python36/harmoni/harmoni-supplementary-data/'
data_file = op.join(save_dir, 'lemon_exp')
dict1 = load_pickle(data_file)

src_i = dict1['srcsig']
topo_i = dict1['topo']
src_10 = dict1['src10']
src_20 = dict1['src20']
fs = 250
# apply harmoni to plot the opt surface-----------------
n = 2
c_range = np.arange(-1, 1 + 0.01, 0.001)
phi_range = np.arange(-np.pi/2, np.pi/2, np.pi / 50)
ts1, ts2 = src_10, src_20
ts1_h = hilbert_(ts1)
ts1_ = np.abs(ts1_h) * np.exp(1j * n * np.angle(ts1_h))
ts1_ = ts1_ / np.std(np.real(ts1_))
ts2_ = hilbert_(ts2) / np.std(np.real(ts2))
# ~~~~~~~~~~~~
plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
for n_c, c in enumerate(c_range):
    for n_phi, phi in enumerate(phi_range):
        sig_res = ts2_ - c * np.exp(1j * phi) * ts1_
        plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(ts1_, sig_res, 1, 1, fs, plv_type='abs', coh=True)
print('for finished')
dict_save = {'c_range': c_range, 'phi_range': phi_range, 'plv_sigx_yres_c_phi': plv_sigx_yres_c_phi}
save_pickle(save_dir + 'lemon_exp_optfunc', dict_save)
print('saved')

# --------------------------
# read and plot
# ---------------------------
dict1 = load_pickle('/data/p_02076/CODES/Codes_CurrentlyWorking/EEG_Networks/build_nets_python36/harmoni/harmoni-supplementary-data/lemon_exp_optfunc')
c_range = dict1['c_range']
phi_range = dict1['phi_range']
plv_sigx_yres_c_phi = dict1['plv_sigx_yres_c_phi']
plot_3d(c_range, phi_range, plv_sigx_yres_c_phi)