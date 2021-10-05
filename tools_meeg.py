import matplotlib.pyplot as plt
import numpy as np


#  --------------------------------  --------------------------------  --------------------------------
# read, save
#  --------------------------------  --------------------------------  --------------------------------

def read_eeglab_standard_chanloc(raw_name, bads=None, monatage_name='standard_1005'):
    import mne
    raw = mne.io.read_raw_eeglab(raw_name)
    new_names = dict(
        (ch_name,
         ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
        for ch_name in raw.ch_names)
    raw.rename_channels(new_names)
    raw.info['bads'] = bads if bads is not None else []

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
    raw = raw.pick(picks)
    montage = mne.channels.make_standard_montage(monatage_name)
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling
    return raw

#  --------------------------------  --------------------------------  --------------------------------
# plot on topomap
#  --------------------------------  --------------------------------  --------------------------------


def plot_topomap_(map, info, title='', vmax=None, vmin=None, ax=None, cmap=None, mask=None):
    from mne.viz import plot_topomap
    map = map.ravel()
    if ax is None:
        fig = plt.figure()
    im, _ = plot_topomap(map, info, vmax=vmax, vmin=vmin, cmap=cmap, mask=mask)
    plt.colorbar(im)
    plt.title(title)
    if ax is None:
        return fig

#  --------------------------------  --------------------------------  --------------------------------
# forward and inverse
#  --------------------------------  --------------------------------  --------------------------------

def inverse_operator(size1, fwd, raw_info):
    import mne
    from mne.minimum_norm import make_inverse_operator
    data = np.random.normal(loc=0.0, scale=1.0, size=size1)
    raw1 = mne.io.RawArray(data, raw_info)
    noise_cov = mne.compute_raw_covariance(raw1, tmin=0, tmax=None)

    inv_op = make_inverse_operator(raw_info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8)
    return inv_op

