import numpy as np
import mne
from scipy.fftpack import fft
from mne.parallel import parallel_func
from matplotlib import pyplot as plt
import os.path as op


def label_idx_whole_brain(src, label):
    """
    finding the vertex numbers corresponding to vertices in parcel specified in label
    :param src: source spaces
    :param label: label object of the desired parcel
    :return: parc_idx: indexes of the vertices in label,
             parc_hemi_idx: indexes of the vertices in label, but in the corresponding hemisphere
    """
    offset = src[0]['vertno'].shape[0]
    this_hemi = 0 if (label.hemi == 'lh') else 1
    idx, _, parc_hemi_idx = np.intersect1d(label.vertices, src[this_hemi]['vertno'], return_indices=True)
    # parc_hemi_idx = np.searchsorted(src[this_hemi]['vertno'], idx)
    parc_idx = parc_hemi_idx + offset * this_hemi

    return parc_idx, parc_hemi_idx



def _extract_svd(data, src, n, label):
    from sklearn.decomposition import TruncatedSVD
    """
    - apply svd on time series of the voxels of the given parcel
    - if n=None, as many as the svd components are selected that they explain >=95% of the variance
    - if n=n_select, n_select components are selected
    :param data: [vertex x time]
    :param n:
    :return:
    """
    lbl_idx, _ = label_idx_whole_brain(src, label)
    data_lbl = data[lbl_idx, :]
    svd = TruncatedSVD(n_components=n)
    svd.fit(data_lbl)
    component = svd.components_ * svd.singular_values_[np.newaxis, :].T
    return component


def _extract_svd_par(data, labels, src, var_perc, n, ind_lbl):
    """
    - apply svd on time series of the voxels of the given parcel
    - if n=None, as many as the svd components are selected that they explain >=95% of the variance
    - if n=n_select, n_select components are selected
    :param data: [vertex x time]
    :param n:
    :return:
    """
    label = labels[ind_lbl]
    lbl_idx, _ = label_idx_whole_brain(src, label)
    data_lbl = data[lbl_idx, :]
    u, s, _ = np.linalg.svd(data_lbl.T, full_matrices=False)
    if n is None:
        var_explained = np.cumsum(s ** 2 / np.sum(s ** 2))
        ind1 = np.where(var_explained >= var_perc)[0]
        if len(ind1):
            n_select = ind1[0] + 1
            component = u[:, 0:n_select] * s[0:n_select]
        else:
            component = np.zeros((data.shape[1], 1))
    else:
        component = u[:, :n] * s[:n]
    return component


def extract_parcel_time_series(data, labels, src, mode='svd', n_select=1, fs=None, freqs=None, n_jobs=1):
    """

    :param stc:
    :param labels: should be of class ModifiedLabel
    :param src:
    :param mode:
    :param n_select:
    :param n_jobs:
    :return:
    """
    from mne.parallel import parallel_func
    import itertools
    from functools import partial

    n_parc = len(labels)

    if mode == 'svd':
        # label_ts = mne.extract_label_time_course(stc, labels, src, mode='pca_flip', return_generator=False)
        # TODO: hack the function to extract the spatial patterns and build ParcSeries
        # parcel_series = label_ts
        # parallel, extract_svd, _ = parallel_func(_extract_svd, n_jobs)
        # parcel_series = parallel(extract_svd(stc.data, label, src, n=n_select) for label in labels[0:2])
        # n_samples = stc.data.shape[1]
        print('applying svd on each parcel... it may take some time')
        parcel_series = [None] * n_parc
        # TODO: n_jobs>1 does not work!!!!
        func = partial(_extract_svd, data, src, n_select)
        parallel, extract_svd_prl, _ = parallel_func(func, n_jobs=n_jobs)
        parcel_series = parallel(extract_svd_prl(label) for label in labels)
        #
        # for this_parc, label in enumerate(labels):
        #     this_parc == int(n_parc/2) and print('... We are half way done! ;-)')
        #     parcel_series[this_parc] = _extract_svd(data, label, src, n=n_select)
    return parcel_series
