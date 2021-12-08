"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------
"""
import numpy as np
from tools_signal import hilbert_
import multiprocessing
from functools import partial
import itertools


def compute_phase_connectivity(ts1, ts2, n, m, measure='coh', axis=1, type1='complex'):
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]
    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    if measure == 'coh':
        nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        denom = np.sqrt(np.mean(ts1_abs ** 2, axis=axis) * np.mean(ts2_abs ** 2, axis=axis))
        coh = nom / denom
        if type1 == 'abs':
            coh = np.abs(coh)
        elif type1 == 'imag':
            coh = np.imag(coh)
        return coh
    elif measure == 'plv':
        plv = np.mean(np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        plv = np.abs(plv) if type1 == 'abs' else plv
        if type1 == 'abs':
            plv = np.abs(plv)
        elif type1 == 'imag':
            plv = np.imag(plv)
        return plv


def compute_plv(ts1, ts2, n, m, plv_type='abs', coh=False):
    """
    computes complex phase locking value.
    :param ts1: array_like [channel x time]
                [channel x time] multi-channel time-series (real or complex)
    :param ts2: array_like
                [channel x time] multi-channel time-series (real or complex)
    :param n: the ratio of coupling of x and y is n:m
    :param m: the ratio of coupling of x and y is n:m
    :param plv_type: 'complex', 'abs' (default), 'imag'
    :param coh: Bool
    :return: plv: phase locking value
    """
    # TODO: test for multichannel
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]

    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    cplv = np.zeros((nchan1, nchan2), dtype='complex')
    for chan1, ts1_ph_chan in enumerate(ts1_ph):
        for chan2, ts2_ph_chan in enumerate(ts2_ph):
            if coh:
                cplv[chan1, chan2] = np.mean(
                    ts1_abs[chan1, :] * ts2_abs[chan2, :] * np.exp(1j * m * ts1_ph_chan - 1j * n * ts2_ph_chan)) / \
                                     np.sqrt(np.mean(ts1_abs[chan1, :] ** 2)) / np.sqrt(np.mean(ts2_abs[chan2, :] ** 2))
            else:
                cplv[chan1, chan2] = np.mean(np.exp(1j * m * ts1_ph_chan - 1j * n * ts2_ph_chan))

    if plv_type == 'complex':
        return cplv
    elif plv_type == 'abs':
        return np.abs(cplv)
    elif plv_type == 'imag':
        return np.imag(cplv)


def compute_plv_windowed(ts1, ts2, n, m, winlen='whole', overlap_perc=0.5, plv_type='abs', coh=False):
    # TODO: test for multichannel
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]
    if winlen == 'whole':
        winlen = n_samples
    winlen_start = int(np.floor(winlen * (1 - overlap_perc)))

    stop_flag = 0
    n_loop = 0
    plv = np.zeros((nchan1, nchan2), dtype='complex')
    while not stop_flag:
        n_start = n_loop * winlen_start
        n_stop = n_start + winlen
        if n_stop <= n_samples:
            ts1_ = ts1[:, n_start:n_stop]
            ts2_ = ts2[:, n_start:n_stop]
            plv += compute_plv(ts1_, ts2_, n, m, plv_type=plv_type, coh=coh)
            n_loop += 1
        else:
            stop_flag = 1
    plv /= n_loop
    if not plv_type == 'complex':
        plv = np.real(plv)
    return plv


def compute_phaseconn_with_permtest(ts1, ts2, n, m, sfreq, seg_len=None, plv_type='abs', coh=False,
                                    iter_num=0, plv_win='whole', verbose=False):
    """
        do permutation test for testing the significance of the PLV between ts1 and ts2
        :param ts1: [chan1 x time]
        :param ts2: [chan2 x time]
        :param n:
        :param m:
        :param sfreq:
        :param seg_len:
        :param plv_type:
        :param coh:
        :param iter_num:
        :param plv_win:
        :param verbose:
        :return:
        plv_true: the true PLV
        plv_sig: the pvalue of the permutation test
        plv_stat: the statistics: the ratio of the observed plv to the mean of the permutatin PLVs
        plv_perm: plv of the iterations of the permutation test
        """
    # TODO: test for multichannel
    # TODO: verbose
    # ToDo: add axis
    # setting ------------------------------
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    seg_len = int(sfreq) if seg_len is None else int(seg_len)
    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]

    if plv_win is None:
        plv_winlen = sfreq
    elif plv_win == 'whole':
        plv_winlen = n_samples

    plv_true = compute_plv_windowed(ts1, ts2, n, m, winlen=plv_winlen, plv_type=plv_type, coh=coh)
    if nchan1 == 1 and nchan2 == 1:
        plv_true = np.reshape(plv_true, (1, 1))

    n_seg = int(n_samples // seg_len)
    n_omit = int(n_samples % seg_len)
    ts1_rest = ts1[:, -n_omit:] if n_omit > 0 else np.empty((nchan1, 0))
    ts1_truncated = ts1[:, :-n_omit] if n_omit > 0 else ts1
    ts1_truncated = ts1_truncated.reshape((nchan1, seg_len, n_seg), order='F')

    if not plv_type == 'complex':
        plv_true = np.real(plv_true)

    plv_perm = -1
    pvalue = -1
    if iter_num:
        # TODO: seeds should be corrected
        seeds = np.random.choice(range(10000), size=iter_num, replace=False)
        plv_perm = np.zeros((nchan1, nchan2, iter_num), dtype='complex')
        for n_iter in range(iter_num):
            if verbose:
                print('iteration ' + str(n_iter))
            np.random.seed(seed=seeds[n_iter])
            perm1 = np.random.permutation(n_seg)
            ts1_perm = ts1_truncated[:, :, perm1].reshape((nchan1, n_samples - n_omit), order='F')
            ts1_perm = np.concatenate((ts1_perm, ts1_rest), axis=1)
            plv_perm[:, :, n_iter] = compute_plv_windowed(ts1_perm, ts2, n, m,
                                                          winlen=plv_winlen, plv_type=plv_type, coh=coh)

        plv_perm = np.abs(plv_perm)
        pvalue = np.zeros((nchan1, nchan2))
        for c1 in range(nchan1):
            for c2 in range(nchan2):
                plv_perm1 = np.squeeze(plv_perm[c1, c2, :])
                pvalue[c1, c2] = np.mean(plv_perm1 >= np.abs(plv_true)[c1, c2])

    if iter_num:
        return plv_true, pvalue, plv_perm
    else:
        return plv_true



def compute_conn(n, m, fs, plv_type, signals):
    x = signals[0]
    y = signals[1]
    conn = np.mean(compute_phaseconn_with_permtest(x, y, n, m, fs, plv_type=plv_type))
    return conn


def compute_conn_2D_parallel(ts_list1, ts_list2, n, m, fs, plv_type):
    list_prod = list(itertools.product(ts_list1, ts_list2))
    pool = multiprocessing.Pool()
    func = partial(compute_conn, n, m, fs, plv_type)
    conn_mat_beta_corr_list = pool.map(func, list_prod)
    pool.close()
    pool.join()
    conn_mat = np.asarray(conn_mat_beta_corr_list).reshape((len(ts_list1), len(ts_list2)))
    return conn_mat


def _synch_perm(n, m, seg_len, measure, type1, sig):
    x = sig[0]  # x.ndim = 1
    y = sig[1]
    seed1 = sig[2]

    if seed1 == -1:
        x_perm = x
    else:
        n_samples = x.shape[0]
        n_seg = int(n_samples // seg_len)
        n_omit = int(n_samples % seg_len)
        x_rest = x[-n_omit:] if n_omit > 0 else np.empty((0,))
        x_truncated = x[:-n_omit] if n_omit > 0 else x
        x_truncated = x_truncated.reshape((1, seg_len, n_seg), order='F')
        np.random.seed(seed=seed1)
        perm1 = np.random.permutation(n_seg)
        x_perm = x_truncated[:, :, perm1].reshape((1, n_samples - n_omit), order='F')
        x_perm = np.concatenate((x_perm.ravel(), x_rest))

    return compute_phase_connectivity(x_perm, y, n, m, measure=measure, type1=type1)[0]


def compute_synch_permtest_parallel(ts1, ts2, n, m, sfreq, ts1_ts2_eq=False, type1='abs', measure='coh',
                                    seg_len=None, iter_num=1000):
    """
    parallelized permutation test for multiple channels

    :param ts1: [nchan1 x time]
    :param ts2: [nchan2 x time]
    :param n:
    :param m:
    :param sfreq:
    :param ts1_ts2_eq: if True then only the upper triangle is computed. Put True only if ts1 and ts2 are identical and n = m = 1
    :param type1: {'abs', 'imag'}.
    :param measure: {'coh', 'plv'}
    :param seg_len: length of the segment for permutation testing
    :param iter_num: number of iterations for the perm test
    :return:
    conn_orig [nchan1 x nchan2]: the original true values of the conenctivity
    pvalue [nchan1 x nchan2]: the pvalues of the connections
    """
    if ts1.ndim == 1:
        ts1 = ts1[np.newaxis, :]
    if ts2.ndim == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    seg_len = int(sfreq) if seg_len is None else seg_len
    nchan1 = ts1.shape[0]
    nchan2 = ts2.shape[0]

    seeds = np.random.randint(low=0, high=2 ** 32, size=(iter_num,))
    seeds = np.append(-np.ones((1,), dtype='int64'), seeds)
    list_prod = list(itertools.product(ts1, ts2, seeds))

    if ts1_ts2_eq:
        ind_triu = np.triu_indices(nchan1, k=1)
        l1 = np.reshape(np.arange(len(list_prod)), (nchan1, nchan2, iter_num + 1))
        list_prod_ind = list(l1[ind_triu].ravel())
        list_prod = [list_prod[i] for i in list_prod_ind]

    pool = multiprocessing.Pool()
    func = partial(_synch_perm, n, m, seg_len, measure, type1)
    synch = pool.map(func, list_prod)
    pool.close()

    if ts1_ts2_eq:
        c1 = np.asarray(synch).reshape((-1, iter_num + 1))
        conn_mat = np.zeros((nchan1, nchan2, iter_num + 1))
        conn_mat[ind_triu] = c1
        conn_mat = conn_mat + np.transpose(conn_mat, axes=(1, 0, 2))
    else:
        conn_mat = np.asarray(synch).reshape((nchan1, nchan2, iter_num + 1))
    conn_mat = np.abs(conn_mat)
    conn_orig = conn_mat[:, :, 0]
    conn_perm = conn_mat[:, :, 1:]

    pvalue = np.zeros((nchan1, nchan2))
    pvalue_rayleigh = np.zeros((nchan1, nchan2))
    if ts1_ts2_eq:
        for i1 in range(nchan1):
            for i2 in range(i1 + 1, nchan2):
                pvalue[i1, i2] = np.mean(conn_perm[i1, i2, :] >= conn_orig[i1, i2])

                plv_perm1 = np.squeeze(conn_perm[i1, i2, :])
                plv_perm1_mean = np.mean(plv_perm1)
                plv_stat = conn_orig[i1, i2] / plv_perm1_mean
                pvalue_rayleigh[i1, i2] = np.exp(-np.pi * plv_stat ** 2 / 4)

        pvalue = pvalue + pvalue.T
        pvalue_rayleigh = pvalue_rayleigh + pvalue_rayleigh.T
    else:
        for i1 in range(nchan1):
            for i2 in range(nchan2):
                pvalue[i1, i2] = np.mean(conn_perm[i1, i2, :] >= conn_orig[i1, i2])

                plv_perm1 = np.squeeze(conn_perm[i1, i2, :])
                plv_perm1_mean = np.mean(plv_perm1)
                plv_stat = conn_orig[i1, i2] / plv_perm1_mean
                pvalue_rayleigh[i1, i2] = np.exp(-np.pi * plv_stat ** 2 / 4)
    return conn_orig, pvalue, pvalue_rayleigh


def random_synchronization_dist(n, m, duration, f0=10, fs=256, maxiter=5000):
    from scipy.signal import butter, filtfilt

    b1, a1 = butter(N=2, Wn=np.array([n*f0-2, n*f0+2]) / fs * 2, btype='bandpass')
    b2, a2 = butter(N=2, Wn=np.array([m*f0-4, m*f0+4]) / fs * 2, btype='bandpass')

    random_coh = np.zeros((maxiter,))
    n_samples = int(duration * fs)  # number of time samples

    print('the iterations started. It may take some time ...')
    for n_iter in range(maxiter):
        if n_iter == int(maxiter/2):
            print('half way done')
        z = np.random.randn(1, n_samples)
        x = filtfilt(b1, a1, z)
        z = np.random.randn(1, n_samples)
        y = filtfilt(b2, a2, z)
        random_coh[n_iter] = compute_phase_connectivity(x, y, n, m, 'coh', type1='abs')
    return random_coh

