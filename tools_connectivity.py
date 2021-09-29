import numpy as np
from tools_signal import hilbert_


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