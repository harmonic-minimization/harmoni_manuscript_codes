import mne
import numpy as np
import multiprocessing
from functools import partial
from numpy import pi
from tools_connectivity import compute_plv_with_permtest
from tools_signal import hilbert_


def optimize_1_prime(sig_y_, sig_x2_, fs, return_all=False):
    c_range = np.arange(-1, 1 + 0.1, 0.2)
    phi_range = np.arange(-pi/2, pi/2, pi / 5)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs')
    ind_temp = np.argsort(np.abs(plv_sigx_yres_c_phi), axis=None)
    ind_temp1 = ind_temp[0]
    ind_temp1 = np.unravel_index(ind_temp1, plv_sigx_yres_c_phi.shape)
    ind_c_min = ind_temp1[0]
    ind_phi_min = ind_temp1[1]
    c1 = c_range[ind_c_min - 1] if ind_c_min != 0 else c_range[0]
    c2 = c_range[ind_c_min + 1] if ind_c_min != (len(c_range)-1) else c_range[-1]

    phi1 = c_range[ind_phi_min - 1] if ind_phi_min != 0 else phi_range[0]
    phi2 = c_range[ind_phi_min + 1] if ind_phi_min != len(phi_range)-1 else phi_range[-1]

    if return_all:
        plv_sigx_yres_c_phi_all, c_range_all, phi_range_all = plv_sigx_yres_c_phi, c_range, phi_range
    if np.abs(c1 - c2) <= 0.1:
        if c1 == 0:
            return 0, 0
        c_range = np.array([c1])
    else:
        c_range = np.arange(min([c1, c2]), max([c1, c2]) + 0.1, 0.1)
    if np.abs(phi1 - phi2) <= np.pi / 10:
        phi_range = np.array([phi1])
    else:
        phi_range = np.arange(min([phi1, phi2]), max([phi1, phi2]), pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return (plv_sigx_yres_c_phi_all, c_range_all, phi_range_all), (c_range[ind_temp[0]], phi_range[ind_temp[1]])
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]


def optimize_1(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-1, 1 + 0.1, 0.2)
    phi_range = np.arange(-pi/2, pi/2, pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.argsort(np.abs(plv_sigx_yres_c_phi), axis=None)
    ind_temp1 = ind_temp[0:2]
    ind_temp1 = np.unravel_index(ind_temp1, plv_sigx_yres_c_phi.shape)
    c1, c2 = c_range[ind_temp1[0][0]], c_range[ind_temp1[0][1]]
    phi1, phi2 = phi_range[ind_temp1[1][0]], phi_range[ind_temp1[1][1]]
    if return_all:
        plv_sigx_yres_c_phi_all, c_range_all, phi_range_all = plv_sigx_yres_c_phi, c_range, phi_range
    if np.abs(c1 - c2) <= 0.1:
        if c1 == 0:
            return 0, 0
        c_range = np.array([c1])
    else:
        c_range = np.arange(min([c1, c2]), max([c1, c2]) + 0.1, 0.1)
    if np.abs(phi1 - phi2) <= np.pi / 10:
        phi_range = np.array([phi1])
    else:
        phi_range = np.arange(min([phi1, phi2]), max([phi1, phi2]), pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return (plv_sigx_yres_c_phi_all, c_range_all, phi_range_all), (c_range[ind_temp[0]], phi_range[ind_temp[1]])
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]


def optimize_1_gridsearch(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-1, 1 + 0.01, 0.01)
    phi_range = np.arange(-pi/2, pi/2, pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return plv_sigx_yres_c_phi, c_range[ind_temp[0]], phi_range[ind_temp[1]]
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]


def _compute_grad(x, y, c1, phi1, dc, dphi, func):
    y_res_1 = y - c1 * np.exp(1j * phi1) * x
    y_res_dc = y - (c1 + dc) * np.exp(1j * phi1) * x
    y_res_dphi = y - c1 * np.exp(1j * (phi1 + dphi)) * x
    grad_c = (func(x, y_res_dc) - func(x, y_res_1)) / dc
    grad_phi = (func(x, y_res_dphi) - func(x, y_res_1)) / dphi
    return grad_c, grad_phi


def _compute_grad_2(x, y, c1, phi1, dc, dphi, func):
    y_res_c_1 = y - (c1-dc) * np.exp(1j * phi1) * x
    y_res_phi_1 = y - c1 * np.exp(1j * (phi1 - dphi)) * x
    y_res_dc = y - (c1 + dc) * np.exp(1j * phi1) * x
    y_res_dphi = y - c1 * np.exp(1j * (phi1 + dphi)) * x
    grad_c = (func(x, y_res_dc) - func(x, y_res_c_1)) / dc / 2
    grad_phi = (func(x, y_res_dphi) - func(x, y_res_phi_1)) / dphi / 2
    return grad_c, grad_phi


def compute_coherency_1(ts1, ts2, n, m, axis=1):
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
    nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
    nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
    denom = np.sqrt(np.mean(ts1_abs ** 2, axis=axis) * np.mean(ts2_abs ** 2, axis=axis))
    coh = nom / denom
    return np.abs(coh)


def optimize_gradient(y, x, dphi=1e-5, dc=1e-5):
    # from tools_connectivity import compute_coherency
    synch_func1 = lambda z1, z2: compute_coherency_1(z1, z2, 1, 1)
    c1 = np.random.random(1)[0] * 2 - 1
    phi1 = np.random.random(1)[0] * pi - pi / 2
    c1_old, phi1_old = c1, phi1
    # print(c1, phi1)
    value = [synch_func1(x, y - c1 * np.exp(1j * phi1) * x)[0]]
    for iter_n in range(500):
        factor = value[-1] / 0.1
        alpha = 0.01 * np.max([factor, 1])
        grad_c, grad_phi = _compute_grad_2(x, y, c1, phi1, dc, dphi, synch_func1)
        c1_old, phi1_old = c1, phi1
        c1 -= alpha * grad_c
        phi1 -= alpha * grad_phi
        value.append(synch_func1(x, y - c1 * np.exp(1j * phi1) * x)[0])
        # print(iter_n, '......', grad_c, grad_phi, '.....', c1, phi1, '.....', value[-1])
        if value[-1] - value[-2] > 0:
            break
    return c1_old, phi1_old, value


def optimize_c_gridsearch(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-1, 1 + 0.01, 0.01)
    plv_sigx_yres_c = np.empty((c_range.shape[0],))
    for n_c, c in enumerate(c_range):
        sig_res = sig_y_ - c * sig_x2_
        plv_sigx_yres_c[n_c] = compute_plv_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(np.abs(plv_sigx_yres_c)), plv_sigx_yres_c.shape)
    if return_all:
        return plv_sigx_yres_c, c_range[ind_temp[0]]
    return c_range[ind_temp[0]]


def _regress_out_roi(parcel_series_low, parcel_series_high, fs, coh, n, opt_strat, n_lbl):
    this_lbl_ts_low = hilbert_(parcel_series_low[n_lbl], axis=1)
    this_lbl_ts_high = hilbert_(parcel_series_high[n_lbl], axis=1)

    this_lbl_high_std = np.std(np.real(this_lbl_ts_high), axis=1)
    this_lbl_ts_high /= this_lbl_high_std
    this_lbl_ts_low /= np.std(np.real(this_lbl_ts_low), axis=1)
    # this_lbl_ts_alpha2 = this_lbl_ts_low ** 2
    this_lbl_ts_low2 = np.abs(this_lbl_ts_low) * np.exp(1j * np.angle(this_lbl_ts_low) * n)
    this_lbl_ts_low2 /= np.std(np.real(this_lbl_ts_low2), axis=1)

    n_low = this_lbl_ts_low.shape[0]
    n_high = this_lbl_ts_high.shape[0]

    for n_b in range(n_high):
        for n_a in range(n_low):
            if opt_strat == 'grad':
                c_abs_opt_1, c_phi_opt_1, _ = optimize_gradient(this_lbl_ts_high[n_b, :], this_lbl_ts_low2[n_a, :])
            elif opt_strat == 'grid':
                c_abs_opt_1, c_phi_opt_1 = optimize_1_gridsearch(this_lbl_ts_high[n_b, :],
                                                                 this_lbl_ts_low2[n_a, :], fs, coh)
            this_lbl_ts_high[n_b, :] = \
                this_lbl_ts_high[n_b, :] - c_abs_opt_1 * np.exp(1j * c_phi_opt_1) * this_lbl_ts_low2[n_a, :]
    this_lbl_ts_high *= this_lbl_high_std
    return this_lbl_ts_high


def regress_out(parcel_series_low, parcel_series_high, fs, n=2, coh=False, opt_strat='grad', mp=True, pool=None):
    n_parc = len(parcel_series_low)
    assert len(parcel_series_high) == n_parc

    if mp:
        pool = multiprocessing.Pool() if pool is None else pool
        func = partial(_regress_out_roi, parcel_series_low, parcel_series_high, fs, coh, n, opt_strat)
        parcel_series_high_corr = pool.map(func, range(n_parc))
        # pool.join()
        pool.close()

    else:
        parcel_series_high_corr = [None] * n_parc
        for i_lbl in range(n_parc):
            parcel_series_high_corr[i_lbl] = _regress_out_roi(parcel_series_low, parcel_series_high, fs,
                                                              coh, n, opt_strat, i_lbl)

    return parcel_series_high_corr


def harmonic_removal_simple(ts1, ts2, sfreq, n=2, method='grad', return_all=False):
    """
    a function for running harmoni on two single time series

    :param sfreq: [int], sampling frequency
    :param n: [int] the higher frequency is the n-th harmonic frequency of the lower frequency
    :param return_all: [bool] if True all the optimizing arguments are also return

    :return: the list of the corrected higher frequency signal
    """
    # exception handling ----------------
    assert(len(ts1.shape) <= 2), \
        'The input signals cannot have a dimension with size more than 2: raise for the first signal'
    assert (len(ts2.shape) <= 2), \
        'The input signals cannot have a dimension with size more than 2: raise for the second signal'

    if len(ts1.shape) == 2:
        assert(1 in ts1.shape), \
            "the input signals should be a single time series. Fix the dimensions, raise for the first input signal"
    else:
        ts1 = ts1.ravel()

    if len(ts2.shape) == 2:
        assert(1 in ts1.shape), \
            "the input signals should be a single time series. Fix the dimensions, raise for the second input signal"
    else:
        ts2 = ts2.ravel()

    assert (ts1.shape[0] == ts2.shape[0]), \
        'the number of time samples of the two signals should be the same'

    print('the harmonic correction has started, it may take a while ... ')
    ts1_h = hilbert_(ts1)
    ts1_ = np.abs(ts1_h) * np.exp(1j * n * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(ts2) / np.std(np.real(ts2))

    if method == 'grid':
        plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
    elif method == 'grad':
        c_opt, phi_opt, plv_sigx_yres_c_phi_all = optimize_gradient(ts2_, ts1_, dphi=1e-5, dc=1e-5)
    ts2_corr = ts2_ - c_opt * np.exp(1j * phi_opt) * ts1_
    if return_all:
        return ts2_corr, c_opt, phi_opt, plv_sigx_yres_c_phi_all
    return ts2_corr
