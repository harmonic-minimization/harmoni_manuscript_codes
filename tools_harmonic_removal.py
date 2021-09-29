import mne
import numpy as np
import multiprocessing
from functools import partial
from numpy import pi
from tools_connectivity import compute_phaseconn_with_permtest
from tools_signal import hilbert_


def optimize_1_gridsearch(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-1, 1 + 0.01, 0.01)
    phi_range = np.arange(-pi/2, pi/2, pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_phaseconn_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return plv_sigx_yres_c_phi, c_range[ind_temp[0]], phi_range[ind_temp[1]]
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]


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
            c_abs_opt_1, c_phi_opt_1 = optimize_1_gridsearch(this_lbl_ts_high[n_b, :],
                                                             this_lbl_ts_low2[n_a, :], fs, coh)
            this_lbl_ts_high[n_b, :] = \
                this_lbl_ts_high[n_b, :] - c_abs_opt_1 * np.exp(1j * c_phi_opt_1) * this_lbl_ts_low2[n_a, :]
    this_lbl_ts_high *= this_lbl_high_std
    return this_lbl_ts_high


def regress_out(parcel_series_low, parcel_series_high, fs, n=2, coh=False, opt_strat='grid', mp=True, pool=None):
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

