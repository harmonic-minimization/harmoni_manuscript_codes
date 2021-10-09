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
from scipy.signal import filtfilt, butter
from tools_signal import hilbert_

def data_fun_pink_noise(times1):
    import colorednoise as cn
    n_sample = len(times1)
    data = cn.powerlaw_psd_gaussian(1, n_sample)
    data /= np.linalg.norm(data)
    return data


def adjust_snr(sig, noise, snr, freq):
    b2, a2 = butter(2, freq, btype='bandpass')
    noise_nb = filtfilt(b2, a2, noise)
    noise_var = np.mean(noise_nb**2)
    sig_var = np.mean(sig**2)
    snr_current = sig_var / noise_var
    factor1 = np.sqrt(snr_current / snr)
    return factor1


def filtered_randn(f1, f2, sfreq, n_time_samples, n_sig=1):
    x1 = np.random.randn(n_sig, n_time_samples)
    b1, a1 = butter(N=2, Wn=np.array([f1, f2])/sfreq*2, btype='bandpass')
    x1 = filtfilt(b1, a1, x1, axis=1)
    x1_h = hilbert_(x1)
    return x1_h


def produce_nm_phase_locked_sig(sig, phase_lag, n, m, wn_base, sfreq, nonsin_mode=2, kappa=None):
    """

    :param sig:
    :param phase_lag:
    :param n:
    :param m:
    :param wn_base:
    :param sfreq:
    :param kappa:
                if None, the signals are completely locked to each other
    :return:
    """

    if not np.iscomplexobj(sig):
        sig = hilbert_(sig)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]

    sig_angle = np.angle(sig)
    n_samples = sig.shape[1]
    if nonsin_mode == 2: # the same amplitude envelopes
        sig_abs = np.abs(sig)
    else:
        sig_ = filtered_randn(m*wn_base[0], m*wn_base[1], sfreq, n_samples)
        sig_abs = np.abs(sig_)
    if kappa is None:
        sig_hat = sig_abs * np.exp(1j * m / n * sig_angle + 1j * phase_lag)
    # TODO: kappa von mises
    return sig_hat

