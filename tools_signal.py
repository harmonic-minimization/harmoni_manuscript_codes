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
import matplotlib.pyplot as plt


#  --------------------------------  --------------------------------  --------------------------------
# general
#  --------------------------------  --------------------------------  --------------------------------

def dB(data, coeff=10):
    return coeff * np.log10(data)


def dBinv(data, coeff=10):
    return 10 ** (data / 10)


def zero_pad_to_pow2(x, axis=1):
    """
    for fast computation of fft, zeros pad the signals to the next power of two
    :param x: [n_signals x n_samples]
    :param axis
    :return: zero-padded signal
    """
    n_samp = x.shape[axis]
    n_sig = x.shape[1-axis]
    n_zp = int(2 ** np.ceil(np.log2(n_samp))) - n_samp
    zp = np.zeros_like(x)
    zp = zp[:n_zp] if axis == 0 else zp[:, :n_zp]
    y = np.append(x, zp, axis=axis)
    return y, n_zp

#  --------------------------------  --------------------------------  --------------------------------
# frequency domain and complex signals
#  --------------------------------  --------------------------------  --------------------------------
def fft_(x, fs, axis=1, n_fft=None):
    from scipy.fftpack import fft
    if np.iscomplexobj(x):
        x = np.real(x)
    if x.ndim == 1:
        x = x.reshape((1, x.shape[0])) if axis == 1 else x.reshape((x.shape[0], 1))
    n_sample = x.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(n_sample))) if n_fft is None else n_fft
    x_f = fft(x, n_fft)
    freq = np.arange(0, fs / 2, fs / n_fft)
    n_fft2 = int(n_fft / 2)
    x_f = x_f[0, : n_fft2]
    return freq, x_f


def plot_fft(x, fs, axis=1, n_fft=None):
    freq, x_f = fft_(x, fs, axis=axis, n_fft=n_fft)
    xf_abs = np.abs(x_f)
    plt.plot(freq, xf_abs.ravel())
    plt.title('Magnitude of FFT')
    plt.grid()


def hilbert_(x, axis=1):
    """
    computes fast hilbert transform by zero-padding the signal to a length of power of 2.


    :param x: array_like
              Signal data.  Must be real.
    :param axis: the axis along which the hilbert transform is computed, default=1
    :return: x_h : analytic signal of x
    """
    if np.iscomplexobj(x):
        return x
    from scipy.signal import hilbert
    if len(x.shape) == 1:
        x = x[np.newaxis, :] if axis == 1 else x[:, np.newaxis]
    x_zp, n_zp = zero_pad_to_pow2(x, axis=axis)
    x_zp = np.real(x_zp)
    x_h = hilbert(x_zp, axis=axis)
    x_h = x_h[:, :-n_zp] if axis == 1 else x_h[:-n_zp, :]
    return x_h


def psd(data, fs, f_max=None, overlap_perc=0.5, freq_res=0.5, axis=1, plot=True, dB1=True,
        fig='new', interactivePlot=True, clab=None):
    """
    plots the spectrum of the input signal

    :param data: ndarray [n_chan x n_samples]
                 data array . can be multi-channel
    :param fs: sampling frequency
    :param f_max: maximum frequency in the plotted spectrum
    :param overlap_perc: overlap percentage of the sliding windows in welch method
    :param freq_res: frequency resolution, in Hz
    :return: no output, plots the spectrum
    """
    from scipy.signal import welch
    if np.iscomplexobj(data):
        data = np.real(data)
    if data.ndim == 1:
        axis = 0
    nfft = 2 ** np.ceil(np.log2(fs / freq_res))
    noverlap = np.floor(overlap_perc * nfft)
    f, pxx = welch(data, fs=fs, nfft=nfft, nperseg=nfft, noverlap=noverlap, axis=axis)
    if f_max is not None:
        indices = {axis: f <= f_max}
        ix = tuple(indices.get(dim, slice(None)) for dim in range(pxx.ndim))
        pxx = pxx[ix]
        f = f[f <= f_max]
    if plot:
        if fig == 'new':
            fig = plt.figure()
            ax = plt.subplot(111)
        else:
            fig, ax = fig[0], fig[1]
        if dB1:
            line = ax.plot(f, dB(pxx.T), lw=1, picker=1)
        else:
            line = ax.plot(f, pxx.T, lw=1, picker=1)
        if interactivePlot:
            def onpick1(event, clab):
                thisline = event.artist
                n_line = int(str(thisline)[12:-1])
                if clab is not None:
                    print(clab[n_line])
                else:
                    print('channel ' + str(n_line))

            onpick = lambda event: onpick1(event, clab)
            fig.canvas.mpl_connect('pick_event', onpick)
        plt.ylabel('PSD (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True, ls='dotted')
        return f, pxx, (fig, ax, line)
    return f, pxx
