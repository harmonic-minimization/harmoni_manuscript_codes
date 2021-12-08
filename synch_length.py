import numpy as np
from harmoni.harmonitools import  harmonic_removal_simple, optimize_1_gridsearch
from scipy.signal import butter, filtfilt
from tools_signal import *
from numpy import pi
from tools_connectivity import compute_phase_connectivity
from tools_general import *
from tools_simulations import data_fun_pink_noise, filtered_randn, produce_nm_phase_locked_sig, adjust_snr


fs = 256  # sampling frequency


b10, a10 = butter(N=2, Wn=np.array([8, 12])/fs*2, btype='bandpass')
b20, a20 = butter(N=2, Wn=np.array([16, 24])/fs*2, btype='bandpass')

maxiter = 5000
random_coh = np.zeros((maxiter,))
n_samples = int(1*fs)  # number of time samples
times = np.arange(0, n_samples)/fs  # the time points - used for plotting purpose

for n_iter in range(maxiter):
    print(n_iter)
    z = np.random.randn(1, n_samples)
    x = filtfilt(b10, a10, z)
    z = np.random.randn(1, n_samples)
    y = filtfilt(b20, a20, z)
    random_coh[n_iter] = compute_phase_connectivity(x, y, 1, 2, 'coh', type1='abs')

data = [np.sort(random_coh)]
parts = plt.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3, perc95, perc99 = np.percentile(data, [25, 50, 75, 95, 99], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
plt.hlines([perc95, perc99], 0, 1.5, linestyle='--')

# y = produce_nm_phase_locked_sig(x, 0, 1, 2, [8, 12], fs, nonsin_mode=1)
compute_phase_connectivity(x, y, 1, 2, 'coh', type1='abs')

z = np.random.randn(1, n_samples)
x = filtfilt(b10, a10, z)
y = produce_nm_phase_locked_sig(x, 0, 1, 2, [8, 12], fs, nonsin_mode=1)

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value






# -------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)


ax2.set_title('Customized violin plot')
parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
