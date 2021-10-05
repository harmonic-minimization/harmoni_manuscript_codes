import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#  --------------------------------  --------------------------------  --------------------------------
# general: strings, saving, loading, directories
#  --------------------------------  --------------------------------  --------------------------------

def strround(x, n=3):
    return str(np.round(x, n))


def combine_names(connector, *nouns):
    word = nouns[0]
    for noun in nouns[1:]:
        word += connector + str(noun)
    return word


def save_pickle(file, var):
    import pickle
    with open(file, "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(file):
    import pickle
    with open(file, "rb") as input_file:
        var = pickle.load(input_file)
    return var


def listdir_restricted(dir_path, string_criterion):
    """
    returns a list of the names of the files in dir_path, whose names contains string_criterion
    :param dir_path: the directory path
    :param string_criterion: the string which should be in the name of the desired files in dir_path
    :return: list of file names
    """
    import os
    IDs_all = os.listdir(dir_path)
    IDs_with_string = [id1 for id1 in IDs_all if string_criterion in id1]
    return IDs_with_string


#  --------------------------------  --------------------------------  --------------------------------
# vectors, matrices and tensors
#  --------------------------------  --------------------------------  -------------------------------

def demean(x, axis=-1):
    return x - np.mean(x, axis=axis, keepdims=True)

def zscore_matrix(mat):
    return (mat - np.mean(mat)) / np.std(mat)


def threshold_matrix(mat, perc=95, threshold=None, binary=False):
    mat1 = mat.copy()
    if perc is not None:
        threshold = np.percentile(mat1, perc)
    mat1[mat1 < threshold] = 0
    if binary:
        mat1[mat1 != 0] = 1
    return mat1


def graph_roc(groundtruth_graph, test_graph):
    from tools_general import threshold_matrix
    G = threshold_matrix(groundtruth_graph, perc=None, threshold=np.max(groundtruth_graph), binary=True)
    # G = blur_matrix(G, 2)
    G = (G > 0) + 0
    Gnot = np.mod(G + 1, 2)

    thresh_range = np.arange(0, 1.01, .01)[::-1]
    tp = np.zeros(thresh_range.shape)
    fp = np.zeros(thresh_range.shape)
    for n_thresh, thresh in enumerate(thresh_range):
        test_graph_th_1 = threshold_matrix(test_graph, perc=None, threshold=thresh, binary=False)
        tp[n_thresh] = np.sum(test_graph_th_1 * G) / np.sum(G * test_graph)
        fp[n_thresh] = np.sum(test_graph_th_1 * Gnot) / np.sum(Gnot * test_graph)
    return fp, tp

#  --------------------------------  --------------------------------  --------------------------------
# plotting
#  --------------------------------  --------------------------------  --------------------------------

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_matrix(mat, cmap='viridis', cmap_level_n=50, title='', vmin=None, vmax=None, axes=None):
    from matplotlib import cm as cm
    cmp = cm.get_cmap(cmap, cmap_level_n)
    matmin = np.min(mat)
    matabsmax = np.max(np.abs(mat))
    if vmax is None:
        vmax = matabsmax
    else:
        vmax = max([vmax, matabsmax])
    if matmin < 0:
        cmp = 'RdBu_r'
    if vmin is None:
        vmin = matmin

    if axes is None:
        # caxes = plt.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax
        caxes = plt.matshow(mat, cmap=cmp, norm=MidpointNormalize(midpoint=0., vmin=vmin, vmax=vmax))

        plt.colorbar(caxes)
    else:
        # caxes = axes.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax)
        caxes = axes.matshow(mat, cmap=cmp, norm=MidpointNormalize(midpoint=0., vmin=vmin, vmax=vmax))
        plt.colorbar(caxes)
    plt.title(title)
    plt.grid(False)


def plot_3d(x, y, z):
    from mpl_toolkits import mplot3d
    t1, t2 = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(t1.T, t2.T, z, cmap='viridis', edgecolor='none')
    return ax

def plot_boxplot_paired(ax, data1, data2, labels, paired=True, violin=False, notch=True, datapoints=False):
    # does not open a new figure
    n_points = data1.shape[0]
    data1 = np.reshape(data1, (n_points, 1))
    data2 = np.reshape(data2, (n_points, 1))

    if violin:
        import seaborn as sns
        from pandas import DataFrame
        data_all = np.append(data1, data2)
        type = np.append([labels[0]] * n_points, [labels[1]] * n_points)
        data_all = {'data_all': data_all, 'type': type}
        df_thresh = DataFrame(data=data_all)
        sns.set_theme(style="whitegrid")
        ax = sns.violinplot(x='type', y='data_all', data=df_thresh)
        x1, x2 = 0, 1
    else:
        plt.boxplot(np.concatenate((data1, data2), axis=1), labels=labels, notch=notch)
        x1, x2 = 1, 2

    if datapoints:
        for k in range(n_points):
            plt.plot(np.ones((1, 1)) * x1 + np.random.randn(1, 1) * 0.02, data1[k],
                     marker='.', color='lightskyblue', markersize=3)
            plt.plot(np.ones((1, 1)) * x2 + np.random.randn(1, 1) * 0.02, data2[k],
                     marker='.', color='lightskyblue', markersize=3)

    if paired:
        for k in range(n_points):
            x = np.array([x1, x2])
            y = np.array([data1[k], data2[k]])
            plt.plot(x, y, '-', linewidth=.05)
    ax.yaxis.grid(True)


def plot_colorbar2(data, cmap, ax, ori='vertical'):
    import matplotlib as mpl
    data = np.sort(data, axis=0)
    norm = mpl.colors.Normalize(vmin=data[0], vmax=data[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)


#  --------------------------------  --------------------------------  --------------------------------
# stat, information theory
#  --------------------------------  --------------------------------  --------------------------------

def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """
    (c) https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------

    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    import scipy.stats as stats
    import scipy as sp
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        # pc = sp.polyfit(xs, ys + resamp_resid, 1)
        res = stats.linregress(xs, ys + resamp_resid)
        y_hat1 = res[0] * xs + res[1]
        # Plot bootstrap cluster
        ax.plot(xs, y_hat1, "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax


def plot_scatterplot_linearReg_bootstrap(x, y, ax, xlabel='', ylabel='', title=''):
    from scipy.stats import linregress
    x, y = x.ravel(), y.ravel()
    res = linregress(x, y)
    y_hat = res[0] * x + res[1]
    ax.plot(
        x, y, "o", color="#b9cfe7", markersize=8,
        markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
    )
    ax.plot(x, y_hat, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")
    plot_ci_bootstrap(x, y, y - y_hat, nboot=500, ax=None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)


def significance_perc_increase(x, y):
    from scipy.stats import pearsonr, norm
    n = len(x)
    var_x = np.sum((x - np.mean(x))**2) / (n - 1)
    vx = np.sqrt(var_x) / np.mean(x)
    var_y = np.sum((y - np.mean(y))**2) / (n - 1)
    vy = np.sqrt(var_y) / np.mean(y)
    k = vx / vy
    r_obs, pval_init = pearsonr(x, y / x)
    rxy, _ = pearsonr(x, y)
    r0 = -np.sqrt((1 - rxy) / 2)
    zscore = np.sqrt(n - 3)/2 * (np.log((1 + r_obs) / (1 - r_obs)) - np.log((1 + r0) / (1 - r0)))
    p_value = norm.sf(abs(zscore))  # one sided
    return p_value, zscore, r_obs, r0, pval_init, k

