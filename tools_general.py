import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
# saving and loading
#  --------------------------------  --------------------------------  --------------------------------

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