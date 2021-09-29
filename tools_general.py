import numpy as np
import matplotlib.pyplot as plt


#  --------------------------------  --------------------------------  --------------------------------
# plotting
#  --------------------------------  --------------------------------  --------------------------------


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