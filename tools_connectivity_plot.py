
import numpy as np
from matplotlib import pyplot as plt


def rearrange_labels(labels, order='anterior_posterior'):

    """
    adopted from https://martinos.org/mne/stable/auto_examples/connectivity/plot_mne_inverse_label_connectivity.html
    :param labels: is the label list
    :return:
    """
    label_names = [label.name for label in labels]

    lh_labels_idx = [i for i in range(len(labels)) if labels[i].hemi == 'lh']
    rh_labels_idx = [i for i in range(len(labels)) if labels[i].hemi == 'rh']

    lh_labels = [label for label in labels if label.hemi == 'lh']
    rh_labels = [label for label in labels if label.hemi == 'rh']

    # Get the y-location of the label
    label_ypos_lh = list()
    for label in lh_labels:
        ypos = np.mean(label.pos[:, 1])
        label_ypos_lh.append(ypos)

        """
            if order == 'anterior_posterior':
        sort_idx_lh = np.argsort(label_ypos_lh)  # [::-1]
    elif  order == 'posterior_anterior':
        sort_idx_lh = np.argsort(label_ypos_lh)[::-1]
        """
    sort_idx_lh = np.argsort(label_ypos_lh)[::-1]
    lh_labels_idx = np.asarray([lh_labels_idx[idx] for idx in sort_idx_lh])

    label_ypos_rh = list()
    for label in rh_labels:
        ypos = np.mean(label.pos[:, 1])
        label_ypos_rh.append(ypos)

    sort_idx_rh = np.argsort(label_ypos_rh)
    rh_labels_idx = np.asarray([rh_labels_idx[idx] for idx in sort_idx_rh])

    labels_sorted = [None for lbl in labels]
    labels_sorted[0:len(lh_labels_idx)] = list(np.array(labels)[lh_labels_idx])
    labels_sorted[len(lh_labels_idx):] = list(np.array(labels)[rh_labels_idx])
    # labels_sorted = list(np.array(labels)[lh_labels_idx]).append(list(np.array(labels)[rh_labels_idx]))

    sorted_labels_idx = np.concatenate((lh_labels_idx, rh_labels_idx), axis=0)
    return labels_sorted, sorted_labels_idx


def rearrange_labels_network(labels):

    # node_colors = [label.color for label in labels]
    node_network = [label.name[10:-3] for label in labels]
    idx_sorted = np.argsort(node_network)
    labels_sorted = [labels[ii] for ii in idx_sorted]
    return labels_sorted, idx_sorted


def plot_mne_circular_connectivity(con_mat, labels, perc_conn=0.25, fig=None, cfc=False, subplot=111,
                                   fig_title=None, node_name=True, vmax=None, vmin=None,
                                   facecolor='black', colormap='hot', textcolor='white'):
    from mne.viz import circular_layout, plot_connectivity_circle
    fig_title = '' if fig_title is None else fig_title
    # prepare the figure ---------------------------------
    node_colors = [label.color for label in labels]

    # We reorder the labels based on their location in the left hemi
    # label_names = [label.name[13:] for label in labels]  # for schaefer
    label_names = [label.name for label in labels]
    # lh_labels = [name for name in label_names if name.endswith('lh')]
    # rh_labels = [name for name in label_names if name.endswith('rh')]

    lh_labels = [label.name for label in labels if label.hemi=='lh']
    rh_labels = [label.name for label in labels if label.hemi=='rh']

    # Get the y-location of the label
    label_ypos_lh = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos_lh.append(ypos)
    try:
        idx = label_names.index('Brain-Stem')
    except ValueError:
        pass
    else:
        ypos = np.mean(labels[idx].pos[:, 1])
        lh_labels.append('Brain-Stem')
        label_ypos_lh.append(ypos)
    # ---
    label_ypos_rh = list()
    for name in rh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos_rh.append(ypos)
    try:
        idx = label_names.index('Brain-Stem')
    except ValueError:
        pass
    else:
        ypos = np.mean(labels[idx].pos[:, 1])
        rh_labels.append('Brain-Stem')
        label_ypos_rh.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]
    rh_labels = [label for (yp, label) in sorted(zip(label_ypos_rh, rh_labels))]

    # Save the plot order
    node_order = lh_labels[::-1] + rh_labels

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) // 2])
    if not node_name:
        # label_names = [label.name[10:-3] for label in labels]
        label_names = [''] * len(label_names)
    else:
        label_names = [label.name[:-3] for label in labels]  # for DK
    con_mat = np.abs(con_mat)
    n_lines = int(np.prod(con_mat.shape)/2 * perc_conn)
    print(n_lines)
    if fig is None:
        fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
    if cfc:
        return plot_connectivity_circle_cfc(con_mat, label_names, n_lines=n_lines,
                                            node_angles=node_angles, node_colors=node_colors,
                                            title=fig_title, fig=fig, subplot=subplot, vmax=vmax)
    else:
        plot_connectivity_circle(con_mat, label_names, n_lines=n_lines,
                                 node_angles=node_angles, node_colors=node_colors,
                                 title=fig_title, fig=fig, subplot=subplot, vmax=vmax,  vmin=vmin,
                                 facecolor=facecolor, colormap=colormap, textcolor=textcolor)
    # plot_connectivity_circle_cfc(con_mat, label_names, n_lines=n_lines,
    #                          node_angles=node_angles, node_colors=node_colors,
    #                          title=fig_title, fig=fig, subplot=subplot)



def plot_mne_circular_connectivity_network(con_mat, labels, perc_conn=0.25, cfc=False,
                                           fig=None, subplot=111, fig_title=None, node_name=True,
                                           vmax=None, vmin=0, colormap='Blues',
                                           facecolor='white', textcolor='black'):
    from mne.viz import circular_layout, plot_connectivity_circle
    fig_title = '' if fig_title is None else fig_title
    # prepare the figure ---------------------------------
    node_colors = [label.color for label in labels]

    # We reorder the labels based on their location in the left hemi
    # label_names = [label.name[13:] for label in labels]  # for schaefer
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if name.endswith('lh')]
    rh_labels = [name for name in label_names if name.endswith('rh')]

    labels_network_sorted, idx_lbl_sort = rearrange_labels_network(labels)
    label_names_sorted = [label_names[ii] for ii in idx_lbl_sort]

    # Reorder the labels based on their location
    lh_labels = [name[:-3] for name in label_names_sorted if name.endswith('lh')]
    rh_labels = [name[:-3] for name in label_names_sorted if name.endswith('rh')]
    label_names = [name[:-3] for name in label_names]

    # Save the plot order
    node_order = lh_labels[::-1] + rh_labels

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) // 2])
    if not node_name:
        label_names = [''] * len(label_names)
    else:
        label_names = [label.name[13:-3] for label in labels]
    #con_mat = np.abs(con_mat)
    if perc_conn < 1:
        n_lines = int(np.prod(con_mat.shape)/2 * perc_conn)
    else:
        n_lines = None
    if fig is None:
        fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
    if cfc:
        return plot_connectivity_circle_cfc(con_mat, label_names, n_lines=n_lines,
                                            node_angles=node_angles, node_colors=node_colors,
                                            title=fig_title, fig=fig, subplot=subplot, vmax=vmax, vmin=vmin,
                                            facecolor=facecolor, colormap=colormap, textcolor=textcolor)
    else:
        plot_connectivity_circle(con_mat, label_names, n_lines=n_lines,
                                 node_angles=node_angles, node_colors=node_colors,
                                 title=fig_title, fig=fig, subplot=subplot, vmax=vmax,  vmin=vmin,
                                 facecolor=facecolor, colormap=colormap, textcolor=textcolor)



def plot_connectivity_circle_cfc(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, padding=6.,
                             fig=None, subplot=111,
                             node_linewidth=2., show=True):
    """
    (c) adopted from mne.viz.plot_connectivity_circle by Mina Jamshidi Idaji @ MPI CBS

    Visualize connectivity as a circular graph.

    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches
    from tools_general import plot_colorbar

    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        try:
            spectral = plt.cm.spectral
        except AttributeError:
            spectral = plt.cm.Spectral
        node_colors = [spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indd1 = np.tril_indices(n_nodes, -1)
        indd2 = np.triu_indices(n_nodes, -1)
        indices = (np.append(indd1[0], indd2[0]), np.append(indd1[1], indd2[1]))
        con_deg_1 = np.sum(con, axis=1)
        con_deg_2 = np.sum(con, axis=0)
        con = con[indices]

        max_val = np.max(np.append(con_deg_1, con_deg_2))
        cmap_range = np.arange(0, max_val, step=0.001)
        # cmap_deg1 = plt.get_cmap('YlGnBu', len(cmap_range))  # YlGnBu, Purples
        # # cmap_deg1 = cmap_deg1.reversed()
        # cmap_deg2 = plt.get_cmap('pink', len(cmap_range))  # YlGn
        # cmap_deg2 = cmap_deg2.reversed()
        #
        # node_colors_deg1 = [cmap_deg1(np.argmin(np.abs(cmap_range-deg1))) for deg1 in con_deg_1]
        # node_colors_deg2 = [cmap_deg2(np.argmin(np.abs(cmap_range - deg2))) for deg2 in con_deg_1]


        cmap_deg1 = plt.get_cmap('YlGnBu', len(con_deg_1)) # YlGnBu, Purples
        # cmap_deg1 = cmap_deg1.reversed()
        cmap_deg2 = plt.get_cmap('pink', len(con_deg_2)) # YlGn
        cmap_deg2 = cmap_deg2.reversed()

        node_colors_deg1 = [cmap_deg1(ii) for ii in np.argsort(np.argsort(con_deg_1))]
        node_colors_deg2 = [cmap_deg2(ii) for ii in np.argsort(np.argsort(con_deg_2))]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True)
    axes.set_facecolor(facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additional space if requested
    plt.ylim(0, 10 + padding)

    # Remove the black axes border which may obscure the labels
    axes.spines['polar'].set_visible(False)

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    del con_abs
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int64)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=1.)
        axes.add_patch(patch)

    # Draw ring with colored nodes for deg1
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors_deg1):
        bar.set_facecolor(color)

    # Draw ring with colored nodes for deg2
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=10,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors_deg2):
        bar.set_facecolor(color)
    # Draw ring with colored nodes for network
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=12,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        axes.text(angle_rad, 13.4, name, size=fontsize_names,
                  rotation=angle_deg, rotation_mode='anchor',
                  horizontalalignment=ha, verticalalignment='center',
                  color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin, vmax))
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    # plt_show(show)
    return (cmap_range, cmap_deg1, cmap_deg2), (np.linspace(vmin, vmax), colormap)



def _build_node_verts_bipartite(x0, y0, xw, yw):
    verts = [
        (x0, y0),  # left, bottom
        (x0, y0+yw),  # left, top
        (x0+xw, y0+yw),  # right, top
        (x0+xw, y0),  # right, bottom
        (x0, y0),  # ignored
    ]
    return verts


def _build_edge_verts_bipartite(x0, x1, y0, y1, xw, yw):
    verts = [
        (x0+xw/2, y0),  # left, bottom
        (x1+xw/2, y1+yw),  # left, top
    ]
    return verts


def _plot_a_node_bipartite(verts, ax, color):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=2)
    ax.add_patch(patch)


def _plot_an_edge_bipartite(verts, ax, color):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, edgecolor=color, lw=2)
    ax.add_patch(patch)



def plot_connectivity_bipartite_2_prime(con, labels, strength_perc, edge_cmp, fig_title='', vmin=None, vmax=None,
                                        only_lbl=None, arrange='network', edge_thresh=None):
    """
    similar to _2 - only with one line of nodes with hte color-code of the node
    """
    import itertools
    from tools_general import plot_colorbar2

    edge_thresh = np.percentile(con.ravel(), strength_perc) if edge_thresh is None else edge_thresh

    n_lbl = len(labels)
    label_names = [label.name for label in labels]

    if arrange == 'network':
        labels_network_sorted, idx_lbl_sort = rearrange_labels_network(labels)
    else:
        labels_network_sorted, idx_lbl_sort = rearrange_labels(labels)
    label_names_sorted = [label_names[ii] for ii in idx_lbl_sort]


    # label names
    lh_labels = [name[:-3] for name in label_names_sorted if name.endswith('lh')]
    rh_labels = [name[:-3] for name in label_names_sorted if name.endswith('rh')]

    # label colors
    label_color_lh = [lbl.color for lbl in labels_network_sorted if lbl.name.endswith('lh')]
    label_color_rh = [lbl.color for lbl in labels_network_sorted if lbl.name.endswith('rh')]

    # label indices
    ind_lbl_sorted_lh = [idx_lbl_sort[ii] for ii, lbl in enumerate(labels_network_sorted) if lbl.name.endswith('lh')]
    ind_lbl_sorted_rh = [idx_lbl_sort[ii] for ii, lbl in enumerate(labels_network_sorted) if lbl.name.endswith('rh')]

    if arrange == 'network':
        label_names = lh_labels + rh_labels[::-1]
        node_colors = label_color_lh + label_color_rh[::-1]
        node_ind = ind_lbl_sorted_lh + ind_lbl_sorted_rh[::-1]
    else:
        label_names = lh_labels + rh_labels
        node_colors = label_color_lh + label_color_rh
        node_ind = ind_lbl_sorted_lh + ind_lbl_sorted_rh
    labels_s = [labels[i] for i in node_ind]
    n_lbl_lh = len(lh_labels)
    con_lbl_sorted = con[:, node_ind][node_ind, :]
    if only_lbl is not None:
        con2 = np.zeros_like(con_lbl_sorted)
        con2[only_lbl, :] = con_lbl_sorted[only_lbl, :]
        con2[:, only_lbl] = con_lbl_sorted[:, only_lbl]
        con_lbl_sorted = con2
    # edge and node colormap ---------
    vmin = edge_thresh if vmin is None else vmin
    vmax = np.max(con) if vmax is None else vmax
    edge_cmap_range = np.arange(vmin, vmax, step=0.001)
    cmap_edge = plt.get_cmap(edge_cmp, len(edge_cmap_range))

    # start x-coordinates ---------
    xw, yw = 1, 0.1
    nodes_x_start_lh = np.arange(0, len(lh_labels))
    nodes_x_start_rh = np.arange(n_lbl_lh+1, len(labels)+1)
    nodes_x_start = np.append(nodes_x_start_lh, nodes_x_start_rh)

    # plot
    fig = plt.figure()

    # plot the edges ---------
    ax1 = plt.subplot2grid((8, 2), (0, 0), colspan=2, rowspan=7)
    ax1.set_xlim(-5, n_lbl + 5)
    ax1.set_ylim(-2, 2)
    con_array = con_lbl_sorted.ravel(order='C')
    con_array_ind = list(itertools.product(np.arange(0, n_lbl), np.arange(0, n_lbl)))
    con_array_argsort = np.argsort(con_array)

    for k in con_array_argsort:
        edge_strength = con_array[k]
        i1, i2 = con_array_ind[k]
        # if i1 == i2:
        #     continue
        if edge_strength >= edge_thresh and edge_strength > 0:
            this_edge_color = cmap_edge(np.argmin(np.abs(edge_cmap_range - edge_strength)))
            verts = _build_edge_verts_bipartite(nodes_x_start[i1], nodes_x_start[i2], 1, -1, xw, yw)
            _plot_an_edge_bipartite(verts, ax1, this_edge_color)

    # plot the nodes  ----------
    for i_lbl, nodes_x in enumerate(nodes_x_start):
        this_node_color =node_colors[i_lbl][:-1]
        verts = _build_node_verts_bipartite(nodes_x, 1, xw, yw)
        _plot_a_node_bipartite(verts, ax1, this_node_color)

    for i_lbl, nodes_x in enumerate(nodes_x_start):
        this_node_color = node_colors[i_lbl][:-1]
        verts = _build_node_verts_bipartite(nodes_x, -1, xw, yw)
        _plot_a_node_bipartite(verts, ax1, this_node_color)

    ax1.title.set_text(fig_title)
    plt.axis('off')
    # plot colorbar
    ax2 = plt.subplot2grid((8, 2), (7, 0))
    plot_colorbar2(edge_cmap_range, cmap_edge, ax2, ori='horizontal')
    ax2.title.set_text('edges color-code')
    return con_lbl_sorted, labels_s