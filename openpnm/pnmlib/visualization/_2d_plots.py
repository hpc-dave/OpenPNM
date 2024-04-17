import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from openpnm.visualization import plot_coordinates, plot_connections


__all__ = [
    'plot_patches',
    'draw_pores',
    'draw_throats',
    'annotate_pores',
    'annotate_throats',
    'label_pores',
    'label_throats',
    'annotate_heatmap',
]


def get_end_points_as_coords(network):
    from openpnm.models.geometry.throat_endpoints import spheres_and_cylinders
    end_points = spheres_and_cylinders(network=network)
    s = (2*network.Nt, 3)
    new_coords = np.hstack(list(end_points.values())).reshape(s)
    # new_coords = np.mean(new_coords, axis=1)
    # op.topotools.extend(network=network, pore_coords=new_coords, labels=label)
    return new_coords


def plot_patches(
    network,
    cmap=None,
    color_by='diameter',
    show_ends=True,
    show_centers=True,
    end_points=None,
):
    r"""
    Plot a 2D pore-throat diagram using circular and rectangular patches

    Parameters
    ----------
    network : dict
        The OpenPNM network object
    cmap : matplotlib.pyplot cmap object or str
        The color map to use. Can be an `cmap` object or a `str` of the cmap name.
        If `None` (the default) then the pores will be red and throats will be
        blue, and the `prop` argument will be ignored.
    prop : str
        The network property to use for coloring the patches. This property must
        be present on both pores and throats.  The default is 'diameter'.
    show_ends : bool
        If `True` (default), white markers are added at the end points of each throat
    show_centers : bool
        If `True` (default), black markers are added at the center of each pore

    Returns
    -------
    fig, ax
        The matplotlib figure and axis objects

    Notes
    -----
    All the z-coordinates must be equal, but they do not have to be 0.
    """
    from matplotlib.patches import Rectangle, Circle
    from openpnm.models.geometry.throat_endpoints import spheres_and_cylinders
    from openpnm.models.geometry.throat_vector import pore_to_pore

    # Ensure network is 2D
    assert np.all(network.coords[0, :] == network.coords, axis=0)[2] == True, \
        "Network must be 2D for this function to work"

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = getattr(plt.cm, cmap)
        cmin = min(network[f'pore.{color_by}'].min(), network[f'throat.{color_by}'].min())
        cmax = max(network[f'pore.{color_by}'].max(), network[f'throat.{color_by}'].max())
        pore_color = cmap((network[f'pore.{color_by}'] - cmin)/(cmax - cmin))
        throat_color = cmap((network[f'throat.{color_by}'] - cmin)/(cmax - cmin))
    else:
        pore_color = ['tab:red']*network.Np
        throat_color = ['tab:blue']*network.Nt

    pores = []
    for p in network.Ps:
        pores.append(
            Circle(
                xy=network.coords[p, :2],
                radius=network['pore.diameter'][p]/2,
                alpha=0.8,
                facecolor=pore_color[p],
                edgecolor='k',
                linewidth=3,
            ),
        )

    if end_points is None:
        end_points = spheres_and_cylinders(network=network)
    else:
        end_points = network[end_points]
    s = (network.Nt, 2, 3)
    new_coords = np.hstack(list(end_points.values())).reshape(s)
    vecs = pore_to_pore(network)
    q = np.rad2deg(np.arctan(-vecs[:, 0]/vecs[:, 1]))
    throats = []
    for t in network.Ts:
        d = network['throat.diameter'][t]
        xy = new_coords[t, :, :2]
        L = np.sqrt(np.sum((xy[1]-xy[0])**2))
        L = L if xy[0][1] <= xy[1][1] else -L
        offset = -d/2*np.array((np.sin(np.deg2rad(90-q[t])),
                                np.cos(np.deg2rad(90-q[t]))))
        throats.append(
            Rectangle(
                xy=xy[0]+offset,
                width=d,
                height=L,
                angle=q[t],
                alpha=0.6,
                facecolor=throat_color[t],
                edgecolor='k',
                linewidth=3,
            ),
        )
    a = 1.0 if show_centers else 0.0
    _ = plot_coordinates(network, c='k', alpha=a, s=100, zorder=4)
    fig, ax = plt.gcf(), plt.gca()
    for item in pores + throats:
        ax.add_patch(item)
    if show_ends:
        end_coords = get_end_points_as_coords(network)
        ax.plot(end_coords[:, 0], end_coords[:, 1], 'w.', zorder=3)
    return fig, ax


def annotate_pores(
    network,
    pores=None,
    font_kws={},
    arrow_kws={},
    scale=1.0,
    ax=None,
):
    r"""
    """
    font_defaults = {
        'size': 18,
        'color': 'k',
        'ha': 'center',
    }
    arrow_defaults = {
        'overhang': 0.1,
        'edgecolor': 'k',
        'facecolor': 'k',
    }
    font_kws = font_defaults | font_kws
    arrow_kws = arrow_defaults | arrow_kws
    if pores is None:
        pores = network.Ps
    elif pores.dtype == bool:
        pores = np.where(pores)[0]
    if ax is None:
        _, ax = plt.subplots()
    coords = network['pore.coords']
    R = network['pore.diameter']/2
    for i in pores:
        ax.arrow(
            x=coords[i][0],
            y=coords[i][1],
            dx=0,
            dy=R[i],
            width=0.002*scale,
            length_includes_head=True,
            head_width=R[i]*0.05*scale,
            zorder=5,
            **arrow_kws,
        )
        ax.text(
            x=coords[i][0],
            y=coords[i][1] + 1.05*R[i],
            s=str(np.around(R[i], decimals=1)),
            **font_kws,
        )
    return ax


def annotate_throats(
    network,
    throats=None,
    end_points=None,
    scale=1.0,
    ax=None,
    arrow_kws={},
    font_kws={},
):
    r"""
    """
    font_defaults = {
        'size': 18,
        'color': 'k',
    }
    arrow_defaults = {
        'overhang': 0.1,
        'edgecolor': 'k',
        'facecolor': 'k',
    }
    font_kws = font_defaults | font_kws
    arrow_kws = arrow_defaults | arrow_kws
    if ax is None:
        _, ax = plt.subplots()
    if throats is None:
        throats = network.Ts
    elif throats.dtype == bool:
        throats = np.where(throats)[0]
    if end_points is None:
        from openpnm.models.geometry.throat_endpoints import spheres_and_cylinders
        end_points = spheres_and_cylinders(network=network)
    else:
        end_points = network[end_points]

    R = network['throat.diameter']/2
    for i in throats:
        coords = (end_points['head'][i] + end_points['tail'][i])/2
        ax.arrow(
            x=coords[0],
            y=coords[1],
            dx=0,
            dy=R[i],
            width=0.002*scale,
            length_includes_head=True,
            head_width=R[i]*0.05*scale,
            zorder=5,
            **arrow_kws,
        )
        ax.text(
            x=coords[0],
            y=coords[1] + 1.05*R[i],
            s=str(np.around(R[i], decimals=1)),
            horizontalalignment='center',
            **font_kws,
        )
    return ax


def label_throats(
    network,
    throats=None,
    label_by=None,
    end_points=None,
    ax=None,
    font_kwargs={},
):
    r"""
    """
    font_defaults = {
        'color': 'k',
        'size': 14,
        'ma': 'center',
        'ha': 'center',
        'va': 'center',
        'bbox': dict(
            boxstyle='round',
            fc="w",
            ec='none',
            alpha=0.3,
        ),
    }
    font_kws = font_defaults | font_kwargs

    if ax is None:
        _, ax = plt.subplots()

    if throats is None:
        throats = network.Ts

    if label_by is None:
        label_by = network.Ts
    elif label_by.size < network.Nt:
        temp = np.zeros(network.Nt)
        temp[throats] = label_by
        label_by = np.copy(temp)

    if end_points is None:
        from openpnm.models.geometry.throat_endpoints import spheres_and_cylinders
        end_points = spheres_and_cylinders(network=network)
    else:
        end_points = network[end_points]

    for t in throats:
        coords = (end_points['head'][t] + end_points['tail'][t])/2
        ax.text(
            x=coords[0],
            y=coords[1],
            s=str(label_by[t]),
            **font_kws,
        )
    return ax


def label_pores(
    network,
    pores=None,
    label_by=None,
    ax=None,
    font_kwargs={},
    stroke_kwargs={},
):
    r"""
    """
    font_defaults = {
        'color': 'k',
        'size': 14,
        'ma': 'center',
        'ha': 'center',
        'va': 'center',
        'bbox': dict(
            boxstyle='round',
            fc="w",
            ec='none',
            alpha=0.3,
        ),
    }
    font_kws = font_defaults | font_kwargs
    c = 'w' if font_kws['color'] == 'k' else 'k'
    stroke_defaults = dict(linewidth=1, foreground=c, alpha=0.5)
    stroke_kws = stroke_defaults | stroke_kwargs

    if ax is None:
        _, ax = plt.subplots()
    if pores is None:
        pores = network.Ps
    if label_by is None:
        label_by = network.Ps
    elif len(label_by) < network.Np:
        temp = np.zeros(network.Np)
        temp[pores] = label_by
        label_by = np.copy(temp)
    coords = network['pore.coords']
    for p in pores:
        ax.text(
            x=coords[p][0],
            y=coords[p][1],
            s=str(label_by[p]),
            path_effects=[pe.withStroke(**stroke_kws)],
            **font_kws,
        )
    return ax


def draw_pores(
    network,
    pores=None,
    style='circles',
    color_by=None,
    crange=None,
    show_centers=0,
    cmap='turbo',
    facecolor='tab:red',
    scale=1.0,
    patch_kwargs={},
    ax=None,
):
    r"""
    Plot a 2D diagram of pores using "patches"

    Parameters
    ----------
    network : dict
        The OpenPNM network object
    pores : array_like
        A list of which pores to draw. Can be numerical indices or a boolean mask.
        The default is to plot all pores.
    style : str
        The shape of the pores. Options are `'circles'` and `'squares'`, and the
        default is `'circles'`.
    color_by : ndarray
        The network property to use for coloring the patches. If not given then
        all pores are given the same color, specified by the 'facecolor' argument.
    crange : tuple
        The (lo, hi) values for the color map.  If not provided it will use
        the min and max of the values in `color_by`.
    show_centers : bool
        If `True` (default), black markers are added at the center of each pore
    cmap : str or matplotlib colormap object
        The default is `'turbo'`. This is only used if the `color_by` argument is
        specified.
    facecolor : str
        The color to apply to each pore if `color_by` is not given.  The default
        is `'tab:red'`.
    ax : Matplotlib axis handle
        The matplotlib axis on which to add the pore patches. If not given a fresh
        axis is created.
    patch_kwargs : dict
        A dictionary of arguments to pass to the patch drawing function.  The
        default values are:

        ============== =============================================
        `arg`          `value`
        ============== =============================================
        'alpha'        0.8
        'edgecolor'    'k'
        'linewidth'    3
        ============== =============================================

    Returns
    -------
    ax : Matplotlib axis handle
        The matplotlib axis object.  This can be passed to `draw_throats` to
        add them to the plot

    Notes
    -----
    All the z-coordinates must be equal, but they do not have to be 0.
    """
    from matplotlib.patches import Circle, Rectangle
    patch_defaults = {
        'alpha': 0.8,
        'edgecolor': 'k',
        'linewidth': 3,
    }
    patchkws = patch_defaults | patch_kwargs
    # Ensure network is 2D
    assert np.all(network.coords[0, :] == network.coords, axis=0)[2] == True, \
        "Network must be 2D for this function to work"

    if pores is None:
        pores = network.Ps

    if color_by is not None:
        if color_by.size == network.Np:
            color_by = color_by[pores]
        if crange is None:
            crange = (color_by[np.isfinite(color_by)].min(),
                      color_by[np.isfinite(color_by)].max())
        if isinstance(cmap, str):
            cmap = getattr(plt.cm, cmap)
        facecolor = cmap((color_by - min(crange))/(max(crange) - min(crange)))
    else:
        facecolor = [facecolor]*len(pores)

    patches = []
    if style.startswith('circ') or style.startswith('round'):
        for i, p in enumerate(pores):
            patches.append(
                Circle(
                    xy=network.coords[p, :2]*scale,
                    radius=network['pore.diameter'][p]/2*scale,
                    facecolor=facecolor[i],
                    **patchkws
                ),
            )
    elif style.startswith('square') or style.startswith('rect'):
        for i, p in enumerate(pores):
            patches.append(Rectangle(
                xy=(network.coords[p, 0]*scale - network['pore.diameter'][p]/2*scale,
                    network.coords[p, 1]*scale - network['pore.diameter'][p]/2)*scale,
                width=network['pore.diameter'][p]*scale,
                height=network['pore.diameter'][p]*scale,
                facecolor=facecolor[i],
                **patchkws,
            ))

    if ax is None:
        fig, ax = plt.subplots()
    if show_centers:
        ax.plot(
            network.coords[:, 0]*scale,
            network.coords[:, 1]*scale,
            'w.',
            markersize=show_centers*scale,
            zorder=3,
        )
    for item in patches:
        ax.add_patch(item)
    if 0:
        from matplotlib.patches import Shadow
        for item in patches:
            s = Shadow(item, -0.01, -0.01, alpha=1, facecolor='grey', edgecolor=None)
            ax.add_patch(s)
    ax.axis('equal')
    return ax


def draw_throats(
    network,
    throats=None,
    color_by=None,
    crange=None,
    cmap='turbo',
    end_points=None,
    show_ends=0,
    ax=None,
    facecolor='tab:blue',
    patch_kwargs={},
    scale=1.0,
):
    r"""
    Plot a 2D diagram of throats using "patches"

    Parameters
    ----------
    network : dict
        The OpenPNM network object
    throats : array_like
        A list of which throats to draw. Can be numerical indices or a boolean mask.
        The default is to plot all throats.
    color_by : ndarray
        The network property to use for coloring the patches. If not given then
        all throats are given the same color, specified by the 'facecolor' argument.
    crange : tuple
        The (lo, hi) values for the color map.  If not provided it will use
        the min and max of the values in `color_by`.
    end_points : str
        The dictionary key for the throat endpoint coordinates. If not given the
        `spheres_and_cylinders` model is used.
    show_ends : int
        Indicates the size of white markers to add at the endpoints of each throat.
        If 0 then no markers are drawn.
    cmap : str or matplotlib colormap object
        The default is `'turbo'`. This is only used if the `color_by` argument is
        specified.
    facecolor : str
        The color to apply to each throat if `color_by` is not given.  The default
        is `'tab:blue'`.
    ax : Matplotlib axis handle
        The matplotlib axis on which to add the throat patches. If not given a fresh
        axis is created.
    patch_kwargs : dict
        A dictionary of arguments to pass to the patch drawing function.  The
        default values are:

        ============== =============================================
        `arg`          `value`
        ============== =============================================
        'alpha'        0.8
        'edgecolor'    'k'
        'linewidth'    2
        ============== =============================================

    Returns
    -------
    ax : Matplotlib axis handle
        The matplotlib axis object.  This can be passed to `draw_pores` to
        add them to the plot

    Notes
    -----
    All the z-coordinates must be equal, but they do not have to be 0.
    """
    from matplotlib.patches import Rectangle
    from openpnm.models.geometry.throat_vector import pore_to_pore

    patch_defaults = {
        'alpha': 0.8,
        'edgecolor': 'k',
        'linewidth': 2,
    }
    patchkws = patch_defaults | patch_kwargs

    # Ensure network is 2D
    assert np.all(network.coords[0, :] == network.coords, axis=0)[2] == True, \
        "Network must be 2D for this function to work"

    if throats is None:
        throats = network.Ts

    if color_by is not None:
        if color_by.size == network.Nt:
            color_by = color_by[throats]
        if crange is None:
            crange = (color_by[np.isfinite(color_by)].min(),
                      color_by[np.isfinite(color_by)].max())
        if isinstance(cmap, str):
            cmap = getattr(plt.cm, cmap)
        facecolor = cmap((color_by - min(crange))/(max(crange) - min(crange)))
    else:
        facecolor = [facecolor]*len(throats)
    if end_points is None:
        from openpnm.models.geometry.throat_endpoints import spheres_and_cylinders
        end_points = spheres_and_cylinders(network=network)
    else:
        end_points = network[end_points]
    s = (network.Nt, 2, 3)
    new_coords = np.hstack(list(end_points.values())).reshape(s)
    vecs = pore_to_pore(network)
    q = np.rad2deg(np.arctan(-vecs[:, 0]/vecs[:, 1]))
    patches = []
    for i, t in enumerate(throats):
        d = network['throat.diameter'][t]
        xy = new_coords[t, :, :2]
        L = np.sqrt(np.sum((xy[1]-xy[0])**2))
        L = L if xy[0][1] <= xy[1][1] else -L
        offset = -d/2*np.array((np.sin(np.deg2rad(90-q[t])),
                                np.cos(np.deg2rad(90-q[t]))))
        patches.append(
            Rectangle(
                xy=(xy[0]+offset)*scale,
                width=d*scale,
                height=L*scale,
                angle=q[t],
                facecolor=facecolor[i],
                **patchkws,
            ),
        )
    if ax is None:
        fig, ax = plt.subplots()
    for item in patches:
        ax.add_patch(item)
    if 0:
        from matplotlib.patches import Shadow
        for item in patches:
            s = Shadow(item, -0.01, -0.01, alpha=1, facecolor='grey', edgecolor=None)
            ax.add_patch(s)
    if show_ends:
        end_coords = get_end_points_as_coords(network)
        ax.plot(
            end_coords[:, 0]*scale,
            end_coords[:, 1]*scale,
            'w.',
            markersize=show_ends*scale,
            zorder=3,
        )
    ax.axis('equal')
    return ax


def annotate_heatmap(im, label_by=None, label_at=None, ax=None, font_kwargs={}, **kwargs):
    r"""
    Overlays text representation of pixel values within a 2D image

    Parameters
    ----------
    im : ndarray
        A 2D array containing numerical values
    label_by : ndarray
        A 2D array the same shape as `im` with the specific values to write
        at each location. If not given then the values in `im` are used.
    label_at : ndarray
        A 2D array the same shape as `im` with `True` values at the locations
        where labels should be applied. If not given then all locations are used.
    ax : matplotlib axis handle
        The handle of the
    font_kwargs : dict
        The arguments to send to the `text` function to control font properties
    kwargs : keyword arguments
        All other keyword arguments are passed to the `pcolormesh` function to
        control the formatting of the heatmap.
    """
    font_defaults = {
        'size': 10,
        'ha': 'center',
        'va': 'center',
        'color': 'k',
    }
    font_kwargs = font_defaults | font_kwargs
    c = 'w' if font_kwargs['color'] == 'k' else 'k'
    strokekws = dict(linewidth=1, foreground=c, alpha=0.5)

    if label_by is None:
        label_by = np.copy(im)

    if label_at is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.zeros_like(im, dtype=bool)
        mask[label_at] = True

    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(im, **kwargs)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if mask[i, j]:
                ax.text(j+0.5, i+0.5, label_by[i, j],
                        path_effects=[pe.withStroke(**strokekws)],
                        **font_kwargs)
    return ax


if __name__ == "__main__":
    import openpnm as op
    # np.random.seed(0)
    pn = op.network.Cubic([2, 1, 1], spacing=1.5)
    # pn = op.network.DelaunayVoronoiDual(shape=[5, 5, 0], points=100, relaxation=4)
    # op.topotools.trim(pn, pores=pn.pores('voronoi'))
    # Apply geometry models to the network and make some adjustments
    pn.regenerate_models()
    pn['pore.diameter'] = [0.7, 0.99]
    pn['throat.diameter'] = 0.5

    ax = None
    ax = draw_throats(
        network=pn,
        show_ends=0,
        ax=ax,
    )
    ax = draw_pores(
        network=pn,
        style='circles',
        show_centers=90,
        ax=ax,
    )
    ax.axis(False)
    ax = annotate_pores(network=pn, ax=ax, font_kws={'size': 32})
    ax = annotate_throats(network=pn, ax=ax)
    ax = label_pores(network=pn, ax=ax)
    ax = label_throats(network=pn, ax=ax)
