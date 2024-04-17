import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from numba import njit
import matplotlib.patheffects as pe


__all__ = [
    "union_to_conns",
    "plot_union",
    "unionify",
    "root",
    "resolve",
    "path",
    "label_clusters",
    "union_to_labels",
    "quick_find",
    "quick_union",
    "plot_labels",
    "hk",
]


def hk(arr):
    import openpnm as op
    ws = op.Workspace()
    pn = op.network.CubicTemplate(np.ones_like(arr))
    pn['pore.active'] = arr.flatten()
    pn['pore.label'] = 0
    lil = pn.create_adjacency_matrix(fmt='lil', triu=False)
    label = 1
    for p in range(pn.Np):
        if pn['pore.active'][p]:
            neighbors = np.array(lil.rows[p])
            neighbors = neighbors[neighbors < p]
            neighbors = np.array(neighbors)[pn['pore.active'][neighbors]]
            if neighbors.size == 0:  # Begin labeling a new cluster
                pn['pore.label'][p] = label
                label += 1
            elif neighbors.size == 1:
                pn['pore.label'][p] = pn['pore.label'][neighbors[0]]
            else:
                q = pn['pore.label'][neighbors].min()
                pn['pore.label'][p] = q
                for i in pn['pore.label'][neighbors]:
                    ind = np.where(pn['pore.label'][:p] == i)
                    pn['pore.label'][ind] = q
    a = pn['pore.label'].reshape(arr.shape)
    del ws[pn.project.name]
    return a


def update_weights(ind):
    bins = np.arange(ind[0, :].size + 1)
    weights, _ = np.histogram(ind[0, :], bins=bins)
    ind[1, :] = weights
    return ind


def plot_labels(ind):
    if ind.ndim == 2:
        ind = update_weights(ind)
        labels, weights = ind[0, :], ind[1, :]
    else:
        labels = np.copy(ind)
        weights = np.ones_like(labels)*np.nan

    fig, ax = plt.subplots()
    ax.pcolormesh(np.vstack((weights, labels)), edgecolor='w',
                  cmap=plt.cm.turbo, vmin=-1, vmax=labels.size-1)

    strokekws = dict(linewidth=1, foreground='k', alpha=0.5)
    s = 16  # Font size
    ax.text(-0.1, 2.5, "node id", color='k', va='top', ha='right', size=s)
    ax.text(-0.1, 1.5, "label", color='k', va='center', ha='right', size=s)
    if not np.any(np.isnan(weights)):
        ax.text(-0.1, 0.5, "weight",  color='k', va='center', ha='right', size=s)

    for i in range(labels.size):
        ax.text(i+0.5, 2.5, i, color='k', va='top', ha='center', size=s)
        ax.text(i+0.5, 1.5, labels[i], color='w', va='center', ha='center',
                size=s, path_effects=[pe.withStroke(**strokekws)])
        if not np.any(np.isnan(weights)):
            ax.text(i+0.5, 0.5, weights[i], color='w', va='center', ha='center',
                    size=s, path_effects=[pe.withStroke(**strokekws)])

    ax.axis('scaled')
    ax.set_axis_off()

    return ax


def union_to_conns(ind):
    r"""
    Generates the adjacency matrix from connections in the union in the COO format

    Parameters
    ----------
    ind : ndarray
        The array of current node labels
    """
    conns = []
    for i, val in enumerate(ind[0, ...]):
        if val != i:
            conns.append([val, i])
    conns = np.vstack(conns)
    return conns


def plot_union(ind, color_by=None, size_by=None, ax=None, **kwargs):
    r"""
    Plots the label array as a tree to visualize clusters.

    Parameters
    ----------
    ind : ndarray
        The array of current node labels
    """
    conns = union_to_conns(ind)
    g = nx.DiGraph()
    g.add_nodes_from(np.arange(len(ind[0, ...])))
    g.add_edges_from(conns)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    if color_by is None:
        c = np.arange(len(ind[0, ...]))
    else:
        c = color_by
    if size_by is None:
        s = 1000
    else:
        s = size_by
    options = {
        "font_size": 16,
        "font_color": "k",
        "node_size": s,
        "node_color": c,
        "edgecolors": "black",
        "linewidths": 2,
        "width": 2,
        "arrows": False,
        "cmap": plt.cm.turbo,
        "vmin": -len(ind[0, ...])/10,
        "vmax": max(c) + 0.5
    }
    if ax is None:
        _, ax = plt.subplots()
    options.update(**kwargs)
    nx.draw(g, pos, with_labels=True, ax=ax, **options)
    return ax


def unionify(N):
    r"""
    Generate a 2-row list of index numbers for each node and weights

    Parameters
    ----------
    N : int
        The number of elements in the set

    Returns
    -------
    inds : ndarray
        A numpy array of values ranging from 0 to N in the first row, and weights of
        1 for all elements in the second row.
    """
    ind = np.arange(N)
    sz = np.ones(N, dtype=int)
    ind = np.vstack((ind, sz))
    return ind


@njit
def root(ind, i, compress=False):
    r"""
    Given the current list of node labels, finds the root value for node `i`

    Parameters
    ----------
    inds : ndarray
        The array of current node labels
    i : int
        The index of the node whose actual root is sought
    compress : boolean
        If `True` this will take the opportunity to compress the tree as it
        performs the search for the root of `i`.

    Returns
    -------
    root : int
        The index of the root node of node `i`.
    """
    while i != ind[0, i]:
        if compress:
            ind[0, i] = ind[0, ind[0, i]]
        i = ind[0, i]
    return i


@njit
def resolve(ind):
    r"""
    Scans all nodes and resolves the final root node for each

    Parameters
    ----------
    ind : ndarray
        The array of current node labels

    Returns
    -------
    ind : ndarray
        An array containing the node labels all updated to their final root value
    """
    i = 0
    while i < len(ind[0, :]):
        j = root(ind, i)
        if j > i:
            ind[0, i] = i
            ind[0, j] = i
        else:
            ind[0, i] = j
        i += 1
    return ind


def union_to_labels(ind):
    r"""
    Labels each node according to which cluster is belongs

    Parameters
    ----------
    ind : ndarray
        The array of current node labels

    Returns
    -------
    labels : ndarray
        The cluster label to which each node belongs, starting from 0 and following
        continously increasing values. Note that in this form the array is no longer
        a union data set and will provide erroneous results if used as such.
    """
    ind = resolve(ind)
    labels = rankdata(ind[0, :], method='dense') - 1
    return labels


def label_clusters(conns, active):
    r"""
    Labels connected clusters in a network using the weighted quick union with
    path compression algorithm

    Parameters
    ----------
    conns : ndarray
        The COO sparse representation of the networks adjacency matrix
    active : ndarray
        A boolean array the same length as `conns` with `True` values indicating
        that a bond is open or active, meaning that the two sites connected by
        that bond are part of the same cluster.

    Returns
    -------
    labels : ndarray
        An array containing the label number of each site in the network.

    Notes
    -----
    Sites not appearing in the `conns` list (i.e., because they are isolated)
    will get a label *unless* their index number is larger than the largest
    value in `conns`. In other words, if sites 10 and 100 are both isolated,
    and the largest value in `conns` is 99, the site 10 will be given a label,
    but site 100 will not. The returned array will only be 100 elements long.

    """
    N = np.amax(conns)+1
    ind = unionify(N)
    ind = _apply_union_to_conns(ind, conns, active)
    labels = union_to_labels(ind)
    return labels


@njit
def _apply_union_to_conns(ind, conns, active):
    r"""
    JIT compiled function for scanning through the adjacency matrix in COO format
    to label connected clusters
    """
    for i in range(conns.shape[0]):
        if active[i]:
            ind = quick_union(ind, conns[i, 0], conns[i, 1],
                              compress=True, weighted=True)
    return ind


@njit
def path(ind, i):
    r"""
    Computes the path of nodes between given pore and its root

    Parameters
    ----------
    ind : ndarray
        The array of current node labels
    i : int
        The index of the node whose path is sought

    Returns
    path : list
        A list of node numbers, starting from `i` and ending at the root node.
    """
    path = [i]
    while True:
        if ind[0, i] != i:
            i = ind[0, i]
            path.append(i)
        else:
            break
    return path


@njit
def quick_find(ind, p, q):
    r"""
    Performs a union between `p` and `q` using the Quick-Find algorithm

    The Quick-Find algorithm uses eager relabeling of nodes so all nodes on the same
    cluster have the same label.

    Parameters
    ----------
    ind : ndarray
        The array of current node labels
    p, q : int
        The index of the two nodes which are to be joined

    Returns
    -------
    ind : ndarray
        The array of cluster labels updated to indicate the joining of nodes `p` and
        `q`.

    Notes
    -----
    The Quick-Find algorithm is unreasonably slow. It is only included here
    for comparison, completeness, and academic interest.
    """
    if ind.ndim == 1:
        for k in range(len(ind[:])):
            if ind[k] == ind[p]:
                hits = ind == ind[k]
                ind[hits] = ind[q]
    if ind.ndim == 2:
        weights = np.zeros_like(ind[1, :])
        for k in range(len(ind[0, :])):
            if ind[0, k] == ind[0, p]:
                hits = ind[0, :] == ind[0, k]
                ind[0, hits] = ind[0, q]
            weights[ind[0, k]] += 1
        ind[1, :] = weights
    return ind


@njit
def quick_union(ind, p, q, compress=True, weighted=True):
    r"""
    Performs a union between `p` and `q` using the Quick-Union algorithm

    Parameters
    ----------
    ind : ndarray
        The array of current node labels
    p, q : int
        The index of the two nodes which are to be joined.
    compress : boolean
        This is passed on to the `root` function to indicate if paths should
        be compressed while searching for root nodes. This keeps the tree
        shallow and enhances performance.
    weighted : boolean
        If `True` then `q` is attached to `p` if `q` is part of a smaller tree,
        otherwise `p` is attached to `q`.

    Returns
    -------
    ind : ndarray
        The array of updated node labels after connected `p` and `q`.
    """
    i = root(ind, p, compress)
    j = root(ind, q, compress)
    if weighted:
        if ind[1, i] < ind[1, j]:
            ind[0, i] = j
            ind[1, j] += ind[1, i]
        else:
            ind[0, j] = i
            if i != j:  # Only update the weights if the nodes do not share a root
                ind[1, i] += ind[1, j]
    else:
        ind[0, i] = j
    return ind
