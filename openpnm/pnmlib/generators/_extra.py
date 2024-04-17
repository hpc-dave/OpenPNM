import numpy as np
from numpy.linalg import norm
import scipy.spatial as sptl
from openpnm.pnmlib.generators import delaunay as _delaunay
from openpnm.pnmlib.tools import conns_to_am


__all__ = [
    'relative_neighborhood',
    'urquhart',
    'cubic2',
]


def relative_neighborhood(
    points=None, 
    delaunay=None, 
    shape=None, 
    reflect=False,
    node_prefix='node', 
    edge_prefix='edge',
):
    r"""
    Generate a network based on a relative neighborhood, which is a subset of
    the Delaunay triangulation

    Parameters
    ----------
    points : array_like or int, optional
        Can either be an N-by-3 array of point coordinates which will be used,
        or a scalar value indicating the number of points to generate.
        This can be omitted if ``delaunay`` is provided.
    delaunay : network dictionary, optional
        A dictionary containing coords and conns as produced by the
        ``delaunay`` function.  If ``points`` are provided this is
        ignored.
    shape : array_like
        Indicates the size and shape of the domain
    reflect : boolean, optional (default = ``False``)
        If ``True`` then points are reflected across each face of the domain
        prior to performing the tessellation. These reflected points are
        automatically trimmed.  Enabling this behavior prevents long-range
        connections between surface pores.

    Returns
    -------
    network : dict
        A dictionary containing 'node.coords' and 'edge.conns'

    """
    if points is not None:
        dn, tri = _delaunay(points=points, shape=shape, reflect=reflect,
                            node_prefix=node_prefix, edge_prefix=edge_prefix)
    else:
        dn = delaunay
    crds = dn[node_prefix+'.coords']
    cn = dn[edge_prefix+'.conns']
    C1 = crds[cn[:, 0]]
    C2 = crds[cn[:, 1]]
    L = norm(C1 - C2, axis=1)
    tree = sptl.KDTree(crds)
    keep = np.ones(cn.shape[0], dtype=bool)
    for i, pair in enumerate(cn):
        hits = tree.query_ball_point(crds[pair], r=L[i])
        if len(set(hits[0]).intersection(set(hits[1])).difference(set(pair))) > 0:
            keep[i] = False
    # Reduce the connectivity to all True values found in keep
    cn = cn[keep]

    d = {}
    d.update(dn)
    d[edge_prefix+'.conns'] = cn
    d[node_prefix+'.coords'] = crds
    return d


def urquhart(
    points=None,
    delaunay=None,
    shape=None,
    reflect=False,
    node_prefix='node',
    edge_prefix='edge',
):
    r"""
    Generate a network based on a relative neighborhood, which is a subset of
    the Delaunay triangulation

    Parameters
    ----------
    points : array_like or int, optional
        Can either be an N-by-3 array of point coordinates which will be used,
        or a scalar value indicating the number of points to generate.
        This can be omitted if ``delaunay`` is provided.
    delaunay : network dictionary, optional
        A dictionary containing coords and conns as produced by the
        ``delaunay`` function.  If ``points`` are provided this is
        ignored.
    shape : array_like
        Indicates the size and shape of the domain
    reflect : boolean, optional (default = ``False``)
        If ``True`` then points are reflected across each face of the domain
        prior to performing the tessellation. These reflected points are
        automatically trimmed.  Enabling this behavior prevents long-range
        connections between surface pores.

    Returns
    -------
    network : dict
        A dictionary containing 'node.coords' and 'edge.conns'

    """
    if points is not None:
        dn, tri = _delaunay(points=points, shape=shape, reflect=reflect,
                            node_prefix=node_prefix, edge_prefix=edge_prefix)
    else:
        dn = delaunay

    # Find the length of each connection
    crds = dn[node_prefix+'.coords']
    cn = dn[edge_prefix+'.conns']
    C1 = crds[cn[:, 0]]
    C2 = crds[cn[:, 1]]
    L = norm(C1 - C2, axis=1)

    # Find longest connection in each simplex
    sim = tri.simplices
    sim = np.pad(sim, [1, 0], mode='wrap')
    L = []
    for i in range(sim.shape[1]-1):
        L.append([norm(crds[sim[:, i]] - crds[sim[:, i+1]], axis=1)])
    L = np.vstack(L).T

    # Decide which throat to delete
    hits = np.where(L == np.atleast_2d(L.max(axis=1)).T)[1]
    inds = [slice(h, h + 2, None) for h in hits]
    trim = [tuple(sorted(sim[i, inds[i]])) for i in range(len(inds))]
    am = conns_to_am(cn)
    am.data = np.arange(am.nnz)
    am = am.todok()
    throats = np.unique([am[pair] for pair in trim])

    # Build mask of throats to keep
    mask = np.ones(cn.shape[0], dtype=bool)
    mask[throats] = False
    cn = cn[mask]

    d = {}
    d.update(dn)
    # Reduce the connectivity to all True values found in g
    d[edge_prefix+'.conns'] = cn
    d[node_prefix+'.coords'] = crds
    return d


def cubic2(
    shape,
    spacing=1.0,
    offset=0.5,
    faces=True,
    corners=False,
    edges=False,
    order='F',
):
    r"""
    Generate coordinates and connections for a cubic lattice

    Parameters
    ----------
    shape : array-like
        Number of sites along each dimension. A value of 0 or 1 are both considered
        to be a non-existent axis.
    spacing : scalar or array-like
        The spacing between sites in each direction. If a scalar value is given,
        it is applied to all directions.
    offset : scalar of array-like
        The offset to apply to the coordinates in each direction. If a scalar is
        given it is applied to all directions.
    faces, corners, edges : bool
        Flags to indicate which connections to create. If `faces=True` then
        connections are made between sites if their unit cells share a face, and
        so on.  `faces=True` therefore creates 6 connections, `corners=True` adds 8
        connections, and `edges=True` adds 12 connections. Each can be turned off
        or on independently. Note that using only `faces=True` results in 2 separate
        networks (in 3D) and also that `corners` and `edges` are treated the same
        in 2D.
    order : str
        Specifies the ordering of the coordinates to be in either `"C"` style where
        the last index is incremented first, or `"F"` style where the first index is
        incremented first.  The default is `"C"` since this is what Numpy uses.

    Returns
    -------
    coords, conns : ndarrays
        The ndarrays containing the [X, Y, Z] coordinates of each site, and the
        [source, target] connections between sites.

    """
    # Parse inputs
    if np.array(spacing).size == 2:
        spacing = np.array([spacing[0], spacing[1], 1.0])
    if np.array(offset).size == 2:
        offset = np.array([offset[0], offset[1], 0.0])
    if np.array(shape).size == 2:
        shape = np.array([shape[0], shape[1], 1])
    shape = np.clip(shape, 1, None).astype(int)
    if np.any(shape == 1) and corners:
        edges = True  # corners means edges in 2D
        corners = False
    # Create coordinates
    Nx, Ny, Nz = shape
    Np = np.prod((Nx, Ny, Nz))
    ind = np.arange(Np)
    if order == 'F':
        X = ind % Nx
        Y = (ind // Nx) % Ny
        Z = (ind // (Nx * Ny)) % Nz
    elif order == 'C':
        Z = ind % Nz
        Y = (ind // Nz) % Ny
        X = (ind // (Nz * Ny)) % Nx
    else:
        raise Exception("Order must be 'C' or 'F'")
    coords = np.vstack((X, Y, Z)).T
    coords = coords*spacing + offset
    # Now create connections
    idx = ind.reshape(shape, order=order)
    source = []
    target = []
    if faces:
        source.extend(idx[:-1, :, :].flatten(order))
        target.extend(idx[1:, :, :].flatten(order))
        source.extend(idx[:, :-1, :].flatten(order))
        target.extend(idx[:, 1:, :].flatten(order))
        source.extend(idx[:, :, :-1].flatten(order))
        target.extend(idx[:, :, 1:].flatten(order))
    if edges:
        source.extend(idx[:-1, :-1, :].flatten(order))
        target.extend(idx[1:, 1:, :].flatten(order))
        source.extend(idx[1:, :-1, :].flatten(order))
        target.extend(idx[:-1, 1:, :].flatten(order))
        source.extend(idx[:-1, :, :-1].flatten(order))
        target.extend(idx[1:, :, 1:].flatten(order))
        source.extend(idx[1:, :, :-1].flatten(order))
        target.extend(idx[:-1, :, 1:].flatten(order))
        source.extend(idx[:, :-1, :-1].flatten(order))
        target.extend(idx[:, 1:, 1:].flatten(order))
        source.extend(idx[:, :-1, 1:].flatten(order))
        target.extend(idx[:, 1:, :-1].flatten(order))
    if corners:
        source.extend(idx[:-1, :-1, :-1].flatten(order))
        target.extend(idx[1:, 1:, 1:].flatten(order))
        source.extend(idx[1:, :-1, :-1].flatten(order))
        target.extend(idx[:-1, 1:, 1:].flatten(order))
        source.extend(idx[1:, 1:, :-1].flatten(order))
        target.extend(idx[:-1, :-1, 1:].flatten(order))
        source.extend(idx[:-1, 1:, :-1].flatten(order))
        target.extend(idx[1:, :-1, 1:].flatten(order))
    conns = np.vstack((source, target)).T  # Convert to numpy array
    conns = np.sort(conns, axis=1)  # Make upper-triangular (r , c) where r < c
    return coords, conns
