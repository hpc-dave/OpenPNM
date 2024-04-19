import numpy as np
import scipy.spatial as sptl
from openpnm import pnmlib


def delaunay(
    points,
    shape=[1, 1, 1],
    reflect=False,
    f=1,
    trim=True,
    node_prefix='node',
    edge_prefix='edge',
):
    r"""
    Generate a network based on Delaunay triangulation of random points

    Parameters
    ----------
    points : array_like or int
        Can either be an N-by-3 array of point coordinates which will be used,
        or a scalar value indicating the number of points to generate
    shape : array_like
        Indicates the size and shape of the domain
    reflect : boolean, optional (default = ``False``)
        If ``True`` then points are reflected across each face of the domain
        prior to performing the tessellation. These reflected points are
        automatically trimmed.  Enabling this behavior prevents long-range
        connections between surface pores.
    f : float
        The fraction of points which should be reflected.  The default is 1 which
        reflects all the points in the domain, but this can lead to a lot of
        unnecessary points, so setting to 0.1 or 0.2 helps speed, but risks that
        the tessellation may not have smooth faces if not enough points are
        reflected.
    trim : boolean, optional (default = ``True``)
        If ``True`` then any points laying outside the domain are removed. This is
        mostly only useful if ``reflect=True``.

    Returns
    -------
    network : dict
        A dictionary containing 'node.coords' and 'edge.conns'
    tri : Delaunay tessellation object
        The Delaunay tessellation object produced by ``scipy.spatial.Delaunay``
    """
    points = pnmlib.generators.tools.parse_points(
        points=points, shape=shape, reflect=reflect, f=f)
    mask = ~np.all(points == 0, axis=0)
    tri = sptl.Delaunay(points=points[:, mask])
    coo = pnmlib.tools.tri_to_am(tri)
    d = {}
    d[node_prefix+'.coords'] = points
    d[edge_prefix+'.conns'] = np.vstack((coo.row, coo.col)).T
    if trim:
        trim = pnmlib.tools.isoutside(d, shape=shape)
        d = pnmlib.operations.trim_nodes(network=d, inds=np.where(trim)[0])
    return d, tri
