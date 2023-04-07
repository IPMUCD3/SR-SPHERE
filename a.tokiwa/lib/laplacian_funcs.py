"""Functions related to getting the laplacian and the right number of pixels after pooling/unpooling.
"""

import numpy as np
import torch
#from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from spherehealpix import SphereHealpix
from scipy import sparse
from scipy.sparse import csr_matrix,coo_matrix
from maploader import hp_split
import healpy as hp

def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.
    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.
    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor

def estimate_lmax(laplacian, tol=5e-3):
    #Estimate the largest eigenvalue of an operator.
    lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
    lmax = lmax[0]
    lmax *= 1 + 2 * tol  # Be robust to errors.
    return lmax

def scale_operator(L, lmax, scale=1):
    #Scale the eigenvalues from [0, lmax] to [-scale, scale].
    I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= I
    return L

def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """
    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian

def healpix_resolution_calculator(nodes):
    """Calculate the resolution of a healpix graph
    for a given number of nodes.
    Args:
        nodes (int): number of nodes in healpix sampling
    Returns:
        int: resolution for the matching healpix graph
    """
    resolution = int(np.sqrt(nodes / 12))
    return resolution

def get_healpix_laplacians(nodes, depth, laplacian_type):
    """Get the healpix laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    for i in range(depth):
        pixel_num = nodes
        subdivisions = int(healpix_resolution_calculator(pixel_num)/2**i)
        G = SphereHealpix(subdivisions, nest=True, k=20)
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]

def patch_laplacian(laps, nodes, depth, split_npix):
    # (list, int, int, int) -> list:shape(depth, split_npix)
    patch_laps = []
    for i in range(depth):
        tmp_nside = int(hp.npix2nside(nodes)/2**i)
        mat_order = hp_split(hp.nside2npix(tmp_nside), split_npix).mat_order
        tmp_lap = laps[::-1][i].to_dense()
        laps_current_depth = []
        for j in range(len(mat_order)):
            tmp_lap_current_depth = tmp_lap[:, mat_order[j]][mat_order[j], :]
            laps_current_depth.append(scipy_csr_to_sparse_tensor(csr_matrix(tmp_lap_current_depth)))
        patch_laps.append(laps_current_depth)
    return patch_laps[::-1]
