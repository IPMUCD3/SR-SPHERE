from scipy import sparse
from scipy.sparse import coo_matrix
import healpy as hp
import numpy as np
import torch

def nside2index(nside, order):
    nsample = 12 * order**2
    indexes = np.arange(hp.nside2npix(nside) // nsample)
    return indexes

def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    npix = len(indexes)  # Number of pixels.
    indexes = list(indexes)

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose().astype(dtype)

    # Get the 7-8 neighbors.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
    col_index = neighbors.T.reshape((npix * 8))
    row_index = np.repeat(indexes, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere).
    keep = np.isin(col_index, indexes)
    inverse_map = np.full(nside**2 * 12, np.nan)
    inverse_map[indexes] = np.arange(npix)
    col_index = inverse_map[col_index[keep]].astype(int)
    row_index = inverse_map[row_index[keep]].astype(int)

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    kernel_width = distances.mean()
    weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances
    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)

    return W

def build_laplacian(W, lap_type='normalized', dtype=np.float32):
    """Build a Laplacian (tensorflow)."""
    d = np.ravel(W.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        return (D - W).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
    else:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))

def scipy_csr_to_sparse_tensor(csr_mat):
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
    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian

def get_partial_laplacians(nside, depth, order, laplacian_type):
    laps = []
    for i in range(depth):
        subdivisions = int(nside/2**i)
        indexes = nside2index(subdivisions, order)
        L = build_laplacian(healpix_weightmatrix(nside=subdivisions, indexes=indexes), laplacian_type)
        laps.append(prepare_laplacian(L))
    return laps[::-1]

def patch_index_wneighbor(nside, order):
    indexes = list(nside2index(nside, order))
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=True)
    neighbors = np.unique(neighbors.reshape(-1))
    nei_patch = np.setdiff1d(neighbors, indexes)
    return nei_patch

def get_partial_laplacians_pad(nside, depth, order, laplacian_type):
    laps = []
    for i in range(depth):
        subdivisions = int(nside/2**i)
        indexes = nside2index(subdivisions, order)
        nei_patch = patch_index_wneighbor(subdivisions, order)
        indexes = np.concatenate([indexes, nei_patch])
        L = build_laplacian(healpix_weightmatrix(nside=subdivisions, indexes=indexes), laplacian_type)
        laps.append(prepare_laplacian(L))
    return laps[::-1]