import networkx as nx
import numpy as np
from numba import jit, njit
from typing import Union, List
from tqdm.auto import tqdm
from scipy.sparse.linalg import expm_multiply
import scipy

"""
This is a non-official implementation of 
"Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian",
SOURCE:
******************************************************************************
FURUTANI, Satoshi, et al. 
Graph signal processing for directed graphs based on the Hermitian Laplacian. 
In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. 
Springer, Cham, 2019. p. 447-463.
******************************************************************************
"""

# Parameters used by Furutani et al.
HERMLAP_S = np.arange(2.0, 20.1, 0.1)
HERMLAP_T = np.arange(1, 11, 1)


def get_adj(G: nx.Graph) -> np.matrix:
    """Utility function to extract the adjacency matrix of a networkx graph"""
    return nx.to_numpy_array(G)


@njit
def check_binary(x):
    is_binary = True
    for v in np.nditer(x):
        if v.item() != 0 and v.item() != 1:
            is_binary = False
            break

    return is_binary


@jit
def gamma(A: np.matrix, i, j, q) -> np.float32:
    """
    Input:
    A: weighted or unweighted adjacency matrix of a directed graph
    i: row
    j: column
    q: parameter
    Returns:
    the value of the gamma function
    """
    return np.exp(1j * 2 * np.pi * q * (A[i, j] - A[j, i]))


@jit
def degree_matrix(A: np.matrix) -> np.matrix:
    """Returns a degree matrix of a weighted or unweighted adjacency matrix"""
    return np.diag(np.sum(A, axis=1))


def inverse_sqrt_degree_matrix(A: np.matrix) -> np.matrix:
    """Returns an (pseudo) inverse square root degree matrix of a weighted or unweighted adjacency matrix"""
    return np.linalg.pinv(
        np.diag(np.sqrt(np.sum(A, axis=1)))
    )
    

@jit
def hermitian_laplacian(A: np.matrix, q) -> np.matrix:
    """Returns the Hermitian Laplacian of a graph as defined in the paper"""

    A_s = (A + A.T) / 2
    D = degree_matrix(A_s)
    Gamma = np.zeros(A.shape, dtype=np.complex64)
    for i, j in zip(*A_s.nonzero()):
        Gamma[i, j] = gamma(A, i, j, q)

    L_q = D - np.multiply(Gamma, A_s)
    return L_q

def normalize_laplacian(A:np.matrix, L_q: np.matrix) -> np.matrix:    
    i_D = inverse_sqrt_degree_matrix(A)
    L_q = i_D @ L_q @ i_D
    return L_q


@jit
def low_pass_filter_kernel(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Low-pass filter kernel"""
    return 1 / (1 + c * x)


@jit
def heat_kernel(x: np.ndarray) -> np.ndarray:
    """Heat kernel"""
    return np.exp(-x)


@jit
def phi(psi: np.matrix, t: float) -> np.float32:
    """
    Computes an embedding from the wavelet for a node I
    """
    return 1 / psi.shape[0] * np.sum(
        np.exp(1j * t * psi),
        axis=1
    )


@jit
def determine_proper_S(eigenvalues: np.ndarray, n: int = 5) -> np.ndarray:
    """Determine scale parameter values according to graphwave"""
    nu = 0.85
    gamma = 0.95
    denom = np.sqrt(eigenvalues[1] * eigenvalues[-1])
    if denom == 0: denom = 1
    s_max = -np.log(nu) / denom
    s_min = -np.log(gamma) / denom
    S = np.linspace(s_min, s_max, n)
    return S


def eigen(A):
    eigenValues, eigenVectors = scipy.linalg.eigh(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)


def scipy_expm(L_q, S: List[float]):
    N = L_q.shape[0]
    delta = np.eye(N)
    psi = expm_multiply(-L_q, delta, S[0], S[-1], len(S))
    return psi


def eigen_expm(L_q, S: List[float] = None, kernel: callable = heat_kernel, progress: bool = False, **kernel_args):
    N = L_q.shape[0]
    eigenvalues, U = np.linalg.eigh(L_q)
    eigenvalues = np.array(eigenvalues)
    # U = np.matrix(U)
    delta = np.eye(N)

    if S is None:
        if progress: print("auto-inferring S")
        S = determine_proper_S(np.real(eigenvalues))

    psi = []
    for s_idx, s in enumerate(tqdm(S, disable=not progress)):
        G_hat_s = np.diag(kernel(eigenvalues * s, **kernel_args))
        psi.append(U @ G_hat_s @ U.conjugate().T @ delta)
    return np.stack(psi, axis=0), S


def characteristic_function1(psi, T: List[float] = np.arange(0, 101, 2), progress: bool = False):
    N = psi.shape[1]
    re_embeddings = np.zeros((N, len(psi) * len(T)))
    im_embeddings = np.zeros(re_embeddings.shape)

    for t_idx, t in enumerate(tqdm(T, disable=not progress)):
        phi_i = 1 / N * np.sum(np.exp(1j * t * psi), axis=-1).transpose(1, 0)
        start_idx = t_idx * phi_i.shape[-1]
        end_idx = start_idx + phi_i.shape[-1]
        re_embeddings[:, start_idx:end_idx] = np.real(phi_i)
        im_embeddings[:, start_idx:end_idx] = np.imag(phi_i)
    return np.concatenate([re_embeddings, im_embeddings], axis=1)


def characteristic_function2(psi, T: List[float] = np.arange(0, 101, 2), progress: bool = False):
    N = psi.shape[1]
    re_embeddings = np.zeros((N, len(psi) * len(T)))
    im_embeddings = np.zeros(re_embeddings.shape)

    len_T = len(T)
    for s_idx, psi_ in enumerate(tqdm(psi, disable=not progress)):
        for t_idx, t in enumerate(tqdm(T, leave=False, disable=not progress)):
            phi_i = phi(psi_, t)
            re_embeddings[:, s_idx * len_T + t_idx] = np.real(phi_i)
            im_embeddings[:, s_idx * len_T + t_idx] = np.imag(phi_i)
    return np.concatenate([re_embeddings, im_embeddings], axis=1)


def get_embeddings_fast(A: Union[nx.Graph, np.matrix], S: List[float] = None, T: List[float] = np.arange(0, 101, 2),
                        q: float = 0.5, progress: bool = False, normalize: bool = False) -> np.array:
    """
    Assumes that the heat kernel is used to compute embeddings faster
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html
    A is - L_q
    B is diagonal vector
    start, stop is S.

    replaces
        G_hat_s = np.diag(kernel(eigenvalues*s, **kernel_args))
        psi = U @ G_hat_s @ U.H @ delta
    """
    if isinstance(A, nx.Graph):
        A = get_adj(A)

    if progress: print("computing the hermitian laplacian")
    L_q = hermitian_laplacian(A, q)
    if normalize:
        if progress: print("normalizing laplacian")
        L_q = normalize_laplacian(A, L_q)
        

    psi = scipy_expm(L_q, S)
    embeddings = characteristic_function1(psi, T, progress)

    return embeddings


def get_embeddings(A: Union[nx.Graph, np.matrix], S: List[float] = None, T: List[float] = np.arange(0, 101, 2), q:float = 0.5,
                   kernel: callable = heat_kernel, progress: bool = False, normalize: bool = False, **kernel_args) -> np.ndarray:
    """
    Main function that computes graphwave embeddings from the Hermitian Laplacian
    Can be used with a (un)weighted (di)graph
    params:
    G: Adjacency matrix or networkx graph to extract embeddings from
    S: List of scale parameters. Determines the radius of network neighborhood
    T: List of sampling points
    q: Rotation parameter for the Hermitian Laplacian
    kernel: A kernel callable that can take a one-dimensional numpy array as input and returns a transformed version of the input
    **kernel_args: keyword arguments for the kernel function
    returns:
    a N x 2*|S|*|T| matrix
    if S is not specified, an appropriate range of values will be determined automatically according to the GraphWave paper
    """
    if isinstance(A, nx.Graph):
        A = get_adj(A)

    if progress: print("computing the hermitian laplacian")
    L_q = hermitian_laplacian(A, q)
    if normalize:
        if progress: print("normalizing laplacian")
        L_q = normalize_laplacian(A, L_q)
        
    if progress: print("computing eigenvectors and eigenvalues")
    psi, S = eigen_expm(L_q, S, kernel, progress, **kernel_args)
    embeddings = characteristic_function2(psi, T, progress)

    return embeddings