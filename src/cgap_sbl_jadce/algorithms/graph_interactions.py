import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply


def normalized_connectivity(C, N):
    if np.iscomplexobj(C):
        raise ValueError('C must contain real non-negative weights')
    C = csr_matrix(C, dtype=float, copy=True)
    C.sum_duplicates()
    if C.shape != (N, N):
        raise ValueError('C must have shape (N, N)')
    difference = C - C.T
    if (not np.all(np.isfinite(C.data)) or np.any(C.data < 0)
            or difference.nnz and np.max(np.abs(difference.data)) > 1e-10):
        raise ValueError('C must be finite, symmetric and non-negative')
    xi = np.asarray(C.sum(axis=1)).ravel()
    if not np.all(np.isfinite(xi)) or np.any(xi <= 0):
        raise ValueError('Connectivity must be finite and strictly positive')
    return C, 1. / np.sqrt(xi)


def normalized_affinity(C, inverse_sqrt_xi):
    return C.multiply(inverse_sqrt_xi[:, None]).multiply(inverse_sqrt_xi[None, :]).tocsr()


def glc_penalty(C, gamma, alpha, beta):
    return float(alpha) * np.maximum(float(beta) * np.sum(gamma) - C @ gamma, 0.)


def build_Omega_GLC(C, alpha, beta):
    C, _ = normalized_connectivity(C, C.shape[0])
    _nonnegative(alpha=alpha, beta=beta)
    return float(alpha) * (float(beta) * np.ones(C.shape) - C.toarray())


def _nonnegative(**values):
    for name, value in values.items():
        if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')


def spatial_penalty(C, N, method, alpha, beta=0., graph_kappa=0., diffusion_time=0.,
                    affinity_bandwidth=1., gamma_log_floor=1e-12):
    from cgap_sbl_jadce.variants import SPATIAL_METHODS
    if method not in SPATIAL_METHODS:
        raise ValueError('Unknown spatial interaction')
    _nonnegative(alpha=alpha)
    C, r = normalized_connectivity(C, N)
    if method in {'glc', 'nglc', 'aga'}:
        _nonnegative(beta=beta)
    if method == 'rgl':
        _nonnegative(graph_kappa=graph_kappa)
        if graph_kappa >= 1:
            raise ValueError('RGL needs 0 <= graph_kappa < 1; GL is the boundary at 1')
    if method == 'gd':
        _nonnegative(diffusion_time=diffusion_time)
    if method == 'aga':
        _nonnegative(affinity_bandwidth=affinity_bandwidth, gamma_log_floor=gamma_log_floor)
        if affinity_bandwidth == 0 or gamma_log_floor == 0:
            raise ValueError('AGA affinity_bandwidth and gamma_log_floor must be strictly positive')
    if alpha == 0 or method == 'gd' and diffusion_time == 0:
        return lambda gamma: np.zeros_like(gamma)
    if method == 'glc':
        return lambda gamma: glc_penalty(C, gamma, alpha, beta)
    F = normalized_affinity(C, r)
    if method == 'nglc':
        return lambda gamma: alpha * np.maximum(beta * np.mean(gamma) - F @ gamma, 0.)
    if method in {'rgl', 'gl'}:
        coupling = graph_kappa if method == 'rgl' else 1.
        return lambda gamma: alpha * np.maximum(gamma - coupling * (F @ gamma), 0.)
    if method == 'gd':
        operator = -diffusion_time * (eye(N, format='csr') - F)
        trace = float(operator.diagonal().sum())
        return lambda gamma: alpha * np.maximum(gamma - expm_multiply(operator, gamma, traceA=trace), 0.)

    edges = C.tocoo()
    def adaptive(gamma):
        # The floor affects affinity logarithms only, never the SBL variances.
        log_gamma = np.log(np.maximum(gamma, gamma_log_floor))
        with np.errstate(over='ignore'):
            distance = (log_gamma[edges.row] - log_gamma[edges.col]) / affinity_bandwidth
            weights = edges.data * np.exp(-.5 * distance ** 2)
        A = csr_matrix((weights, (edges.row, edges.col)), shape=C.shape)
        xi = np.asarray(A.sum(axis=1)).ravel()
        if np.any(xi <= 0) or not np.all(np.isfinite(xi)):
            raise ValueError('AGA adaptive connectivity must remain finite and positive')
        F_k = normalized_affinity(A, 1. / np.sqrt(xi))
        return alpha * np.maximum(beta * np.mean(gamma) - F_k @ gamma, 0.)
    return adaptive
