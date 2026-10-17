import numpy as np

from cgap_sbl_jadce.algorithms.graph_interactions import normalized_connectivity
from cgap_sbl_jadce.algorithms.cgap_variants import cgap_sbl_variant


def gl_penalty(C, inverse_sqrt_xi, gamma, alpha):
    return float(alpha) * np.maximum(gamma - inverse_sqrt_xi * (C @ (inverse_sqrt_xi * gamma)), 0.)


def cgap_sbl_gl(Theta, Y, sigma2, C, alpha, **kwargs):
    return cgap_sbl_variant(Theta, Y, sigma2, C, 'gl', alpha, **kwargs)
