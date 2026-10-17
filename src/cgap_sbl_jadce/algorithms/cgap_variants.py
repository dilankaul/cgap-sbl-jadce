from cgap_sbl_jadce.algorithms.cgap_sbl import _run_cgap_sbl
from cgap_sbl_jadce.algorithms.graph_interactions import spatial_penalty


def cgap_sbl_variant(Theta, Y, sigma2, C, method, alpha, beta=0., graph_kappa=0.,
                     diffusion_time=0., affinity_bandwidth=1., gamma_log_floor=1e-12, **kwargs):
    penalty = spatial_penalty(C, Theta.shape[1], method, alpha, beta, graph_kappa,
                              diffusion_time, affinity_bandwidth, gamma_log_floor)
    return _run_cgap_sbl(Theta, Y, sigma2, penalty, **kwargs)


def cgap_sbl_nglc(Theta, Y, sigma2, C, alpha, beta, **kwargs):
    return cgap_sbl_variant(Theta, Y, sigma2, C, 'nglc', alpha, beta=beta, **kwargs)


def cgap_sbl_rgl(Theta, Y, sigma2, C, alpha, graph_kappa, **kwargs):
    return cgap_sbl_variant(Theta, Y, sigma2, C, 'rgl', alpha, graph_kappa=graph_kappa, **kwargs)


def cgap_sbl_gd(Theta, Y, sigma2, C, alpha, diffusion_time, **kwargs):
    return cgap_sbl_variant(Theta, Y, sigma2, C, 'gd', alpha, diffusion_time=diffusion_time, **kwargs)


def cgap_sbl_aga(Theta, Y, sigma2, C, alpha, beta, affinity_bandwidth, **kwargs):
    return cgap_sbl_variant(Theta, Y, sigma2, C, 'aga', alpha, beta=beta,
                            affinity_bandwidth=affinity_bandwidth, **kwargs)
