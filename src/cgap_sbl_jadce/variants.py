SPATIAL_METHODS = ('glc', 'nglc', 'rgl', 'gl', 'gd', 'aga')
TUNED_METHODS = ('sbl', *SPATIAL_METHODS)
ALGORITHM_METHODS = ('omp', 'cosamp', *TUNED_METHODS)

VARIANTS = {
    'glc': ('Global–Local Contrast', ('alpha', 'beta'), {'alpha': (.001, 1000.), 'beta': (0., 5.)}),
    'nglc': ('Normalized Global–Local Contrast', ('alpha', 'beta'), {'alpha': (.001, 1000.), 'beta': (0., 5.)}),
    'rgl': ('Regularized Graph Laplacian', ('alpha', 'graph_kappa'), {'alpha': (.001, 1000.), 'graph_kappa': (0., 1.)}),
    'gl': ('Normalized Graph Laplacian', ('alpha',), {'alpha': (.001, 1000.)}),
    'gd': ('Graph Diffusion', ('alpha', 'diffusion_time'), {'alpha': (.001, 1000.), 'diffusion_time': (.01, 10.)}),
    'aga': ('Adaptive Graph Affinity', ('alpha', 'beta', 'affinity_bandwidth'),
            {'alpha': (.001, 1000.), 'beta': (0., 5.), 'affinity_bandwidth': (.01, 10.)}),
}
SYMBOLS = {'alpha': 'α', 'beta': 'β', 'tau': 'τ', 'graph_kappa': 'κ_g',
           'diffusion_time': 't', 'affinity_bandwidth': 'h'}
MATH_SYMBOLS = {'alpha': r'\alpha', 'beta': r'\beta', 'tau': r'\tau',
                'graph_kappa': r'\kappa_{\mathrm{g}}', 'diffusion_time': 't', 'affinity_bandwidth': 'h'}
PARAMETER_LABELS = {'alpha': 'Alpha α', 'beta': 'Beta β', 'graph_kappa': 'Graph coupling κ_g',
                    'diffusion_time': 'Diffusion time t', 'affinity_bandwidth': 'Affinity bandwidth h'}


def parameters(method):
    return VARIANTS[method][1] if method in VARIANTS else ()


def search_defaults(method):
    settings = {'search_method': 'optuna_tpe', 'trials': 50, 'samples': 100,
                'startup_trials': 10, 'optuna_seed': 17389}
    for name, (low, high) in VARIANTS[method][2].items():
        settings.update({name + '_low': low, name + '_high': high,
                         name + '_log': name in {'alpha', 'diffusion_time', 'affinity_bandwidth'}})
    return settings
