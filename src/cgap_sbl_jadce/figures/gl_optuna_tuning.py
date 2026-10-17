from cgap_sbl_jadce.figures.spatial_optuna_tuning import plot_spatial_optuna_tuning


def plot_gl_optuna_tuning(cfg, formats='png'):
    return plot_spatial_optuna_tuning(cfg, 'gl', formats)
