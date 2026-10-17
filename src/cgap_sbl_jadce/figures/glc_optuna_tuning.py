from cgap_sbl_jadce.figures.spatial_optuna_tuning import plot_spatial_optuna_tuning


def plot_glc_optuna_tuning(cfg, formats='png'):
    return plot_spatial_optuna_tuning(cfg, 'glc', formats)
