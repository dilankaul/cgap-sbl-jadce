"""Task-04 selected spatial dependence heatmaps from saved CSR matrices."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from cgap_sbl_jadce.paths import result_path, figure_path, task_dir
from cgap_sbl_jadce.spatial_dependence_dataset import load_spatial_dependence_matrix
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, selected_realizations


def plot_spatial_dependence(cfg: dict, formats: str = 'png', show_progress: bool = True) -> list[Path]:
    out = Path(cfg['run']['output_dir'])
    summary = json.loads(result_path(out, 'spatial_dependence', 'summary.json').read_text())
    figure_style = {'axes.titlesize': 12, 'axes.labelsize': 12,
                    'xtick.labelsize': 10, 'ytick.labelsize': 10,
                    'legend.fontsize': 10.5, 'colorbar.labelsize': 12, 'colorbar.ticksize': 10}
    paths = []
    for split in ('tuning', 'evaluation'):
        indices = selected_realizations(cfg, split, summary[split]['samples'])
        definition = summary[split]['definition']
        for q in tqdm(indices, desc=f'Spatial dependence figures ({split})', unit='realization',
                      dynamic_ncols=True, disable=not show_progress,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            C = load_spatial_dependence_matrix(task_dir(out, 4) / split, q).toarray()
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            image = ax.imshow(C, interpolation='nearest', vmin=0, vmax=1)
            fig.colorbar(image, ax=ax, label=r'$C_{i,j}$')
            ax.set_xlabel(r'MTD index $j$'); ax.set_ylabel(r'MTD index $i$')
            ax.set_title(r'Spatial Dependence Matrix $\mathbf{C}$' + '\n' +
                         fr"{split.capitalize()} realization {q}: $\rho={definition['rho']:g}$ m, $U={definition['U']:g}$ m")
            _apply_figure_style(fig, figure_style); fig.tight_layout()
            paths.extend(_save(fig, figure_path(out, f'04_{split}_spatial_dependence_matrix_{q:04d}'), formats))
    return paths
