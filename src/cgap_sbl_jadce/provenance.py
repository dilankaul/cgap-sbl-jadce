"""One run YAML and task-local JSON provenance."""
from copy import deepcopy
from pathlib import Path
import json
import platform
import os
import numpy as np
import scipy
from cgap_sbl_jadce.io import save_yaml, save_json
from cgap_sbl_jadce.paths import task_dir, result_path
from cgap_sbl_jadce.metrics import NMSE_SCOPE


def save_task_snapshot(cfg, task, **runtime):
    snapshot = deepcopy(cfg)
    out = cfg['run']['output_dir']
    provenance = dict(snapshot.get('task_runtime', {}))
    provenance.update(runtime)
    provenance['task'] = task
    if task not in {'kappa_calibration', 'activity'}:
        activity_path = result_path(out, 'activity', 'summary.json')
        if activity_path.exists():
            activity = json.loads(activity_path.read_text())
            snapshot['activity']['kappa'] = activity['kappa']
            provenance['activity_summary'] = str(activity_path)
            provenance['activity_parameters_used'] = {
                key: activity[key] for key in ('N', 'V', 'R', 'D', 'kappa', 'target_mean_S') if key in activity
            }
    if task in {'spatial_dependence', 'anc', 'convergence'}:
        from cgap_sbl_jadce.config import activity_spatial_dependence_parameters
        snapshot["spatial_dependence"] = activity_spatial_dependence_parameters(cfg)
        provenance["method_revision"] = 1
    if task in {'omp', 'cosamp', 'sbl', 'anc', 'comparison', 'convergence'}:
        provenance['nmse_scope'] = NMSE_SCOPE
    snapshot['task_runtime'] = provenance
    folder = task_dir(out, task)
    root_config = Path(out) / "config.yaml"
    is_source = Path(provenance.get('input_config', '')).resolve() == root_config.resolve()
    if not is_source and (task == 'activity' or not root_config.exists()):
        save_yaml(root_config, snapshot)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    save_json(manifest_path, {
        **manifest,
        'configuration': snapshot,
        'task': task, 'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        **({'timing_environment': {
            'machine': platform.machine(), 'processor': platform.processor(),
            'logical_cpus': os.cpu_count(),
            'thread_settings': {name: os.environ.get(name) for name in (
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        }} if task in {'omp', 'cosamp', 'sbl', 'anc'} else {}),
    })
    return snapshot
