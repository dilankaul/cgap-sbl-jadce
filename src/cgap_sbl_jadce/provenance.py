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
    if task >= 3:
        activity_path = result_path(out, 'activity', 'summary.json')
        if activity_path.exists():
            activity = json.loads(activity_path.read_text())
            snapshot['activity']['kappa'] = activity['kappa']
            provenance['activity_summary'] = str(activity_path)
            if "randomized_evaluation" in activity:
                seed = activity["randomized_evaluation"]["seed"]
                snapshot["seeds"]["spatial_randomization"] = seed
                provenance["activity_randomization_seed"] = seed

            provenance['activity_parameters_used'] = {
                key: activity[key] for key in ('N', 'V', 'R', 'D', 'kappa', 'target_mean_S') if key in activity
            }
    if task >= 4:
        from cgap_sbl_jadce.config import activity_spatial_dependence_parameters
        snapshot["spatial_dependence"] = activity_spatial_dependence_parameters(cfg)
        provenance["method"] = "CGAP-SBL-ANC"
        provenance["method_revision"] = 1
    if task >= 5:
        provenance['nmse_scope'] = NMSE_SCOPE
    snapshot['task_runtime'] = provenance
    folder = task_dir(out, task)
    root_config = Path(out) / "config.yaml"
    if task == 2 or not root_config.exists():
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
        }} if task == 6 else {}),
    })
    return snapshot
