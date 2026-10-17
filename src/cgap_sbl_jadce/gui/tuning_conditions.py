import json

from cgap_sbl_jadce.checkpoints import output_hashes
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments.estimators import (condition_config, load_selection, tuning_conditions,
                                                 tuning_definition)
from cgap_sbl_jadce.file_cache import validation_snapshot
from cgap_sbl_jadce.gui.algorithm_actions import CompletionPlan, _checkpoint_issue
from cgap_sbl_jadce.paths import tuning_path
from cgap_sbl_jadce.recovery import load_recoveries, validate_recovery


@validation_snapshot()
def tuning_plan(config, method, conditions=None):
    cfg = load_config(config)
    plan = CompletionPlan(tuning_conditions(cfg, conditions, method))
    pending = {}
    for saved in load_recoveries(config, method, 'tune'):
        try:
            pairs = tuning_conditions(cfg, saved['request'].get('conditions'), method)
        except (ValueError, KeyError):
            continue
        for pair in pairs:
            pending.setdefault(pair, saved)
    from cgap_sbl_jadce.tuning_results import pending_tuning_resets
    resetting = pending_tuning_resets(cfg, method)
    checked = {}
    for pair in plan.pairs:
        if pair in resetting:
            plan.blocked[pair] = 'An interrupted tuning reset must finish through Resume on its original request.'
            continue
        current = condition_config(cfg, pair, method)
        folder = tuning_path(current, method, '')
        selection_path = folder / 'selected.json'
        summary_path = folder / 'summary.json'
        units = list(folder.glob('.checkpoints/*/*.npz'))
        try:
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if summary.get('status') == 'complete':
                load_selection(current, method)
                outputs = summary.get('outputs', {})
                files = [folder / name for name in outputs]
                if not outputs or list(output_hashes(files).values()) != list(outputs.values()):
                    raise ValueError('Saved tuning outputs are missing or changed; rerun this condition.')
                plan.completed.append(pair)
                continue
            if selection_path.exists():
                load_selection(current, method)
            if units:
                issue = _checkpoint_issue(folder, tuning_definition(current, method))
                if issue:
                    raise ValueError(issue)
            if pair in pending:
                saved = pending[pair]
                if saved['path'] not in checked:
                    try:
                        validate_recovery(saved['request'])
                        checked[saved['path']] = None
                    except (OSError, ValueError, KeyError) as error:
                        checked[saved['path']] = str(error)
                if checked[saved['path']]:
                    raise ValueError(checked[saved['path']])
            if units or selection_path.exists() or summary_path.exists() or pair in pending:
                plan.resume.append(pair)
            else:
                plan.start.append(pair)
        except (OSError, ValueError, KeyError, TypeError) as error:
            plan.blocked[pair] = str(error)
    return plan
