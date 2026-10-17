import json

from cgap_sbl_jadce.communication_dataset import (validate_communication_dataset, communication_condition_identity,
                                                 communication_checkpoint_definition)
from cgap_sbl_jadce.paths import task_dir, condition_name
from cgap_sbl_jadce.settings import communication_conditions


def completion_plan(cfg, conditions=None):
    from cgap_sbl_jadce.gui.algorithm_actions import CompletionPlan, _checkpoint_issue
    plan = CompletionPlan(communication_conditions(cfg, conditions))
    root = task_dir(cfg['run']['output_dir'], 'communication')
    path = root / 'summary.json'
    summary = json.loads(path.read_text()) if path.exists() else {}
    available, issues = {}, []
    for split in ('evaluation', 'tuning'):
        available[split] = set()
        if split not in summary:
            continue
        try:
            validate_communication_dataset(cfg, split, [])
        except (OSError, ValueError, KeyError) as error:
            issues.append(str(error))
        else:
            available[split] = {tuple(pair) for pair in summary[split]['definition']['conditions']}
    for pair in plan.pairs:
        if issues:
            plan.blocked[pair] = issues[0]
            continue
        saved = []
        problem = None
        units = False
        for split in available:
            if pair in available[split]:
                try:
                    communication_condition_identity(cfg, split, pair)
                except (OSError, ValueError, KeyError) as error:
                    problem = str(error)
                else:
                    saved.append(split)
            owned = root / split / 'additional' / condition_name(pair)
            if list(owned.glob('.checkpoints/*/*.npz')):
                units = True
                try:
                    problem = problem or _checkpoint_issue(owned, communication_checkpoint_definition(cfg, split, pair))
                except (OSError, ValueError, KeyError) as error:
                    problem = str(error)
        if problem:
            plan.blocked[pair] = problem
        elif len(saved) == 2:
            plan.completed.append(pair)
        elif units or saved:
            plan.resume.append(pair)
        else:
            plan.start.append(pair)
    return plan


def condition_rows(cfg):
    pairs = communication_conditions(cfg, include_optional=True)
    plan = completion_plan(cfg, pairs)
    from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
    setup = set(communication_conditions(cfg))
    sweep = {value[:2] for value in evaluation_conditions(cfg)}
    rows = [{'Pilot length L': pair[0], 'SNR (dB)': pair[1],
             'Type': 'Sweep' if pair in sweep else 'Reference' if pair in setup else 'Optional',
             'Status': ('Completed' if pair in plan.completed else 'Incompatible' if pair in plan.blocked
                        else 'Unfinished' if pair in plan.resume else 'Not started')}
            for pair in pairs]
    return rows, list(dict.fromkeys(plan.blocked.values()))
