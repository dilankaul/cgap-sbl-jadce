"""Deterministic coarse-to-fine activity calibration."""
from dataclasses import asdict
import numpy as np
from cgap_sbl_jadce.models.activity import calibrate_kappa_grid, best_kappa_for_target


class CalibrationBracketError(ValueError):
    def __init__(self, message, records):
        super().__init__(message)
        self.records = records


def calibrate_kappa_adaptive(*, target, coarse_values, refinement_steps,
                             validation_seed, validation_samples, tolerance=0.1, checkpoint=None, **kwargs):
    steps = [float(step) for step in refinement_steps]
    coarse = sorted(set(float(value) for value in coarse_values))
    if len(coarse) < 2 or any(not np.isfinite(x) or x <= 0 for x in coarse):
        raise ValueError('Coarse scan needs at least two finite positive values')
    if not steps or any(not np.isfinite(x) or x <= 0 for x in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError('Refinement steps must be positive and strictly decreasing')
    if not np.isfinite(target) or target <= 0 or tolerance <= 0:
        raise ValueError('Target and tolerance must be positive')
    if validation_seed == kwargs['seed'] or validation_samples < 2:
        raise ValueError('Validation needs an independent seed and at least two samples')
    cache = {}
    records = []
    stages = []

    def scan(values, stage):
        fresh = sorted(set(round(float(x), 12) for x in values) - cache.keys())
        if fresh:
            if kwargs.get('show_progress', True):
                print(f'Kappa scan {stage}: {fresh[0]:g} to {fresh[-1]:g}, {len(fresh)} new candidates', flush=True)
            # Same seed, sample count, and batching replay identical geometries
            # and Bernoulli uniforms across every refinement stage.
            options = dict(kwargs)
            if checkpoint:
                options['checkpoint'] = checkpoint.child(stage, {'values': fresh, **{key: value for key, value in kwargs.items() if key != 'show_progress'}})
            rows = calibrate_kappa_grid(kappa_values=fresh, **options)
            for row in rows:
                cache[round(row.kappa, 12)] = row
                records.append({**asdict(row), 'stage': stage})

    def bracket():
        rows = sorted(cache.values(), key=lambda row: row.kappa)
        for left, right in zip(rows, rows[1:]):
            if left.mean_expected_active <= target <= right.mean_expected_active:
                return left.kappa, right.kappa
        raise CalibrationBracketError('Target activity is not bracketed; widen the configured coarse kappa scan.', records)

    scan(coarse, 'coarse')
    low, high = bracket()
    stages.append({'stage': 'coarse', 'bracket': [low, high]})
    for index, step in enumerate(steps, 1):
        # Align the final grid globally, yielding values such as 3.775, 3.780.
        start = int(np.floor(low / step + 1e-9))
        stop = int(np.ceil(high / step - 1e-9))
        values = [round(i * step, 12) for i in range(max(1, start), stop + 1)]
        scan(values, f'refine_{index}')
        low, high = bracket()
        stages.append({'stage': f'refine_{index}', 'step': step, 'bracket': [low, high]})
    # Choose from the final aligned grid, rather than an off-grid coarse point.
    best = best_kappa_for_target([cache[x] for x in values], target)
    validation_args = {**kwargs, 'seed': validation_seed, 'num_samples': validation_samples}
    if checkpoint:
        validation_args['checkpoint'] = checkpoint.child('validation', {'kappa': best.kappa, 'seed': validation_seed, 'samples': validation_samples})
    validation = calibrate_kappa_grid(kappa_values=[best.kappa], **validation_args)[0]
    validation_info = {
        **asdict(validation), 'seed': validation_seed, 'samples': validation_samples,
        'expected_gap': validation.mean_expected_active - target,
        'expected_mean_ci95': [validation.mean_expected_active - 1.96 * validation.se_expected_active,
                               validation.mean_expected_active + 1.96 * validation.se_expected_active],
        'within_tolerance': abs(validation.mean_expected_active - target) <= tolerance,
    }
    return sorted(cache.values(), key=lambda row: row.kappa), best, records, {
        'status': 'success', 'method': 'coarse_to_fine', 'stages': stages,
        'final_step': steps[-1], 'tolerance': tolerance,
        'selection_within_tolerance': abs(best.mean_expected_active - target) <= tolerance,
        'validation': validation_info,
    }
