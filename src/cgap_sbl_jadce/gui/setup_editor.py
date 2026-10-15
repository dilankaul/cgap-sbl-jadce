from pathlib import Path

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import grouped_config, range_parameters
from cgap_sbl_jadce.gui.setups import (create_setup, parse_values, validate_setup,
    update_setup)

LABELS = {'N': 'MTDs N', 'M': 'BS antennas M', 'S': 'Target mean active S', 'R': 'Cell radius R (m)',
          'V': 'Events V', 'D': 'Event influence radius D (m)', 'K': 'Maximum iterations K',
          'delta': 'Stopping tolerance delta', 'gamma_init': 'Initial gamma',
          'reference_pilot_length': 'Reference pilot length L', 'reference_snr_db': 'Reference SNR (dB)',
          'pilot_length': 'Pilot length for SNR sweep', 'snr_db': 'SNR for pilot-length sweep',
          'snr_start': 'SNR start', 'snr_stop': 'SNR stop', 'snr_step': 'SNR step'}


def _numbers(st, values, keys, prefix, labels=None):
    columns = st.columns(2)
    for i, name in enumerate(keys):
        value = values[name]
        if name in {'S', 'R', 'D', 'reference_snr_db', 'snr_db', 'gamma_init', 'delta', 'cosamp_tol',
                    'tau_start', 'tau_stop', 'tau_step', 'calibration_tolerance'} or any(
                    name == parameter + '_' + part for parameter in ('alpha', 'beta', 'kappa', 'snr')
                    for part in ('start', 'stop', 'step')):
            value = float(value)
        step = 1 if type(value) is int else value if name in {'delta', 'cosamp_tol'} and value > 0 else .001
        values[name] = columns[i % 2].number_input((labels or {}).get(name, LABELS.get(name, name.replace('_', ' ').capitalize())),
            value=value, step=step, format='%d' if type(value) is int else '%.6g', key=f'{prefix}_{name}')


def _list(st, values, name, prefix, integer=False):
    text = st.text_input(name.replace('_', ' ').capitalize(), ', '.join(str(v) for v in values[name]), key=f'{prefix}_{name}')
    return values, name, text, integer


def _range(st, settings, name, prefix, values_key=None, integer=False):
    key = values_key or name + '_values'
    fields = [name + '_' + part for part in ('start', 'stop', 'step')]
    if all(field in settings for field in fields):
        _numbers(st, settings, fields, prefix)
        return
    values = settings[key]
    proposed = range_parameters(values, name, integer)
    custom = proposed is None
    if custom:
        ordered = sorted(values)
        convert = int if integer else float
        step = min(b - a for a, b in zip(ordered, ordered[1:])) if len(ordered) > 1 else 1
        proposed = dict(zip(fields, map(convert, (ordered[0], ordered[-1], step))))
        st.caption('Custom values')
    original = dict(proposed)
    _numbers(st, proposed, fields, prefix)
    replace_custom = custom and st.checkbox('Use a regular ' + name.replace('_', ' ') + ' range', key=prefix + '_regular_' + name)
    if not custom or proposed != original or replace_custom:
        settings.pop(key, None)
        settings.update(proposed)


def settings_fields(st, cfg, prefix, task=None, action='run'):
    lists = []
    if task is None or task in {'kappa_calibration', 'activity'}:
        with st.expander('Environment', expanded=True):
            _numbers(st, cfg['system'], ['N', 'M', 'S', 'R'], prefix + '_system')
    if task is None or task in {'activity', 'communication'}:
        with st.expander('Realizations'):
            _numbers(st, cfg['run'], ['num_tuning_samples', 'num_evaluation_samples'], prefix + '_run')
    if task is None or task in {'activity', 'kappa_calibration'}:
        with st.expander('Activity model'):
            _numbers(st, cfg['activity'], ['V', 'D', 'batch_size'], prefix + '_activity')
            cfg['activity']['exact_sparsity'] = st.checkbox('Require an exact active count', value=cfg['activity'].get('exact_sparsity', False), key=prefix + '_exact_sparsity')
    if task is None or task == 'kappa_calibration':
        with st.expander('Kappa calibration', expanded=task is not None):
            cal = cfg['kappa_calibration']
            cal.setdefault('calibration_validation_samples', cal['calibration_samples'])
            _numbers(st, cal, ['calibration_samples', 'calibration_validation_samples', 'calibration_tolerance', 'batch_size'], prefix + '_kappa')
            _range(st, cal, 'kappa', prefix + '_kappa', 'kappa_scan')
            lists.append(_list(st, cal, 'kappa_refinement_steps', prefix + '_kappa'))
    if task is None or task == 'communication':
        with st.expander('Communication'):
            if task:
                _numbers(st, cfg['system'], ['M'], prefix + '_communication_system')
            _numbers(st, cfg['communication']['reference'], ['reference_pilot_length', 'reference_snr_db'], prefix + '_communication',
                {'reference_pilot_length': 'Communication preview pilot length L', 'reference_snr_db': 'Communication preview SNR (dB)'})
    if task is None or task in {'communication', 'comparison'} or task in cfg['algorithms'] and action == 'run':
        with st.expander('Evaluation conditions', expanded=True):
            evaluation = cfg['evaluation']
            evaluation.pop('condition_mode', None)
            evaluation.pop('conditions', None)
            _numbers(st, evaluation, ['pilot_length', 'snr_db'], prefix + '_evaluation')
            _range(st, evaluation, 'pilot_length', prefix + '_evaluation', 'pilot_lengths', integer=True)
            _range(st, evaluation, 'snr', prefix + '_evaluation', 'snr_values')
    for method, values in cfg['algorithms'].items():
        if not values.get('solver') and not values.get('tuning'):
            continue
        if task is not None and task != method:
            continue
        from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
        name = ESTIMATORS[method].name if method in ESTIMATORS else method.upper()
        with st.expander(name + ' settings', expanded=task == method):
            if values.get('solver'):
                st.caption('Solver')
                _numbers(st, values['solver'], list(values['solver']), prefix + '_' + method + '_solver')
            tune = values.get('tuning')
            if tune:
                st.caption('Tuning')
                tune['selection_mode'] = st.radio(name + ' tuning policy', ['reference', 'per_condition'],
                    index=int(tune['selection_mode'] == 'per_condition'), key=prefix + '_' + method + '_policy',
                    format_func=lambda value: 'One reference selection' if value == 'reference' else 'Separate selection per condition')
                tuning_label = 'Alpha-beta tuning' if method == 'anc' else 'Threshold tuning'
                _numbers(st, tune, ['reference_pilot_length', 'reference_snr_db', 'tau_start', 'tau_stop', 'tau_step'], prefix + '_' + method + '_tuning',
                    {'reference_pilot_length': tuning_label + ' pilot length L', 'reference_snr_db': tuning_label + ' SNR (dB)'})
                for parameter in ('alpha', 'beta'):
                    if parameter + '_start' in tune or parameter + '_values' in tune:
                        _range(st, tune, parameter, prefix + '_' + method)
    if task is None or task in {'convergence', 'sbl', 'anc'}:
        with st.expander('Analysis and diagnostics'):
            _numbers(st, cfg['evaluation'], ['gamma_distribution_samples', 'convergence_sample_offset'], prefix + '_analysis')
    seed_names = {'kappa_calibration': ['kappa_calibration', 'kappa_validation'], 'activity': ['activity_tuning', 'activity_evaluation'],
                  'communication': ['communication_tuning', 'communication_evaluation']}
    if task is None or task in seed_names:
        with st.expander('Independent random seeds'):
            _numbers(st, cfg['seeds'], seed_names.get(task, list(cfg['seeds'])), prefix + '_seeds')
    if task is None or task in {'activity', 'communication', 'spatial_dependence'}:
        with st.expander('Figure selections'):
            for split in ('tuning', 'evaluation'):
                key = split + '_indices'
                value = cfg['figures'][key]
                text = st.text_input(split.capitalize() + ' indices (or all)', value if value == 'all' else ', '.join(str(v) for v in value), key=prefix + '_' + key)
                lists.append((cfg['figures'], key, text, True))
    return lists


def _parse(lists):
    for values, key, text, integer in lists:
        values[key] = 'all' if key.endswith('_indices') and text.strip().lower() == 'all' else parse_values(text, integer)


def reset_settings_forms(st):
    for key in list(st.session_state):
        if key.startswith(('settings_form_', 'settings_draft_', 'new_setup_', 'edit_settings_')):
            del st.session_state[key]
    st.session_state['_settings_generation'] = st.session_state.get('_settings_generation', 0) + 1
    st.session_state.pop('_settings_context', None)


def form_prefix(st, context):
    if st.session_state.get('_settings_context') != context:
        reset_settings_forms(st)
        st.session_state['_settings_context'] = context
    return f'settings_form_{st.session_state.get("_settings_generation", 0)}'


def edit_settings(st, cfg, config, jobs=None, task=None, action='run'):
    version = Path(config).stat().st_mtime_ns
    prefix = form_prefix(st, (str(config), task, action, version))
    candidate = grouped_config(cfg)
    with st.form(prefix, enter_to_submit=False):
        lists = settings_fields(st, candidate, prefix, task, action)
        controls = st.columns(2)
        submitted = controls[0].form_submit_button('Save changes', type='primary', icon=':material/save:')
        controls[1].form_submit_button('Reset', icon=':material/restart_alt:', on_click=reset_settings_forms, args=(st,))
    if submitted:
        try:
            _parse(lists)
            validate_setup(candidate)
            update_setup(config, candidate, jobs)
        except (ValueError, TypeError, KeyError, OSError) as error:
            st.error(str(error))
        else:
            reset_settings_forms(st)
            st.rerun()


def setup_editor(st, project, default, current, jobs=None):
    mode = st.radio('Setup action', ['Edit current setup', 'Create setup'], horizontal=True)
    if mode == 'Edit current setup':
        st.subheader('Edit current setup')
        edit_settings(st, load_config(current), current, jobs)
        return
    st.subheader('Create an experiment setup')
    base_path = st.radio('Start from', [default, current] if current != default else [default],
        format_func=lambda value: 'Default settings' if value == default else 'Current setup', horizontal=True)
    cfg = grouped_config(load_config(base_path))
    prefix = form_prefix(st, ('create', str(base_path), Path(base_path).stat().st_mtime_ns))
    with st.form(prefix, enter_to_submit=False):
        name = st.text_input('Setup name', placeholder='N500 R50 experiment', key=prefix + '_name')
        lists = settings_fields(st, cfg, prefix)
        controls = st.columns(2)
        submitted = controls[0].form_submit_button('Save setup', type='primary', icon=':material/save:')
        controls[1].form_submit_button('Reset', icon=':material/restart_alt:', on_click=reset_settings_forms, args=(st,))
    if submitted:
        try:
            _parse(lists); validate_setup(cfg)
            path = create_setup(project, name, cfg)
        except (OSError, ValueError, TypeError) as error:
            st.error(str(error))
        else:
            reset_settings_forms(st)
            st.session_state['pending_setup'] = path
            st.rerun()
