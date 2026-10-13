"""Code-defined setup form; plotting styles stay in figure code."""
from copy import deepcopy

from cgap_sbl_jadce.gui.setups import create_setup, parse_values, validate_setup


def setup_editor(st, project, default, current):
    from cgap_sbl_jadce.config import load_config
    st.subheader('Create an experiment setup')
    base_path = st.radio('Start from', [default, current] if current != default else [default],
                         format_func=lambda value: 'Default settings' if value == default else 'Current setup', horizontal=True)
    base = deepcopy(load_config(base_path))
    st.caption('Edit the settings before saving. Each setup has its own configuration, datasets, results and checkpoints.')
    with st.form('new_setup_' + str(base_path)):
        name = st.text_input('Setup name', placeholder='N500 R50 experiment')
        fields = [
            ('system', 'System and environment', ['N', 'M', 'S', 'R']),
            ('run', 'Realizations', ['num_tuning_samples', 'num_evaluation_samples']),
            ('activity', 'Activity and kappa calibration', ['V', 'D', 'batch_size', 'calibration_samples', 'calibration_tolerance']),
            ('tuning', 'Reference condition', ['reference_pilot_length', 'reference_snr_db']),
            ('evaluation', 'Evaluation and convergence', ['pilot_length', 'gamma_distribution_samples', 'convergence_sample_offset']),
            ('algorithms', 'Solver settings', ['gamma_init', 'K', 'delta', 'cosamp_max_iter', 'cosamp_tol']),
            ('seeds', 'Independent random seeds', list(base['seeds'])),
        ]
        for group, title, keys in fields:
            with st.expander(title, expanded=group in {'system', 'run'}):
                columns = st.columns(3)
                for i, key in enumerate(keys):
                    value = base[group].get(key, 0)
                    if group == 'system' and key == 'S':
                        value = float(value)
                    label = {'N': 'MTDs N', 'M': 'BS antennas M', 'S': 'Target mean active S', 'R': 'Cell radius R (m)',
                             'D': 'Event influence radius D (m)'}.get(key, key.replace('_', ' ').capitalize())
                    base[group][key] = columns[i % 3].number_input(label, value=value, step=1 if type(value) is int else .001,
                                                                  key=f'{group}_{key}_{base_path}')
        lists = {}
        with st.expander('Sweep ranges', expanded=True):
            for group, key, integer in [('system', 'pilot_lengths', True), ('system', 'snr_db', False),
                    ('tuning', 'alpha_values', False), ('tuning', 'beta_values', False),
                    ('activity', 'kappa_scan', False), ('activity', 'kappa_refinement_steps', False)]:
                lists[group, key] = (st.text_input(key.replace('_', ' ').capitalize(), ', '.join(str(v) for v in base[group][key])), integer)
        base['evaluation']['condition_mode'] = st.radio('Evaluation conditions', ['sweeps', 'grid'],
            index=int(base['evaluation'].get('condition_mode') == 'grid'),
            format_func=lambda mode: 'Pilot-length and SNR sweeps at fixed reference values (default)' if mode == 'sweeps' else 'All pilot lengths × all SNR values')
        policy = st.radio('SBL / ANC tuning policy', ['reference', 'per_condition'],
                          index=int(base['tuning'].get('selection_mode') == 'per_condition'),
                          format_func=lambda value: 'One reference selection shared across conditions' if value == 'reference' else 'Tune separately for each L/SNR condition')
        st.caption('Separate tuning uses only tuning realizations at each condition; held-out evaluation remains separate.')
        base['tuning']['selection_mode'] = policy
        figure_text = {}
        with st.expander('Standard figure selections'):
            for split in ('tuning', 'evaluation'):
                values = base['figures'][f'{split}_indices']
                figure_text[split] = st.text_input(f'{split.capitalize()} indices (or all)', values if values == 'all' else ', '.join(str(v) for v in values))
        submitted = st.form_submit_button('Create setup', type='primary', icon=':material/add:')
        if submitted:
            try:
                for (group, key), (text, integer) in lists.items():
                    base[group][key] = parse_values(text, integer)
                for split, text in figure_text.items():
                    base['figures'][f'{split}_indices'] = 'all' if text.strip().lower() == 'all' else parse_values(text, True)
                validate_setup(base)
                path = create_setup(project, name, base)
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.session_state['pending_setup'] = path
                st.rerun()
