import json

from cgap_sbl_jadce.additional_conditions import additional_conditions
from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
from cgap_sbl_jadce.gui.jobs import ACTIVE, RECOVERABLE
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.settings import communication_pairs


def condition_rows(cfg):
    expected = {split: set(communication_pairs(cfg, split)) for split in ('tuning', 'evaluation')}
    required = set.union(*expected.values())
    optional = set(additional_conditions(cfg))
    path = result_path(cfg['run']['output_dir'], 'communication', 'summary.json')
    summary = json.loads(path.read_text()) if path.exists() else {}
    available, issues = {}, []
    for split in expected:
        available[split] = set()
        try:
            validate_communication_dataset(cfg, split, [])
        except (OSError, ValueError, KeyError) as error:
            issues.append(str(error))
        else:
            available[split] = {tuple(pair) for pair in summary.get(split, {}).get('definition', {}).get('conditions', [])}
    rows = []
    for L, snr in sorted(required | optional):
        pair = (L, snr)
        row = {'Pilot length L': L, 'SNR (dB)': snr,
               'Type': 'Optional' if pair in optional and pair not in required else 'Setup'}
        for split in expected:
            row[split.capitalize() + ' inputs'] = ('Ready' if pair in available[split] else 'Missing') if pair in expected[split] | optional else 'Not required'
        rows.append(row)
    return rows, list(dict.fromkeys(issues))


def communication_condition_controls(st, cfg, config, jobs):
    rows, issues = condition_rows(cfg)
    pairs = [(row['Pilot length L'], row['SNR (dB)']) for row in rows]
    pending = [pair for pair, row in zip(pairs, rows) if 'Missing' in (row['Tuning inputs'], row['Evaluation inputs'])]
    with st.expander('Condition status', expanded=True):
        st.dataframe(rows, hide_index=True, width='stretch')
        for issue in issues:
            st.info(issue)
        key = f'communication_conditions_{config}'
        if key in st.session_state:
            st.session_state[key] = [pair for pair in st.session_state[key] if pair in pairs]
        selected = st.multiselect('Communication conditions', pairs,
            default=None if key in st.session_state else pending,
            format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB', key=key)
        controls = st.columns(2)
        for column, selection, label, suffix in (
                (controls[0], [pair for pair in selected if pair in pending], 'Complete selected inputs', 'selected'),
                (controls[1], pending, 'Complete all missing inputs', 'all')):
            request = {'kind': 'task', 'task': 'communication', 'action': 'run', 'config': str(config),
                       'conditions': [list(pair) for pair in selection]}
            own = jobs.task_job(config, 'communication', 'run', conditions=request['conditions']) if selection else None
            recoverable = bool(own and own['status'] in RECOVERABLE | {'paused'} and own['request'] == request)
            if recoverable:
                from cgap_sbl_jadce.recovery import validate_recovery
                try:
                    validate_recovery(request)
                except (OSError, ValueError, KeyError):
                    recoverable = False
            disabled = not selection or bool(issues) or jobs.busy() is not None or bool(own and own['status'] in ACTIVE and not recoverable)
            if column.button('Resume input preparation' if recoverable else label, disabled=disabled,
                             key=f'complete_communication_{config}_{suffix}'):
                try:
                    jobs.resume(own['id']) if recoverable else jobs.start(request)
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(str(error))
