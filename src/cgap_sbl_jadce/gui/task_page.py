"""Action controls, cooperative worker management, and graphs in one task page."""
import json

from cgap_sbl_jadce.tasks import task_actions
from cgap_sbl_jadce.gui.jobs import ACTIVE, RECOVERABLE
from cgap_sbl_jadce.gui.figure_panel import figure_panel
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.experiments.estimators import (ESTIMATORS, evaluation_conditions,
                                                completed_methods, convergence_methods)


def analysis_options(cfg, task, conditions=None, condition=None):
    """Only completed runs can be selected; greedy methods have no SBL convergence trace."""
    return (completed_methods(cfg, conditions) if task == 'comparison'
            else convergence_methods(cfg, condition))


def worker_panel(st, jobs, config, task, action):
    @st.fragment(run_every=2)
    def panel():
        record = jobs.task_job(config, task, action)
        if not record:
            return
        status = record['status']
        command = json.loads((jobs.folder / record['id'] / 'control.json').read_text())['command']
        st.subheader('Task progress')
        text = 'Pause requested · finishing current calculation' if command == 'pause' and status in {'running','queued'} else status.capitalize()
        st.write(text)
        controls = st.columns(3)
        try:
            if controls[0].button('Pause', icon=':material/pause:', disabled=status not in {'running','queued'} or command != 'run', width='stretch'):
                jobs.control(record['id'], 'pause'); st.rerun()
            if controls[1].button('Continue', icon=':material/play_arrow:', disabled=status != 'paused' or jobs.busy() is not None, width='stretch'):
                jobs.control(record['id'], 'run'); st.rerun()
            if controls[2].button('Stop', icon=':material/stop:', disabled=status not in ACTIVE or command == 'stop', width='stretch'):
                jobs.control(record['id'], 'stop'); st.rerun()
            if status in RECOVERABLE and st.button('Restart from checkpoints', icon=':material/restart_alt:', disabled=jobs.busy() is not None):
                jobs.restart(record['id']); st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
        logs, progress = jobs.log(record['id'])
        if progress:
            st.progress(min(1., progress['done']/max(1,progress['total'])),
                        text=f"{progress['label']}: {progress['done']}/{progress['total']}")
        if record.get('error'):
            st.error(record['error'])
        with st.expander('Live log', expanded=status in ACTIVE):
            st.code(logs or 'Starting worker…', language=None)
    panel()


def task_page(st, cfg, config, jobs, task, action):
    spec = ESTIMATORS.get(task)
    group = 'Analysis' if task in {'comparison','convergence'} else 'Algorithms' if spec else 'Data preparation'
    st.caption(f'{group} / {action.capitalize()}')
    description = task_actions(task)[action].description
    st.title(f'{action.capitalize()} {spec.name}' if spec else description[0].upper()+description[1:])
    conditions = condition = methods = q = None
    selectable = (task in ESTIMATORS or task == 'comparison') and action == 'run' or (
        action == 'tune' and cfg['tuning'].get('selection_mode') == 'per_condition')
    if selectable:
        pairs = ([tuple(pair) for pair in cfg.get('communication', {}).get('tuning_conditions', [])]
                 if action == 'tune' else list(dict.fromkeys(pair[:2] for pair in evaluation_conditions(cfg))))
        scope = st.radio('Condition scope', ['All conditions', 'Selected conditions'], horizontal=True)
        if scope == 'Selected conditions':
            selected = st.multiselect('L / SNR conditions', pairs, default=pairs[:1],
                                     format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB',
                                     key=f'pairs_{config}_{task}_{action}')
            conditions = [list(pair) for pair in selected]
        st.caption('Completed conditions are retained. Further runs reuse compatible checkpoints.')
    if task == 'convergence':
        pairs = list(dict.fromkeys(pair[:2] for pair in evaluation_conditions(cfg)))
        reference = (cfg['evaluation']['pilot_length'], cfg['tuning']['reference_snr_db'])
        condition = st.selectbox('L / SNR condition', pairs, index=pairs.index(reference) if reference in pairs else 0,
                                format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB')
        q = int(st.number_input('Evaluation realization index (zero-based)', min_value=0,
            max_value=cfg['run']['num_evaluation_samples']-1,
            value=min(cfg['evaluation'].get('convergence_sample_offset',0),cfg['run']['num_evaluation_samples']-1), step=1))
    if task in {'comparison','convergence'}:
        available = analysis_options(cfg, task, conditions, condition) if conditions != [] else []
        # Changing the condition removes stale selections before the widget is rendered.
        key = f'methods_{config}_{task}'
        if key in st.session_state:
            st.session_state[key] = [method for method in st.session_state[key] if method in available]
        methods = st.multiselect('Algorithms', available, default=available,
                                format_func=lambda method: ESTIMATORS[method].name, key=key,
                                disabled=not available)
        st.caption('Only completed algorithms for the selected conditions are available.' if task == 'comparison'
                   else 'Only completed SBL and CGAP-SBL-ANC runs are available. Convergence records their iteration histories.')
    if spec:
        if action == 'tune':
            count = len(conditions) if conditions is not None else (len(cfg.get('communication',{}).get('tuning_conditions',[])) if cfg['tuning'].get('selection_mode') == 'per_condition' else 1)
            candidates = len(spec.tuning_grid(cfg))
            st.caption(f'{count} conditions × {candidates} parameter candidates × {cfg["run"]["num_tuning_samples"]} tuning realizations before checkpoint reuse.')
        if spec.tuning_grid:
            st.caption('Tuning policy: separate selection per condition.' if cfg['tuning'].get('selection_mode') == 'per_condition'
                else f"Tuning policy: reference selection at L={cfg['tuning']['reference_pilot_length']}, SNR={cfg['tuning']['reference_snr_db']:g} dB.")
    reasons = (['Select at least one condition.'] if conditions == [] else
               blocking_reasons(cfg, task, action, conditions, methods, condition))
    for reason in reasons:
        st.info(reason)
    own = jobs.task_job(config, task, action)
    if st.button('Start '+action, type='primary', icon=':material/play_arrow:', disabled=bool(reasons) or jobs.busy() is not None or bool(own and own['status'] in ACTIVE)):
        request = {'kind':'task', 'task':task, 'action':action, 'config':str(config)}
        for key, value in [('conditions',conditions),('methods',methods),('realization_index',q),('condition',condition)]:
            if value is not None:
                request[key] = value
        try:
            jobs.start(request); st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    st.caption('Pause takes effect at the next checkpoint. Once paused, another task can run. Return here to Continue. Restart reuses compatible completed calculations after a failure.')
    worker_panel(st, jobs, config, task, action)
    st.divider()
    figure_panel(st, cfg, config, jobs, task, action)
