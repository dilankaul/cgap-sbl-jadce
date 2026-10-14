"""Action controls, cooperative worker management, and graphs in one task page."""
import json

from cgap_sbl_jadce.tasks import task_actions
from cgap_sbl_jadce.settings import tuning_settings, tuning_pairs, evaluation_settings
from cgap_sbl_jadce.gui.jobs import ACTIVE, RECOVERABLE
from cgap_sbl_jadce.gui.figure_panel import figure_panel
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.experiments.estimators import (ESTIMATORS, evaluation_conditions,
                                                completed_methods, convergence_methods)


def analysis_options(cfg, task, conditions=None, condition=None):
    """Only completed runs can be selected; greedy methods have no SBL convergence trace."""
    return (completed_methods(cfg, conditions) if task == 'comparison'
            else convergence_methods(cfg, condition))


def worker_panel(st, jobs, config, task, action, conditions=None):
    @st.fragment(run_every=2)
    def panel():
        record = jobs.task_job(config, task, action, conditions=conditions)
        if conditions is not None and (record is None or record['status'] not in ACTIVE):
            active = jobs.task_job(config, task, action)
            if active and active['status'] in ACTIVE:
                record = active
        if not record:
            return
        status = record['status']
        command = ('pause' if record.get('portable') else
                   json.loads((jobs.folder / record['id'] / 'control.json').read_text())['command'])
        st.subheader('Task progress')
        if task in ESTIMATORS and action == 'run':
            pairs = record['request'].get('conditions')
            st.caption('Running scope: All conditions' if pairs is None else 'Running scope: ' +
                       ', '.join(f'L={L}, SNR={snr:g} dB' for L, snr in pairs))
        text = status.capitalize()
        if command == 'pause' and status in {'running', 'queued'}:
            text = 'Pause requested · finishing current calculation'
        elif command == 'run' and status == 'paused':
            text = 'Resuming…'
        elif command == 'stop' and status in ACTIVE:
            text = 'Stopping…'
        st.write(text)
        controls = st.columns(3)
        can_resume = status in RECOVERABLE | {'paused'} and jobs.busy() is None
        if can_resume and record['request'].get('kind') == 'task':
            from cgap_sbl_jadce.recovery import validate_recovery
            try:
                validate_recovery(record['request'])
            except (OSError, ValueError, KeyError) as error:
                can_resume = False
                st.warning('Resume unavailable: ' + str(error))
        if status in ACTIVE and command == 'stop':
            can_resume = False
        try:
            if controls[0].button('Pause', icon=':material/pause:', disabled=status not in {'running','queued'} or command != 'run', width='stretch'):
                jobs.control(record['id'], 'pause'); st.rerun()
            if controls[1].button('Resume', icon=':material/play_arrow:',
                                  disabled=not can_resume,
                                  help='Continue a paused worker or recover completed calculations from checkpoints.', width='stretch'):
                jobs.resume(record['id']); st.rerun()
            if controls[2].button('Stop', icon=':material/stop:', disabled=status not in ACTIVE or command == 'stop', width='stretch'):
                jobs.control(record['id'], 'stop'); st.rerun()
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


def optional_input_controls(st, cfg, config, jobs, method, pairs):
    from cgap_sbl_jadce.additional_conditions import input_reasons, input_splits
    from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
    prefix = f'{config}_{method}_run'
    if st.session_state.get('additional_pair_' + prefix) not in pairs:
        st.session_state.pop('additional_pair_' + prefix, None)
    pair = st.selectbox('Optional condition inputs', pairs, key='additional_pair_' + prefix,
                       format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB')
    reasons = input_reasons(cfg, method, pair)
    if reasons:
        st.info('Prepare communication inputs before tuning or running this condition.')
        base_issues = []
        for split in input_splits(cfg, [method]):
            try:
                validate_communication_dataset(cfg, split, [])
            except (OSError, ValueError, KeyError) as error:
                base_issues.append(str(error))
        for reason in base_issues:
            st.info(reason)
        request = {'kind': 'task', 'task': 'communication', 'action': 'run', 'config': str(config),
                   'conditions': [list(pair)], 'methods': [method]}
        own = jobs.task_job(config, 'communication', 'run', conditions=[list(pair)])
        recoverable = own and own['status'] in RECOVERABLE | {'paused'} and own['request'] == request
        if recoverable:
            from cgap_sbl_jadce.recovery import validate_recovery
            try:
                validate_recovery(request)
            except (OSError, ValueError, KeyError):
                recoverable = False
        if st.button('Resume input preparation' if recoverable else 'Prepare inputs', key='prepare_optional_' + prefix,
                     disabled=bool(base_issues) or jobs.busy() is not None or bool(own and own['status'] in ACTIVE and not recoverable)):
            try:
                jobs.resume(own['id']) if recoverable else jobs.start(request)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))
    else:
        st.success('Communication inputs are ready.')
        if ESTIMATORS[method].tuning_grid and tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
            st.caption('Tune this condition on the algorithm’s Tune page before evaluation.')
    worker_panel(st, jobs, config, 'communication', 'run', [list(pair)])



def condition_status_panel(st, cfg, config, jobs, method, plan):
    """Read-only sweep rows and optional rows with inline Add/Delete controls."""
    from cgap_sbl_jadce.additional_conditions import additional_conditions, add_condition, remove_condition
    from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
    from cgap_sbl_jadce.gui.setup_editor import reset_settings_forms
    prefix = f'{config}_{method}_run'
    pairs = [pair for pair in additional_conditions(cfg) if pair not in plan.pairs]
    optional_plan = completion_plan(config, method, pairs) if pairs else None
    with st.expander('Condition status', expanded=True):
        st.caption('Add or delete optional rows here. Optional rows are shared across algorithms and do not count toward sweep completion.')
        widths = [1, 1, 1, 2, 1]
        with st.container(border=True, key='condition_status_table'):
            for column, label in zip(st.columns(widths), ['Pilot length L', 'SNR (dB)', 'Type', 'Status', '']):
                column.markdown('**' + label + '**' if label else '')
            for current_plan, kind in [(plan, 'Sweep'), (optional_plan, 'Optional')]:
                if current_plan is None:
                    continue
                for pair in current_plan.pairs:
                    status = ('Completed' if pair in current_plan.completed else 'Incompatible' if pair in current_plan.blocked
                              else 'Unfinished' if pair in current_plan.resume else 'Not started')
                    columns = st.columns(widths, vertical_alignment='center')
                    for column, text in zip(columns, [str(pair[0]), f'{pair[1]:g}', kind, status]):
                        column.write(text)
                    if kind == 'Optional':
                        in_use = any(job['status'] in ACTIVE and job['request'].get('kind') == 'task' and
                            str(job['request']['config']) == str(config) and
                            (pair in {tuple(value) for value in (job['request'].get('conditions') or [])} or
                             tuple(job['request'].get('condition') or []) == pair) for job in jobs.records())
                        if columns[4].button('Delete', key=f'delete_optional_{prefix}_{pair}', icon=':material/delete:',
                                             disabled=in_use, help='Remove this row for every algorithm. Saved data are retained.'):
                            try:
                                remove_condition(cfg, pair)
                            except (OSError, ValueError) as error:
                                st.error(str(error))
                            else:
                                st.session_state.pop('additional_pair_' + prefix, None)
                                st.session_state.pop('pairs_' + prefix, None)
                                st.rerun()
            form_key = f'settings_form_additional_{st.session_state.get("_settings_generation", 0)}'
            with st.form(form_key, enter_to_submit=False, border=False):
                columns = st.columns(widths, vertical_alignment='center')
                L = columns[0].number_input('Additional pilot length L', min_value=1, value=None, step=1,
                    key=form_key + '_L', label_visibility='collapsed', placeholder='L')
                snr = columns[1].number_input('Additional SNR (dB)', value=None, step=1., format='%.6g',
                    key=form_key + '_snr', label_visibility='collapsed', placeholder='SNR')
                columns[2].write('Optional')
                columns[3].caption('New condition')
                if columns[4].form_submit_button('Add', icon=':material/add:'):
                    try:
                        if L is None or snr is None:
                            raise ValueError('Enter both a pilot length and SNR before adding a row.')
                        pair = add_condition(cfg, method, (L, snr))
                        st.session_state['pairs_' + prefix] = [pair]
                        st.session_state['scope_' + prefix] = 'Selected conditions'
                        if pair in additional_conditions(cfg):
                            st.session_state['additional_pair_' + prefix] = pair
                        reset_settings_forms(st)
                        st.rerun()
                    except (OSError, ValueError) as error:
                        st.error(str(error))
        if pairs:
            optional_input_controls(st, cfg, config, jobs, method, pairs)


def task_page(st, cfg, config, jobs, task, action):
    spec = ESTIMATORS.get(task)
    group = 'Analysis' if task in {'comparison','convergence'} else 'Algorithms' if spec else 'Data preparation'
    st.caption(f'{group} / {action.capitalize()}')
    description = task_actions(task)[action].description
    st.title(f'{action.capitalize()} {spec.name}' if spec else description[0].upper()+description[1:])
    from cgap_sbl_jadce.gui.setup_editor import edit_settings
    from cgap_sbl_jadce.gui.workflow import numerical_state
    state = numerical_state(cfg, task, action)
    if state.status == 'incompatible':
        st.warning('Saved results are incompatible with this setup.')
        with st.expander('Compatibility details'):
            st.write(state.detail)
    with st.expander('Task settings'):
        edit_settings(st, cfg, config, jobs, task, action)
    plan = None
    if spec and action == 'run':
        from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
        plan = completion_plan(config, task)
        st.progress(len(plan.completed) / max(1, len(plan.pairs)),
                    text=f'{len(plan.completed)}/{len(plan.pairs)} conditions completed')
        condition_status_panel(st, cfg, config, jobs, task, plan)
    conditions = condition = methods = q = None
    selectable = (task in ESTIMATORS or task == 'comparison') and action == 'run' or (
        action == 'tune' and tuning_settings(cfg, task)['selection_mode'] == 'per_condition')
    if selectable:
        pairs = (tuning_pairs(cfg, task, include_optional=True) if action == 'tune' else
                 list(dict.fromkeys(pair[:2] for pair in evaluation_conditions(cfg,
                     method=task if spec else None, include_optional=True))))
        scope = st.radio('Condition scope', ['All conditions', 'Selected conditions'], horizontal=True,
                         key=f'scope_{config}_{task}_{action}')
        if spec and action == 'run':
            st.caption('All conditions uses only the experiment sweeps. Select an additional condition explicitly to run it.')
        if scope == 'Selected conditions':
            pair_key = f'pairs_{config}_{task}_{action}'
            if pair_key in st.session_state:
                st.session_state[pair_key] = [pair for pair in st.session_state[pair_key] if pair in pairs]
            selected = st.multiselect('L / SNR conditions', pairs, default=None if pair_key in st.session_state else pairs[:1],
                                     format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB',
                                     key=f'pairs_{config}_{task}_{action}')
            conditions = [list(pair) for pair in selected]
        st.caption('Completion skips finished conditions. Rerun explicitly recalculates the selected conditions.')
    if task == 'convergence':
        pairs = list(dict.fromkeys(pair[:2] for pair in evaluation_conditions(cfg, include_optional=True)))
        reference = (cfg['evaluation']['pilot_length'], evaluation_settings(cfg)['snr_db'])
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
            count = len(conditions) if conditions is not None else (len(tuning_pairs(cfg, task)) if tuning_settings(cfg, task)['selection_mode'] == 'per_condition' else 1)
            candidates = len(spec.tuning_grid(cfg))
            st.caption(f'{count} conditions × {candidates} parameter candidates × {cfg["run"]["num_tuning_samples"]} tuning realizations before checkpoint reuse.')
        if spec.tuning_grid:
            st.caption('Tuning policy: separate selection per condition.' if tuning_settings(cfg, task)['selection_mode'] == 'per_condition'
                else f"Tuning policy: reference selection at L={tuning_settings(cfg, task)['reference_pilot_length']}, SNR={tuning_settings(cfg, task)['reference_snr_db']:g} dB.")
    reasons = (['Select at least one condition.'] if conditions == [] else
               blocking_reasons(cfg, task, action, conditions, methods, condition))
    for reason in reasons:
        st.info(reason)
    own = jobs.task_job(config, task, action, conditions=conditions)
    request = {'kind': 'task', 'task': task, 'action': action, 'config': str(config)}
    for key, value in [('conditions', conditions), ('methods', methods), ('realization_index', q), ('condition', condition)]:
        if value is not None:
            request[key] = value
    busy = jobs.busy() is not None
    try:
        if plan is not None:
            selected_plan = completion_plan(config, task, conditions) if conditions is not None else plan
            label = selected_plan.label(selected=conditions is not None)
            st.caption(selected_plan.summary)
            for (L, snr), reason in selected_plan.blocked.items():
                st.info(f'L={L}, SNR={snr:g} dB: {reason} Select this condition and Rerun to calculate it afresh.')
            controls = st.columns(3)
            fresh = label.startswith('Rerun')
            disabled = bool(reasons) or busy or not selected_plan.pairs
            if fresh:
                disabled |= bool(own and own['status'] in ACTIVE)
            else:
                disabled |= bool(selected_plan.blocked)
            if controls[0].button(label, key='task_action', type='primary', icon=':material/play_arrow:', disabled=disabled):
                if fresh:
                    jobs.start(request)
                elif (label == 'Resume' and own and own['status'] in RECOVERABLE | {'paused'} and
                      set(selected_plan.pairs) == {value[:2] for value in evaluation_conditions(cfg, own['request'].get('conditions'), method=task)}):
                    jobs.resume(own['id'])
                else:
                    jobs.complete(request)
                st.rerun()
            if label != 'Complete all':
                all_reasons = blocking_reasons(cfg, task, action)
                complete_disabled = busy or bool(all_reasons) or bool(plan.blocked) or len(plan.completed) == len(plan.pairs)
                if controls[1].button('Complete all', key='complete_all', icon=':material/done_all:',
                                      disabled=complete_disabled, help='Skip completed conditions, resume unfinished work and start untouched conditions.'):
                    jobs.complete({key: value for key, value in request.items() if key != 'conditions'})
                    st.rerun()
            if not fresh and (selected_plan.completed or selected_plan.resume or selected_plan.blocked):
                rerun_label = 'Rerun selected' if conditions is not None else 'Rerun all'
                if controls[2].button(rerun_label, key='rerun_conditions', icon=':material/restart_alt:',
                                      disabled=bool(reasons) or busy or bool(own and own['status'] in ACTIVE),
                                      help='Calculate every selected condition from the beginning, including completed ones.'):
                    jobs.start(request)
                    st.rerun()
        else:
            from cgap_sbl_jadce.gui.workflow import numerical_state
            status = numerical_state(cfg, task, action).status
            label = 'Rerun' if status in {'completed', 'partial', 'historical', 'incompatible'} else 'Start'
            recoverable = own and own['status'] in RECOVERABLE | {'paused'} and json.dumps(own['request'], sort_keys=True) == json.dumps(request, sort_keys=True)
            if recoverable:
                from cgap_sbl_jadce.recovery import validate_recovery
                try:
                    validate_recovery(own['request'])
                    label = 'Resume'
                except (OSError, ValueError, KeyError):
                    label = 'Rerun'
            disabled = bool(reasons) or busy or bool(own and own['status'] in ACTIVE and label != 'Resume')
            if st.button(label, key='task_action', type='primary', icon=':material/play_arrow:', disabled=disabled):
                jobs.resume(own['id']) if label == 'Resume' else jobs.start(request)
                st.rerun()
    except (OSError, ValueError) as error:
        st.error(str(error))
    st.caption('Pause takes effect at the next checkpoint. Once paused, another task can run. Return here to Resume. The same control recovers compatible checkpoints after a failure, stop, or interruption.')
    worker_panel(st, jobs, config, task, action, conditions)
    st.divider()
    figure_panel(st, cfg, config, jobs, task, action)
