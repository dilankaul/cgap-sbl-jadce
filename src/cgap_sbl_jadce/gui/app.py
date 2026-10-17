import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open the CGAP-SBL Streamlit dashboard.')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args(argv)
    try:
        import streamlit.web.cli as cli
    except ImportError:
        parser.exit(1, 'Install the project first: python -m pip install -e .\n')
    os.environ['CGAP_GUI_CONFIG'] = str(Path(args.config).resolve())
    sys.argv = ['streamlit', 'run', str(Path(__file__).resolve()), '--server.headless=true',
                '--browser.gatherUsageStats=false', '--server.address=127.0.0.1',
                '--theme.base=light', '--theme.primaryColor=#0f766e', '--theme.font=sans serif']
    return cli.main()


def dashboard():
    import streamlit as st
    from yaml import YAMLError
    from cgap_sbl_jadce.file_cache import validation_snapshot
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.gui.jobs import Jobs, ACTIVE
    from cgap_sbl_jadce.gui.setups import setup_files
    from cgap_sbl_jadce.gui.setup_editor import setup_editor, reset_settings_forms
    from cgap_sbl_jadce.gui.workflow import card_state, combined, refresh_results
    from cgap_sbl_jadce.gui.task_page import task_page
    from cgap_sbl_jadce.tasks import task_actions
    from cgap_sbl_jadce.gui.theme import apply_theme, icon, card_markup, section_heading

    st.set_page_config(page_title='CGAP-SBL Lab', page_icon=':material/science:', layout='wide')
    apply_theme(st)
    project = Path.cwd().resolve()
    @st.cache_resource
    def controller(project):
        return Jobs(project)
    jobs = controller(str(project))
    default = Path(os.environ.get('CGAP_GUI_CONFIG', 'configs/experiment.yaml')).resolve()
    files = setup_files(project, default)
    pending = st.session_state.pop('pending_setup', None)
    if pending is not None:
        st.session_state['setup_path'] = Path(pending)
    st.sidebar.markdown(f'<div class="lab-brand"><div class="lab-brand-mark">{icon("activity")}</div>'
                        '<div><strong>CGAP-SBL Lab</strong><small>Experiment workspace</small></div></div>',
                        unsafe_allow_html=True)
    page = st.session_state.get('page', 'overview')
    if st.session_state.get('setup_path') not in files:
        st.session_state['setup_path'] = default
    config = st.sidebar.selectbox('Experiment setup', files, key='setup_path',
        format_func=lambda value: load_config(value).get('experiment', {}).get('name', 'Default experiment'))
    try:
        cfg = load_config(config)
    except (OSError, ValueError, YAMLError) as error:
        st.error(str(error)); st.stop()

    editor_location = (str(config), page, st.session_state.get('current_task'), st.session_state.get('current_action'))
    if st.session_state.get('_editor_location') != editor_location:
        reset_settings_forms(st)
        st.session_state['_editor_location'] = editor_location

    def navigate(page, task=None, action='run'):
        reset_settings_forms(st)
        st.session_state.update(page=page, current_task=task, current_action=action)
        st.rerun()

    if st.sidebar.button('Workflow overview', icon=':material/dashboard:', type='primary' if page != 'setups' else 'secondary', width='stretch'):
        navigate('overview')
    if st.sidebar.button('Experiment setups', icon=':material/tune:', type='primary' if page == 'setups' else 'secondary', width='stretch'):
        navigate('setups')
    st.sidebar.markdown('<div class="sidebar-label">CURRENT ENVIRONMENT</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="setup-summary">' + ''.join(
        f'<div><strong>{cfg["system"][key]:g}</strong><span>{label}</span></div>'
        for key, label in [('N','Devices'),('M','Antennas'),('S','Target active')]) + '</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"{cfg['run']['num_tuning_samples']} tuning · {cfg['run']['num_evaluation_samples']} evaluation realizations")
    with st.sidebar.expander('Results location', icon=':material/folder_open:'):
        st.caption(str(cfg['run']['output_dir']))

    initial_render = True
    @st.fragment(run_every=2)
    def resident_tasks():
        records = jobs.records()
        refresh_results(st, cfg, records, rerun=not initial_render)
        active = [job for job in records if job['status'] in ACTIVE]
        if active:
            st.sidebar.divider()
            st.sidebar.markdown('<div class="sidebar-label">ACTIVE WORK</div>', unsafe_allow_html=True)
        for job in active:
            request = job['request']
            label = (task_actions(request['task'])[request['action']].description
                     if request['kind'] == 'task' else 'Publication export')
            if st.sidebar.button(f"{label} · {job['status']}", key=f"resident_{job['id']}", icon=':material/pause:' if job['status'] == 'paused' else ':material/play_circle:', width='stretch'):
                st.session_state['pending_setup'] = request['config']
                if request['kind'] == 'task':
                    navigate('task', request['task'], request['action'])
                else:
                    navigate('task', st.session_state.get('current_task', 'activity'))
    resident_tasks()
    initial_render = False

    def card(container, title, task, action='run', parent=False, key=None):
        state = (combined([card_state(cfg, config, task, jobs, part) for part in task_actions(task)])
                 if parent else card_state(cfg, config, task, jobs, action))
        with container.container(key='card_' + (key or task + '_' + action)):
            st.markdown(card_markup(title, task, state, action, compact=bool(key and key.startswith('algorithm_'))), unsafe_allow_html=True)
            if st.button(f"{title}\n{state.label}\n{state.summary}", key='open_' + (key or task+'_'+action), width='stretch'):
                navigate('algorithm' if parent else 'task', task, action)
            if state.status in {'running', 'queued', 'paused', 'partial'}:
                st.progress(state.fraction)

    page = st.session_state.get('page', 'overview')
    if page != 'overview':
        task = st.session_state.get('current_task')
        parent = 'overview'
        if page == 'task':
            if len(task_actions(task)) > 1:
                parent = 'algorithm'
            elif task in {'activity', 'kappa_calibration'}:
                parent = 'activity_model'
        if st.button('Back', key='go_back', icon=':material/arrow_back:', type='tertiary'):
            navigate(parent, task)
    if page == 'setups':
        st.markdown('<div class="eyebrow">ENVIRONMENTS</div>', unsafe_allow_html=True)
        st.title('Experiment setups')
        with st.expander('Current setup settings'):
            st.json({key: value for key, value in cfg.items() if key != 'task_runtime'})
        setup_editor(st, project, default, config, jobs)
    elif page == 'overview':
        st.markdown('<div class="eyebrow">RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
        st.title('Experiment workflow')
        @st.fragment(run_every=2)
        @validation_snapshot()
        def overview():
            section_heading(st, 'Data preparation')
            state = combined([card_state(cfg, config, task, jobs) for task in ('kappa_calibration', 'activity')])
            with st.container(key='card_activity_model'):
                st.markdown(card_markup('Activity model', 'activity', state), unsafe_allow_html=True)
                if st.button(f'Activity model\n{state.label}\n{state.summary}', key='open_activity_model', width='stretch'):
                    navigate('activity_model')
                if state.status in {'running', 'paused', 'partial'}:
                    st.progress(state.fraction)
            card(st, 'Communication dataset', 'communication')
            card(st, 'Spatial dependence matrix C', 'spatial_dependence')
            section_heading(st, 'Algorithms')
            from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
            cols = st.columns(len(ESTIMATORS))
            for col, (task, spec) in zip(cols, ESTIMATORS.items()):
                title = {'omp': 'OMP', 'cosamp': 'CoSaMP'}.get(task, spec.name)
                card(col, title, task, parent=bool(spec.tuning_grid), key='algorithm_'+task)
            section_heading(st, 'Analysis')
            cols = st.columns(3)
            card(cols[2], 'ANC parameter grid study', 'anc_grid_study')
            card(cols[0], 'Compare algorithms', 'comparison')
            card(cols[1], 'Analyze convergence', 'convergence')
        overview()
    elif page == 'activity_model':
        st.markdown('<div class="eyebrow">DATA PREPARATION / ACTIVITY</div>', unsafe_allow_html=True)
        st.title('Activity model')
        cols = st.columns(2)
        card(cols[0], 'Calibrate κ', 'kappa_calibration')
        card(cols[1], 'Generate activity', 'activity')
    elif page == 'algorithm':
        task = st.session_state['current_task']
        from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
        st.markdown('<div class="eyebrow">ALGORITHMS</div>', unsafe_allow_html=True)
        st.title(ESTIMATORS[task].name)
        cols = st.columns(2)
        card(cols[0], 'Tune', task, 'tune')
        card(cols[1], 'Run', task, 'run')
    else:
        task_page(st, cfg, config, jobs, st.session_state['current_task'], st.session_state['current_action'])


if __name__ == '__main__':
    dashboard()
