"""Task-local interactive charts and freshly rendered publication exports."""
from pathlib import Path
from cgap_sbl_jadce.gui.previews import (PREVIEWS, SAMPLE_PREVIEWS, sample_options,
                                       historical_results, missing_preview_inputs)
from cgap_sbl_jadce.gui.charts import interactive_figures, source_version
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, condition_config

CATEGORIES = {
    'kappa_calibration': ['kappa'], 'activity': ['activity', 'active_counts', 'paper_activity'],
    'communication': ['communication'], 'spatial_dependence': ['spatial'],
    'comparison': ['comparison'], 'convergence': ['convergence'],
}


def figure_panel(st, cfg, config, jobs, task, action):
    categories = CATEGORIES.get(task, [f'{task}_{action}'])
    from cgap_sbl_jadce.gui.theme import section_heading
    section_heading(st, 'Graph generator', 'Interactive views from your data')
    st.caption('Every chart is built from numerical data. Drag to pan, scroll to zoom, hover for values, and click legend entries to show or hide series.')
    from cgap_sbl_jadce.gui.workflow import numerical_state
    state = numerical_state(cfg, task, action)
    if state.status == 'incompatible':
        st.warning('Viewing stored results from their original settings.')
    category = st.selectbox('Figure type', categories, format_func=PREVIEWS.get)
    request = {'kind': 'preview', 'category': category, 'config': str(config)}
    ready = True
    plot_cfg = cfg
    if category.endswith('_tune'):
        from cgap_sbl_jadce.gui.previews import stored_tuning_options, tuning_preview_config
        method = category.removesuffix('_tune')
        options = stored_tuning_options(cfg, method)
        if options:
            layout, L, snr = st.selectbox('Saved tuning results', options,
                format_func=lambda item: f'{"Reference" if item[0] == "reference" else "Condition"} tuning: L={item[1]}, SNR={item[2]:g} dB')
            request.update(tuning_layout=layout, condition=[L, snr])
            plot_cfg = tuning_preview_config(cfg, method, request)
    missing = missing_preview_inputs(plot_cfg, category)
    if missing:
        st.info('Numerical data unavailable. Run the corresponding action first: ' + ', '.join(str(path) for path in missing))
        ready = False
    if category in SAMPLE_PREVIEWS:
        selectors = st.columns(3)
        split = selectors[0].selectbox('Split', ['evaluation', 'tuning'])
        try:
            count, pairs = sample_options(cfg, category, split)
            if count < 1:
                raise ValueError('No saved realizations in this split.')
            q = selectors[1].number_input('Realization index (zero-based)', min_value=0, max_value=count - 1,
                value=min(855, count - 1), step=1, key=f'index_{config}_{category}_{split}_{count}')
            request.update(split=split, index=int(q))
            if category == 'communication':
                pair = selectors[2].selectbox('Saved L / SNR', pairs, format_func=lambda pair: f'L={pair[0]}, SNR={pair[1]:g} dB')
                request['condition'] = list(pair)
            else:
                selectors[2].metric('Saved realizations', count)
        except (OSError, ValueError, KeyError) as error:
            st.info(f'Data unavailable: {error}')
            ready = False
    if category.endswith('_run'):
        from cgap_sbl_jadce.paths import result_path
        import json
        method = category.removesuffix('_run')
        summary_path = result_path(cfg['run']['output_dir'], method, 'run_summary.json')
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            pairs = list(dict.fromkeys(tuple(value[:2]) for value in summary.get('completed_conditions', [])))
            view = st.selectbox('Result condition', [None] + pairs,
                                format_func=lambda pair: 'All completed conditions' if pair is None else f'L={pair[0]}, SNR={pair[1]:g} dB')
            if view is not None:
                request['condition'] = list(view)
    if category.endswith('_run') or category == 'comparison':
        task_id = ESTIMATORS[category.removesuffix('_run')].task if category.endswith('_run') else 'comparison'
        if historical_results(cfg, task_id):
            st.warning('Historical output: these recorded results may use full-matrix NMSE. Rerun for current true-active-support NMSE.')
    if ready:
        try:
            figures = interactive_figures(request)
            labels = [label for label, _ in figures]
            if figures:
                selected = st.selectbox('Chart', labels, key=f'chart_{config}_{category}')
                figure = next(figure for label, figure in figures if label == selected)
                st.plotly_chart(figure, width='stretch', theme=None,
                                config={'scrollZoom': True, 'displaylogo': False}, key=f'plot_{category}')
            request['source_version'] = source_version(request)
        except (OSError, ValueError, KeyError) as error:
            st.info(f'Cannot plot these data yet: {error}')
            ready = False
    if st.button('Regenerate publication exports', icon=':material/file_download:', disabled=not ready or jobs.busy() is not None):
        try:
            record = jobs.start(request)
            st.session_state['export_job'] = record['id']
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    exported = next((item for item in jobs.records() if item['id'] == st.session_state.get('export_job')
                     and item['status'] == 'completed' and item['request'] == request), None)
    if exported:
        paths = [Path(path) for path in exported.get('paths', []) if path.endswith('.pdf') and Path(path).exists()]
        if paths:
            path = st.selectbox('Regenerated PDF', paths, format_func=lambda value: value.name)
            st.download_button('Download PDF', path.read_bytes(), icon=':material/download:', file_name=path.name, mime='application/pdf')
    st.caption('Interactive chart downloads use the Plotly toolbar. Publication exports retain the existing code-defined fonts and styles.')
