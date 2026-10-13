from cgap_sbl_jadce.paths import result_path, figure_path, task_dir, tuning_figure_path


def test_artifacts_are_assigned_to_their_producing_tasks(tmp_path):
    for method in ('omp','cosamp','sbl','anc'):
        assert result_path(tmp_path, method, 'per_sample.csv') == task_dir(tmp_path, method) / 'per_sample.csv'
    assert task_dir(tmp_path, 'anc').name == 'cgap_sbl_anc'
    assert result_path(tmp_path, 'comparison', 'aggregate.csv').parent.name == 'comparison'
    for split in ('tuning','evaluation'):
        assert result_path(tmp_path, 'activity', f'{split}_activity.npz') == tmp_path / 'activity' / split / f'{split}_activity.npz'


def test_realization_figures_share_padded_folders(tmp_path):
    for task in ('activity','communication','spatial_dependence'):
        path = figure_path(tmp_path, task, 'matrix', 'evaluation', 714)
        assert path == task_dir(tmp_path, task) / 'figures' / 'evaluation_0714' / 'matrix'
    assert figure_path(tmp_path, 'communication', 'channel', 'tuning', 77).parent.name == 'tuning_0077'


def test_tuning_and_analysis_figures_have_descriptive_owners(tmp_path):
    cfg = {'run':{'output_dir':str(tmp_path)}, 'tuning':{'selection_mode':'reference'}}
    assert tuning_figure_path(cfg, 'anc', 'threshold_scores').parent == task_dir(tmp_path, 'anc') / 'figures' / 'tuning'
    assert figure_path(tmp_path, 'comparison', 'f1_vs_snr', section='evaluation').parent.name == 'evaluation'
