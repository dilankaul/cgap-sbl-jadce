from cgap_sbl_jadce.paths import result_path, figure_path, task_dir


def test_artifacts_are_assigned_to_their_producing_tasks(tmp_path):
    assert result_path(tmp_path, "tuning", "selected_alpha_beta.json") == task_dir(tmp_path, 5) / "selected_alpha_beta.json"
    assert result_path(tmp_path, "tuning", "selected.json") == task_dir(tmp_path, 6) / "selected.json"
    assert result_path(tmp_path, "evaluation", "per_sample.csv") == task_dir(tmp_path, 7) / "per_sample.csv"
    assert result_path(tmp_path, "evaluation", "aggregate.csv") == task_dir(tmp_path, 9) / "aggregate.csv"
    for split in ("tuning", "evaluation"):
        assert result_path(tmp_path, "activity", f"{split}_activity.npz") == task_dir(tmp_path, 2) / split / f"{split}_activity.npz"
        stem = f"02_activity_realization_{split}_0000_probability_field"
        assert figure_path(tmp_path, stem) == task_dir(tmp_path, 2) / "figures" / f"{split}_0000" / stem
    assert figure_path(tmp_path, "06_cgap_threshold").parent == task_dir(tmp_path, 6) / "figures" / "tuning"
    assert figure_path(tmp_path, "09_f1_vs_snr").parent == task_dir(tmp_path, 9) / "figures" / "evaluation"



def test_realization_figures_share_padded_folders(tmp_path):
    for stem in ['02_activity_realization_evaluation_0714_probability_field',
                 '02_activity_realization_randomized_evaluation_0714']:
        assert figure_path(tmp_path, stem).parent == task_dir(tmp_path, 2) / 'figures' / 'evaluation_0714'
    assert figure_path(tmp_path, '03_tuning_channel_hsv_heatmap_0077').parent.name == 'tuning_0077'
    assert figure_path(tmp_path, '03_evaluation_effective_channel_matrix_hsv_heatmap_0714').parent.name == 'evaluation_0714'



def test_spatial_dependence_figures_use_selected_realization_folders(tmp_path):
    for split, q in [('tuning', 77), ('evaluation', 714)]:
        stem = f'04_{split}_spatial_dependence_matrix_{q:04d}'
        assert figure_path(tmp_path, stem).parent == task_dir(tmp_path, 4) / 'figures' / f'{split}_{q:04d}'
