Saved experiment outputs are versioned in Git under `experiment/`.

Data preparation own calibration, event-driven activity, full communication data and
spatial dependence matrices. The algorithms own OMP, CoSaMP, SBL and ANC results;
Algorithm comparison owns comparison figures and Convergence analysis owns convergence. SBL/ANC selections
are saved in their respective `tuning/` subfolders.

Communication generation stores `channels.npz` (H, a, Z), `pilots.npz` (Theta per L), and
`observations.npz` (Y, W, sigma2 per condition) for each tuning/evaluation
realization. All algorithms read the same archives. Spatial dependence generation stores C as sparse CSR.
Activity, communication and spatial dependence tasks plot only configured indices in padded realization folders.

Older event-driven results have been split into the algorithm folders without
changing measured values. `status: historical` records require rerunning their
algorithm tasks before Algorithm comparison. Historical full-matrix NMSE is never labeled as
true-active-support NMSE. The existing event-driven inputs remain reusable.

A single root config YAML records the activity setup; task-local JSON manifests
record the effective task configuration and execution environment. New estimator
results record runtime and `nmse_scope: true_active_support`.

Algorithm evaluation data are separated under `<algorithm>/run/Lxxxx_snrXdB/`,
including per-realization scores, timing summaries, gamma data where applicable,
and condition figures. Algorithm-root tables combine compatible completed conditions.
Unfinished `.checkpoints/` and `.recovery/` data are tracked in Git and removed
after successful completion. Complete all/selected skips finished conditions and
recovers compatible unfinished work. Explicit Rerun recalculates selected conditions.

Optional evaluation conditions are shared in `additional_conditions.json` at the
setup result root, outside `config.yaml`. Their communication inputs are stored
under `communication/<split>/additional/Lxxxx_snrXdB/<index>/inputs.npz`, preserving
base archive bytes and shared H/a/Z. Each algorithm owns its optional condition
results under its usual run folder. Deleting an option removes it from all
algorithm tables while retaining existing data and results. Sweep completion
ignores optional rows.
