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
