# CGAP-SBL for JADCE in MTC — Reproducible Implementation

Codespaces-ready implementation for the paper **Correlation-Guided Adaptive Penalized SBL for Joint Activity Detection and Channel Estimation in MTC**.

The scientific workflow contains **nine uniquely numbered tasks (01–09)**. Standard figures are saved automatically by the numbered task that produces the corresponding data. Numbered tasks save PNG previews for the repository; the paper activity script saves vector PDF.

`AGENTS.md` contains persistent instructions for Codex, while `DECISIONS.md` records locked scientific and implementation decisions.

## Implemented CGAP-SBL theory

The lagged nonnegative penalty is `lambda = positive_part(Omega @ gamma)`.
The complex-MMV second moment is `eta_i = Sigma_ii + ||mu_i||²/M`, and the update is
`gamma_i = 2*eta_i / (1 + sqrt(1 + 4*lambda_i*eta_i/M))`.
Zero penalty recovers conventional SBL exactly. Both methods use relative gamma
change below `delta=1e-4` or at most `K=500` iterations.

The ANC version uses `Omega_ANC = alpha*(beta*1-C)` and computes penalties
without constructing dense Omega. Its spatial parameters match the saved
activity model: `rho=kappa=3.78`, `U=D=15 m` for the current experiment.
The interaction is an adaptive penalization design, not a proper Gaussian
hyperprior or a guarantee of monotonic global-objective improvement.
See [docs/theory.md](docs/theory.md) for equations and interpretation.

## Revised-paper experiment

Tuning uses 100 independent realizations at `L=25`, SNR 12 dB, with
`alpha=[2,4,6,8,10,12,14]` and `beta=[0.05,0.10,...,0.35]`.
Evaluation uses 1,000 realizations: pilot lengths 20–45 in steps of 5 at
12 dB, and SNR 4–24 dB in steps of 4 at `L=30`. Convergence uses `L=30`, 12 dB.
Task 05 saves both F1 and NMSE contours.

Task 02 saves paired spatially randomized evaluation activity; Task 07 loads
these supports for CGAP-SBL-ANC and SBL,
retaining each realization's active count, geometry, channels, and pilots.
Parameters tuned on correlated activity remain fixed. Task 09 saves four-curve
spatial comparisons and `spatial_randomization_advantages.csv`, reporting the
within-scenario F1 gain and NMSE reduction relative to SBL.

Existing event-driven Task-01/02 data remain compatible. Rerun Task 02 once
to create its new paired randomized evaluation dataset and figures. Rerun Tasks 03–09 for the revised
schema and method; old tuning is explicitly rejected. Historical saved outputs
are preserved and are not evidence of CGAP-SBL performance. Provisional numerical
claims in the manuscript are not treated as generated experiment results.
The Python import namespace is `cgap_sbl_jadce`; the implemented methods
are `cgap_sbl` and `cgap_sbl_anc`, and the distribution is `cgap-sbl-jadce`.

## Setup

Verify the environment with:

```bash
PYTHONPATH=src pytest -q
```

Use `configs/experiment.yaml` for the experiment workflow. The activity model uses `D=15 m`; Task 01 automatically selects kappa before Task 02. `system.S` is the **target mean activity level**, not a hard per-realization sparsity constraint.

## Numbered scientific workflow

Each numbered task automatically saves its standard figures as PNG. Paper activity figures are exported separately as vector PDF.

### Task 01 — Calibrate kappa

```bash
python scripts/01_calibrate_kappa.py
```

This saves the calibration CSV/JSON and the \(\kappa\)-calibration figure. It does not overwrite Task-02 activity datasets.

### Task 02 — Generate activity realizations

```bash
python scripts/02_generate_activity.py \
  --config configs/experiment.yaml
```

The event process is unconditioned, so the realized activity count \(S_q\) varies. This task saves tuning/evaluation activity datasets, `realized_s.csv`, summary statistics, every event-driven tuning/evaluation activity realization with the activation-probability field, randomized evaluation maps without event overlays, and the realized-\(S\) distribution.

### Task 03 — Generate the communication-model preview

```bash
python scripts/03_generate_communication.py --config configs/experiment.yaml
```

The model uses

\[
\mathbf Z=\operatorname{diag}(\mathbf a)\mathbf H^T,
\qquad
\mathbf Y=\mathbf\Theta\mathbf Z+\mathbf W,
\]

with unit-norm QPSK pilots and deterministic condition-specific random streams. The pilot cross-correlation figure is saved automatically.

### Task 04 — Build the spatial correlation matrix

```bash
python scripts/04_build_correlation.py --config configs/experiment.yaml
```

The correlation summary and \(\mathbf C\) visualization are saved automatically.

### Task 05 — Tune \(\alpha,\beta\)

```bash
python scripts/05_tune_alpha_beta.py
```

Task 05 uses an internally optimized provisional threshold only for ranking the \((\alpha,\beta)\) grid. The selected grid result and tuning figure are saved automatically. Final CGAP-SBL and SBL activity thresholds are independently selected in Task 06.

### Task 06 — Tune activity thresholds

```bash
python scripts/06_tune_thresholds.py --config configs/experiment.yaml
```

This saves `cgap_thresholds.csv`, `sbl_thresholds.csv`, `selected.json`, and the CGAP-SBL/SBL threshold-tuning figures.

### Task 07 — Run all estimators

```bash
python scripts/07_run_estimators.py
```

This runs CGAP-SBL-ANC, conventional SBL, MMV-OMP/SOMP, and MMV-CoSaMP on identical communication realizations and automatically saves the gamma-distribution diagnostics.

For each realization, the greedy baselines use

\[
B_q=\min(S_q,L,N),
\]

where \(S_q=\sum_i a_i\). Thus `B_q == S_q` whenever feasible. If `S_q > L`, the greedy support budget is capped at `L`. Raw baseline rows record `S_q`, `B_q`, and `oracle_capped`; `run_summary.json` reports how often capping occurs. CGAP-SBL and SBL are **not** given \(S_q\) or \(B_q\).

### Task 08 — Run convergence analysis

```bash
python scripts/08_run_convergence.py --config configs/experiment.yaml
```

The convergence data and convergence figure are saved automatically.

### Task 09 — Aggregate, evaluate, and generate performance figures

```bash
python scripts/09_evaluate.py --config configs/experiment.yaml
```

This task does not rerun estimators. It aggregates the raw Task-07 results and automatically saves the standard F1, NMSE, and runtime performance figures.

## Automatic task figures

Run the numbered tasks normally; each saves its PNG figures automatically.
All task scripts default to `configs/experiment.yaml`. The only CLI options
are `--config` and `--no-progress` (plus standard `--help`); the paper export
additionally requires `--split` and `--index`. All scientific
settings come from the experiment YAML. Run tasks in order for a
new experiment; Task 02 reads Task 01's calibrated kappa. For title-free paper
activity PDFs, use `scripts/paper_figures/activity_realization.py` after Task 02.

## Activity-probability figure

The displayed field is the actual event-model probability

\[
P(\mathbf x)=1-\prod_v\left[1-P_v(\mathbf x)\right].
\]

The probability data are never thresholded, epsilon-shifted, or modified to make the figure white. The only mask is outside the physical circular BS cell. A custom colormap maps `P=0` to pure white and increasing probability toward dark red. Publication PDFs use vector `contourf` with levels spanning exactly `[0,1]`.

## Baselines

- **SBL:** proper complex-MMV SBL using the optimized posterior engine.
- **MMV-OMP / SOMP:** joint row-support greedy baseline using oracle budget \(B_q\).
- **MMV-CoSaMP:** joint row-support CoSaMP using unregularized SVD least squares and the same oracle budget \(B_q\).

The greedy methods should be described in the paper as **oracle sparsity-budget baselines**, not as methods always receiving the uncapped realized sparsity.

## Reproducibility

- Tuning and evaluation activity streams are deterministic and separate.
- Activity and communication random streams are separate.
- All algorithms receive identical \(\mathbf a,\mathbf H,\mathbf Z,\mathbf\Theta,\mathbf W,\mathbf Y\) for a given condition.
- \(\mathbf H\) is fixed across pilot lengths and SNRs for a realization.
- \(\mathbf\Theta\) is fixed across SNR values for a given realization and pilot length.
- Thresholds are selected only from tuning data.
- CGAP-SBL \(\alpha,\beta\) and both thresholds must be retuned for the revised setup.
- Large communication tensors are deterministically regenerated rather than stored for every Monte-Carlo condition.

## Validation

Run:

```bash
PYTHONPATH=src pytest -q
```

See `VALIDATION.md` for the validation scope.

### Native PDF fonts

Paper figures use PDF with Matplotlib's native renderer; no LaTeX
installation is required. Serif fonts are selected in order: Computer Modern
Roman, CMU Serif, STIXGeneral, DejaVu Serif. Math uses Computer Modern.
PDF output embeds TrueType fonts.

```bash
python scripts/paper_figures/activity_realization.py --split tuning --index 0
```

### Figure measurements

Each plotting function in `src/cgap_sbl_jadce/figures/` defines its own `figsize`
and `figure_style` dictionary. Edit those values to change title, axis, tick,
legend, or colorbar font sizes for that figure. Activity realization labels remain
15 pt and ticks/legends 13 pt. Experiment YAML does not contain font overrides.

### Experiment outputs and all activity realizations

`configs/experiment.yaml` is the full experiment configuration and writes to
`results/experiment/`. Saved numerical results, summaries, and PNG/PDF figures
are eligible for version control. Previously saved config/manifest files retain
their original paths as historical run provenance.

Task 02 automatically exports every tuning and evaluation activity realization
in PNG, with the split, zero-based realization number, and realized active
MTD count in its title. The full configuration produces 100 tuning, 1,000 event-driven evaluation, and 1,000 randomized
evaluation spatial PNG figures. Batch export shows progress.

```bash
PYTHONPATH=src python scripts/02_generate_activity.py --config configs/experiment.yaml
```

`plot_activity_sample()` controls dimensions and font sizes for all spatial realizations.
Figures are saved as part of the normal task run.

### Task output folders

Each task owns its data and figures under `results/experiment/`:

```text
01_kappa_calibration/       # calibration CSV and summary; figures/
02_activity_realizations/      # summary and realized_s.csv
  tuning/                     # tuning_activity.npz
  evaluation/                 # evaluation_activity.npz
  randomized_evaluation/      # randomized_evaluation_activity.npz
  figures/
    tuning/                   # one PNG per tuning realization
    evaluation/               # one PNG per event-driven evaluation realization
    randomized_evaluation/    # one PNG per randomized evaluation realization
                              # distribution PNG is directly in figures/
03_communication_preview/      # preview and summaries; figures/
04_spatial_correlation/        # correlation data and summary; figures/
05_alpha_beta_tuning/          # grid and selection; figures/tuning/
06_threshold_tuning/           # thresholds and selection; figures/tuning/
07_estimators/                 # per-sample results and gamma; figures/evaluation/
08_convergence/                # traces and summary; figures/evaluation/
09_evaluation/                 # aggregated results and summary; figures/evaluation/
```

Run-level `config.yaml` remains at the results root. Each task stores its own
`manifest.json`. Existing saved outputs have been
moved without changing their numerical data or historical run metadata.

### Automatic calibrated kappa

The experiment configuration sets `activity.kappa_source: calibration`. Run Task 01,
then Task 02; Task 02 reads `closest_grid_kappa` from `01_kappa_calibration/summary.json`.
No manual YAML update is needed. Missing calibration or mismatched N, R, V, D,
target activity, or calibration seed stops generation and asks you to rerun Task 01.
Task 02 prints its selected kappa and records its source in the activity summary;
the saved run config contains the effective value. Input YAML is unchanged.
The experiment config uses automatic calibration and omits a fixed `activity.kappa`.
Run Task 01 before Task 02. Explicit manual mode remains available
by setting `kappa_source: configured` and adding `activity.kappa`.

### Coarse-to-fine kappa calibration

Task 01 scans ascending `activity.kappa_scan` values, finds a bracket around the
requested mean activity, then refines using `kappa_refinement_steps` (defaults:
0.1, 0.02, 0.005). The selected value comes from the final aligned grid. Every
stage replays the same seed, sample count, and batch size to compare identical
geometries and Bernoulli uniforms. Set the coarse candidates in the experiment YAML.

`kappa_sweep.csv` records all unique candidates, their first evaluation stage,
and expected/realized activity statistics. The summary records brackets,
selection, final spacing, and validation on an independent deterministic seed
(`seeds.kappa_validation`).
Validation reports expected-mean standard error, an approximate 95% interval,
and whether the mean is within `calibration_tolerance` (default 0.1 device).
Validation does not retune the selected value. A tolerance miss is reported;
it does not silently redefine the target. Calibration and validation remain
separate from Task 02 datasets. A missing bracket saves a failed summary and
stops; Task 02 rejects that summary instead of consuming an old selection.

The experiment requires its own Task-01 results. Paper exports use the saved Task-02 model parameters.

### Configuration provenance

`configs/experiment.yaml` is the input. Results contain only one run-level
`config.yaml`: Task 01 creates it when absent, and Task 02 updates it with the
activity-generation configuration. Later tasks leave this root snapshot unchanged.
Each task's `manifest.json` stores its exact effective configuration under
`configuration`, alongside runtime options and environment versions. This preserves
calibration overrides, selected kappa, thresholds, and seeds without extra YAML
files. Existing task YAML snapshots were migrated into JSON manifests before removal.
Source configuration is never rewritten.

### Code organization

The nine numbered scripts are thin, import-safe wrappers around
`task_cli.run_task(task_number)`. Shared CLI handling loads the experiment
configuration, executes one stage, and saves PNG figures. The paper activity
script reads saved Task 02 data and exports separate PDFs.

```text
src/cgap_sbl_jadce/
├── models/          # Activity, communication, realizations, correlation
├── algorithms/      # SBL, CGAP-SBL, OMP, CoSaMP, posterior engine
├── experiments/     # Calibration, threshold tuning, nine-stage pipeline
├── figures/         # Descriptive task-numbered plots and shared rendering
├── task_cli.py      # Registry and common --config / --no-progress arguments
├── config.py
├── dataset.py
├── io.py
├── metrics.py
├── paths.py
├── provenance.py
└── reporting.py
```

Figure filenames identify the producing task and the plotted quantity:

```text
figures/
├── common.py
├── task01_kappa_calibration.py
├── task02_activity_realizations.py
├── task02_active_mtd_count_distribution.py
├── task03_communication_preview.py
├── task04_spatial_correlation.py
├── task05_alpha_beta_tuning.py
├── task06_activity_threshold_tuning.py
├── task07_gamma_distribution.py
├── task08_convergence.py
├── task09_performance.py
└── paper_activity_realizations.py
```

Shared typography and saving live in `figures/common.py`. Figure dimensions and
font sizes stay local to their plotting functions. Paper styling lives in
`paper_activity_realizations.py` and reuses Task 02's drawing function.
`experiments/pipeline.py` coordinates all nine scientific stages.

### Activity figure splits

Task 02 generates both activity datasets and exports both sets of PNG figures by
default. No separate figure selection switches are required.
Known fontTools timestamp messages are filtered during figure saving; other
warnings remain visible.

### Explicit random seeds

The `seeds` section names seven independent streams: `kappa_calibration` selects
the decay parameter that gives the target mean active MTD count;
`kappa_validation` checks that selection on independent samples.
`activity_tuning` and `activity_evaluation` generate the two activity datasets.
`communication_tuning` and `communication_evaluation` generate their channels,
pilots, and noise. All seven seeds are explicit and distinct; no seed offsets
are applied between these streams. `spatial_randomization=6031` generates paired
randomized supports independently of the six retained streams. Algorithms still share realizations within
each condition.

The streams use distinct four-digit seeds listed in `configs/experiment.yaml`.
The six existing activity and communication streams are preserved; only the
spatial-randomization stream is new. Existing saved activity remains compatible.

### Paper activity figures

Task 02 continues saving all standard figures with titles and its default sizes.
Export selected saved realizations as separate, title-free publication PDFs:

```bash
python scripts/paper_figures/activity_realization.py --split tuning --index 0
```

This uses the existing experiment configuration and saved Task 02 data. No
separate paper YAML is needed. Edit `figure_style` and `figsize` inside
`plot_paper_activity_figure` in
`src/cgap_sbl_jadce/figures/paper_activity_realizations.py` to adjust font
sizes in points and dimensions in inches, just like the task plotting functions.
Each run exports exactly one saved realization. Choose `--split tuning` or
`--split evaluation` and a zero-based `--index`. PDFs are written to
`results/experiment/paper_figures/activity/`, with tuning/evaluation subfolders.
The script does not run scientific tasks or write configuration snapshots.
Both export paths reuse the same saved-model and probability-field settings.
Their font sizes remain separate local definitions. The experiment config omits
`max_attempts` because variable-activity sampling does not use rejection retries;
exact-sparsity experiments can still set it explicitly.


Paper activity PDFs use `event_driven_activation_model_[split]_[index].pdf` with four-digit index padding,
for example `event_driven_activation_model_evaluation_0005.pdf`.


### Saved randomized evaluation activity

Task 02 saves `02_activity_realizations/randomized_evaluation/randomized_evaluation_activity.npz`
and automatically draws every map under `02_activity_realizations/figures/randomized_evaluation/`.
Each saved sample retains its paired device/event geometry and active count.
Randomized support uses the same seed/index sequence as the former Task-07 generator.
Its stored marginal activation probabilities are S_q/N conditional on count;
selection is without replacement, not independent Bernoulli activation.
Randomized maps omit events, event boundaries, and the event probability field.
Task 07 requires these saved pairs and never generates new activity supports.
Communication observations are still generated in memory for each condition.
