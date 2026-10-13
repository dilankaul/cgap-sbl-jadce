# CGAP-SBL JADCE experiments

Reproducible event-driven joint activity detection and channel estimation with
MMV-OMP, MMV-CoSaMP, SBL, and CGAP-SBL-ANC. This branch evaluates one event-driven
activity dataset. A future GL variant can reuse the independent estimator workflow.

## Installation

```bash
python -m pip install -e '.[dev]'
```

## Experiment dashboard

Launch from the repository root:

```bash
cgap-sbl-gui
```

Open `http://localhost:8501`; forward port 8501 in VS Code/Codespaces when working
remotely. `python -m cgap_sbl_jadce` opens the same GUI. The old terminal task
menu and `cgap-sbl` command have been removed.

The overview shows three preparation boxes, four algorithm boxes, and Compare
and Convergence boxes. Click a box to open its task page, with a completion
indicator and current progress on the overview. Activity opens Calibrate κ and
Generate activity; SBL and ANC open Tune and Run. Run requires compatible saved
tuning selections. Requirements follow data dependencies: OMP, CoSaMP, SBL and
ANC can run in any order once their inputs are ready. Spatial dependence can be
built before communication; SBL needs no spatial matrices.

Compare has a selector for algorithms completed at the requested conditions,
and can use even one algorithm. Convergence offers only completed SBL and ANC
runs, with a selected evaluation realization and L/SNR condition. OMP and CoSaMP
are excluded from convergence analysis.

Choose **All conditions** or **Selected conditions** to run an algorithm or compare
specific L/SNR pairs. Completed scores and measured runtimes accumulate without
duplication. The reference pair shared by two sweep plots is solved once and
uses the same recorded runtime in both views. Later runs reuse compatible completed units; partial evaluation does
not require finishing unrelated conditions. A setup can use the default fixed-value
SNR/pilot sweeps, or a full pilot-length × SNR grid.

### Experiment setups

In **Experiment setups**, choose **Default settings** (or copy the current setup),
edit the environment, sample counts, sweep ranges, solver settings, seeds and
figure selections, and create a named setup. New setups have isolated folders at
`results/<setup-name>/`, with one stable `config.yaml`. Task manifests record
effective parameters separately. The default settings and existing results stay
available through the sidebar setup selector. Changing an existing experiment is
handled by creating a new setup from it, which preserves earlier datasets/results.
There is no need to edit YAML or enter file paths in the GUI.

Choose either SBL/ANC tuning policy when creating a setup:

- **One reference selection:** tune at the reference L/SNR and reuse its selection
  across all selected evaluation conditions (the default).
- **Separate selections:** save tuning communication at each condition, select
  which conditions to tune, and evaluate with each condition's own saved selection.
  Selections and tuning tables are saved under `tuning/Lxxxx_snrXdB/`.

All tuning uses the independent tuning realizations. Evaluation never tunes on
held-out data. For separately tuned setups, only the requested conditions need
compatible selections before an evaluation run.

### Controls and figures

A detached worker shows progress, live logs and recent jobs. Pause waits for the
current completed calculation/checkpoint, keeping pause time out of solver
runtimes. Once paused, another task can run. Return to the original task to Continue; Stop interrupts the unfinished unit.
**Restart from checkpoints** reuses compatible work after failure. Browser/server
restarts reconnect to a live worker; after machine restart, recovery is manual.
Private worker records, logs and export files live in ignored `.gui/`.

The **Graph generator** inside each task page builds interactive Plotly charts from numerical arrays
and tables. It never displays saved PNG/PDF figures. Drag to pan, scroll to zoom,
hover for values and click legends to toggle traces. Activity/spatial views select
a split and realization; communication also selects an available L/SNR condition.
HSV matrices expose the actual magnitude, phase, real and imaginary values on
hover. NMSE uses logarithmic axes/colors; recorded historical scope is identified.

**Regenerate publication exports** renders fresh PNG/PDFs from the same data using
the existing code-defined Matplotlib typography. Paper activity exports omit the
title. Standard task PNG outputs remain available as reproducible artifacts;
the GUI visualizations use their underlying data directly.

### Task responsibilities

| Task | Responsibility |
|---|---|
| Kappa calibration | Calibrate kappa to the target mean active count |
| Activity | Save tuning/evaluation activity and selected figures |
| Communication | Save all communication inputs |
| Spatial dependence | Save each spatial dependence matrix C |
| OMP / CoSaMP | Evaluate independently |
| SBL | Tune tau, then Run evaluation |
| CGAP-SBL-ANC | Tune alpha/beta/tau, then Run evaluation |
| Compare | Summarize selected completed algorithms |
| Convergence | Record selected completed SBL/ANC iteration histories |


## Configuration and reproducibility

`configs/experiment.yaml` defines dimensions, activity, sweeps and tuning grids.
Kappa calibration scans ascending kappa values, brackets the target expected count, then
refines the bracket on the configured steps. Every candidate uses identical
geometries and Bernoulli uniforms. Independent validation reports uncertainty
and tolerance. Activity generation automatically reads the compatible saved calibration;
the source YAML is never rewritten.

The six distinct seeds separate kappa calibration, kappa validation, tuning
activity, evaluation activity, tuning communication and evaluation communication.
All estimators load identical a, H, Z, Theta, W and Y for a realization and condition.
H stays fixed across conditions; Theta stays fixed across SNRs at each pilot length.
Tuning and evaluation use separate data. Missing or incompatible input archives
stop a task with instructions to regenerate the producing stage.

The legacy default experiment retains its root `results/experiment/config.yaml`
snapshot. Named GUI setups use that file as their stable input definition; tasks
never overwrite it. Each task has a
JSON manifest with its effective configuration and runtime metadata. Font sizes
and figure dimensions are defined locally in plotting functions, not YAML.

## Algorithm tuning and metrics

SBL > Tune runs SBL on tuning realizations and selects its own tau. ANC > Tune runs
the configured alpha/beta grid, selects tau separately for each pair, then ranks
pairs by mean F1, lower NMSE and higher recall. Both use tau=0 through 0.2 inclusive
in steps of 0.001 (201 candidates). ANC alpha=0 recovers conventional SBL; SBL's
selection remains independent of the ANC grid. Each Tune action saves its selection under
its own `tuning/selected.json`. Run reads and validates that selection before
evaluating held-out realizations; missing or incompatible selections require Tune.
Tuning and evaluation figures are generated by their respective actions.

OMP and CoSaMP use the feasible oracle sparsity budget B_q=min(S_q,L,N).
Their raw rows record S_q, B_q and whether capping occurs. SBL and ANC receive
neither the true support nor the oracle sparsity budget.

NMSE uses the true active rows in both numerator and reference-energy denominator:

`NMSE_q = sum_{i:a_i=1} ||z_i-z_hat_i||_2^2 / sum_{i:a_i=1} ||z_i||_2^2`.

Missed active devices contribute error; estimates on inactive rows are excluded.
The full posterior estimate is retained for SBL methods. Zero-reference-energy
NMSE is undefined and excluded from its average; detection scores still include
those realizations. The true activity mask is used only for scoring.

Each algorithm task records solver-call wall time with `time.perf_counter`,
excluding tuning, loading, metrics, plots and saving. Raw scores, aggregate scores,
and runtime statistics belong to that algorithm's folder. Algorithm comparison checks source
identity, setup, selected parameters, completeness and metric scope before comparison.
NMSE line plots and convergence use logarithmic y-axes; the ANC NMSE heatmap uses
logarithmic color scaling. See `docs/computational_complexity.md` for the complexity audit.

## Saved outputs

```text
results/experiment/
  config.yaml
  kappa_calibration/
  activity/
    tuning/tuning_activity.npz
    evaluation/evaluation_activity.npz
    figures/tuning_0051/
    figures/evaluation_0855/
  communication/
    tuning/0000/             # channels.npz, pilots.npz, observations.npz
    evaluation/0000/         # same layout, all evaluation conditions
  spatial_dependence/
    tuning/0000.npz          # compressed sparse CSR C
    evaluation/0000.npz
  omp/
  cosamp/
  sbl/
    tuning/                 # threshold curve and selection
  cgap_sbl_anc/
    tuning/                 # alpha/beta/tau grid, threshold curve and selection
  comparison/
  convergence/
```

Each algorithm folder contains `per_sample.csv`, `aggregate.csv`,
`runtime_summary.csv`, `run_summary.json`, and its own figures. SBL and ANC also
save gamma-distribution data. Saved numerical outputs and figures are versioned.
Migrated older scores are explicitly marked historical and must be rerun before
comparison; their NMSE values are not converted or relabeled.

Communication generation stores H, a and Z once, Theta once per L, and Y, W and sigma2 per unique
condition. Overlapping sweep conditions share inputs. Spatial dependence generation saves every C;
ANC loads those matrices and never constructs them during evaluation.

## Restart after interruption

Restart the same GUI action with the same setup and condition selection. New
runs automatically save private checkpoints under the producing task's
`.checkpoints/` directories. Finished work is reused after a crash or process
restart; the unfinished calculation unit is repeated.

Recovery boundaries are calibration batches, activity/communication/spatial
realizations, individual tuning solves, individual evaluation conditions, and
complete convergence traces. Calibration restores its RNG state and running
sums; activity generation preserves each realization's original seed sequence.
Tuning saves gamma outputs before moving on, and evaluation saves each score
and measured solver runtime before starting the next condition. Resume does
not duplicate rows or rerun completed solvers. Aggregate outputs and figures
can be rebuilt from that committed numerical work.

Checkpoints are isolated by settings, input data identities, scientific code,
numerical library versions and thread settings. Changed settings use a different
checkpoint set. Generated communication and spatial files are checked against
saved hashes before reuse; missing or changed files are rebuilt. Unreadable
checkpoint units are recomputed. Old results produced before this feature are
not retroactively checkpointed.

JSON, YAML, CSV, NPZ, sparse matrices and figures are written through temporary
files, flushed, and atomically replaced. A failure before replacement leaves
the previous complete file intact. Checkpoints and unfinished temporary files
are excluded from Git; final experiment outputs remain versioned. Keep the
results directory to retain restart progress.

## Figures and publication PDFs

Configure zero-based `figures.tuning_indices` and `figures.evaluation_indices`.
A list selects realizations, `all` selects every realization, and `[]` skips that
split's maps. Activity generation still saves the full datasets and active-count histogram.
Activity, communication and spatial dependence tasks save selected figures in padded folders such as `evaluation_0855`.
Communication generation exports HSV H, Theta, standardized W/sigma, Z and Y, effective-channel
row energy and pilot cross-correlation: seven figures per selected realization.
Worker reports show the figure folder and compact counts.

The plotted activation field is the actual model probability
`P(x)=1-product_v(1-P_v(x))`. Values are never thresholded, shifted or changed for
appearance. Only the region outside the physical cell is masked. The colormap
maps zero to white and increasing probability to red.

Normal tasks save PNG. Export one title-free publication PDF from saved Activity generation data:

```bash
python scripts/paper_figures/activity_realization.py --split evaluation --index 855
```

The filename is `event_driven_activation_model_evaluation_0855.pdf` under
`results/experiment/paper_figures/activity/evaluation/`. No separate paper YAML
or LaTeX installation is required. Edit `figure_style` and `figsize` inside
`figures/paper_activity_realizations.py` for publication-specific typography.
Native serif and Computer Modern math fonts are used with embedded TrueType PDF
fonts; bold Greek labels use local STIX math rendering.

## Code structure and validation

The GUI selects independent actions from the shared `tasks.py` registry. Data preparation lives in
`experiments/pipeline.py`; the estimator registry and shared tuning, scoring,
comparison and convergence workflow live in `experiments/estimators.py`.
Algorithm mathematics remains in `algorithms/`, data models in `models/`, and
plotting in `figures/`. A later GL method can supply its solver and tuning grid
through the same estimator interface, with a dedicated task and result folder.

```bash
PYTHONPATH=src pytest -q
```

See `VALIDATION.md` and `docs/theory.md` for validation scope and mathematical details.
