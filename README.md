# CGAP-SBL JADCE experiments

Reproducible event-driven joint activity detection and channel estimation with
MMV-OMP, MMV-CoSaMP, SBL, and CGAP-SBL-ANC. This branch evaluates one event-driven
activity dataset. A future GL variant can reuse the independent estimator workflow.

## Installation

```bash
python -m pip install -e '.[dev]'
```

### Windows with VS Code and Docker

Install Docker Desktop with its WSL 2 backend, and the Dev Containers extension
in VS Code. Keep Docker Desktop running, open the repository folder in VS Code,
and select **Dev Containers: Reopen in Container** from the Command Palette.

The existing container configuration installs Python 3.11, project dependencies
and development tools, and sets the numerical-library thread limits. Once setup
finishes, run `cgap-sbl-lab` in the container terminal. VS Code forwards port 8501
and opens the dashboard in your browser. Results stay in the repository's
`results/` directory across container rebuilds.

## Experiment dashboard

Launch from the repository root:

```bash
cgap-sbl-lab
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
uses the same recorded runtime in both views. **Complete all** skips finished
conditions, resumes compatible unfinished work and starts untouched conditions.
**Complete selected** does the same for the chosen subset. The primary action
shows **Start** for untouched work, **Resume** for unfinished work, or
**Rerun selected/all** when the chosen conditions are already finished. Explicit
reruns recalculate the chosen conditions from the beginning. Partial evaluation
does not require finishing unrelated conditions. Experiment setups define fixed-reference SNR and pilot-length sweeps.
Additional conditions are optional and shared across algorithm Run pages.

### Experiment setups

In **Experiment setups**, edit the current setup or create one from default/current
settings. Settings are grouped by environment, activity, kappa calibration,
communication, evaluation conditions and individual algorithms. ANC owns its
alpha/beta/tau ranges; SBL owns its threshold range. Alpha, beta, tau, the
initial kappa scan, and evaluation pilot-length/SNR sweeps use inclusive
**Start / Stop / Step** controls. Kappa bracket-refinement steps remain separate. Each algorithm has its own
solver settings, reference condition and tuning policy. New setups have isolated
folders at `results/<setup-name>/`, with one editable `config.yaml`. Task manifests
record the settings that produced each result.

Each task also exposes its corresponding **Task settings**, using the same form
and source configuration. Click **Save changes** to apply edits, or **Save setup**
to create a setup. Enter does not submit these forms. **Reset** restores the saved
values (or the selected defaults for a new setup); leaving a page discards its
unsaved draft. Saving updates settings atomically without deleting results or checkpoints. Incompatible
results show an indicator and cannot feed downstream tasks. Restoring matching
values restores compatibility if the original input data and results remain.
Adding conditions preserves compatible completed conditions. Changing figure
selections does not invalidate scientific results. Pause an active task before
editing settings it uses; unrelated settings remain editable. Resume requires its
original relevant settings, inputs, code and numerical environment. YAML formatting
and unrelated edits do not affect recovery.

Evaluation settings separately specify **SNR for the pilot-length sweep** and
**Pilot length for the SNR sweep**. ANC has its own **Alpha-beta tuning pilot length L**
and **Alpha-beta tuning SNR (dB)**; these do not change either sweep's fixed value.
SBL likewise owns its threshold-tuning reference. With per-condition tuning,
each selected condition is tuned independently.

On an algorithm's **Tune** or **Run** page, open **Condition status**, enter L/SNR in the empty
table row and click **Add**. Each saved optional row has a **Delete** button;
these controls apply changes directly without an additional Save step. **Prepare inputs** saves deterministic
communication inputs without altering base archives or running an estimator.
The fields start blank. Adding or deleting a condition updates every algorithm’s
condition table, which labels it Optional. Then select the added pair explicitly to run it. **All conditions** and
**Complete all** cover only setup sweeps; optional results do not affect sweep
completion. Additional input archives live under
`communication/<split>/additional/Lxxxx_snrXdB/<index>/inputs.npz` and reuse saved
shared channels/activity. Condition-specific SBL/ANC policies require tuning
the added pair on their **Tune** page first; reference policies reuse the reference
selection. Completed optional conditions are available for comparison and SBL/ANC convergence.

The Communication page uses the same condition table, shared Add/Delete controls,
All/Selected scope, and Start/Complete/Resume/Rerun actions as algorithm Run pages.
Each condition has one status: Completed means both tuning and evaluation inputs
are saved. Complete skips completed inputs, resumes compatible unfinished work,
and starts missing conditions without changing completed algorithm results.
Optional conditions must be selected explicitly and remain outside All/Complete all.

Both communication splits cover every setup sweep condition and any algorithm
tuning or preview references outside those sweeps. They use separate activity and
communication seeds. Preparing an optional condition from either page also saves
both datasets; algorithm tuning policies and Run prerequisites remain independent.
Selected conditions can start from an empty communication dataset once activity
is available. Each realization is checkpointed, and scoped PNG figures can be
regenerated from the saved arrays.

Choose either tuning policy independently for SBL and ANC:

- **One reference selection:** tune at the reference L/SNR and reuse its selection
  across all selected evaluation conditions.
- **Separate selections (default):** tune SBL tau and ANC alpha/beta/tau
  independently for each selected L/SNR condition, and evaluate with that
  condition's own saved selection.
  Selections and tuning tables are saved under `tuning/Lxxxx_snrXdB/`.
  Tune pages show a Condition status table for every setup and shared optional
  pair, with Completed, Unfinished, Incompatible, or Not started status and
  an overall tuning count. Tune, Run and Communication tables share the same
  Add/Delete controls and optional-input preparation.

All tuning uses the independent tuning realizations. Evaluation never tunes on
held-out data. For separately tuned setups, only the requested conditions need
compatible selections before an evaluation run.

### Controls and figures

A detached worker shows progress, live logs and recent jobs. Pause waits for the
current completed calculation/checkpoint, keeping pause time out of solver
runtimes. Once paused, another task can run. Return to the original task and
select **Resume**. It continues a live paused worker or starts a replacement
worker using compatible checkpoints after a failure, stop, or interruption. Stop interrupts the unfinished unit. Browser/server
restarts reconnect to a live worker; after machine restart, recovery is manual.
Private worker records, logs and export files live in ignored `.gui/`.

The **Graph generator** inside each task page builds interactive Plotly charts from numerical arrays
and tables. It never displays saved PNG/PDF figures. Drag to pan, scroll to zoom,
hover for values and click legends to toggle traces. Activity/spatial views select
a split and realization; communication also selects an available L/SNR condition.
HSV matrices expose the actual magnitude, phase, real and imaginary values on
hover. NMSE uses logarithmic axes/colors.

**Regenerate publication exports** renders fresh PNG/PDFs from the same data using
the existing code-defined Matplotlib typography. Standard task PNG outputs
remain available as reproducible artifacts;
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

Under `figures`, `tuning_main_index` and `evaluation_main_index` set the dashboard's
initial realization for each split. Edit them in experiment settings under Figure
selections. Evaluation defaults to realization 458, the selected paper example.
The index lists independently control automatic figure exports.

The default experiment uses fixed activity: `activity.exact_sparsity: true`
conditions event-driven samples on exactly `system.S: 20` active MTDs in both
tuning and evaluation. Kappa calibration still targets the unconditioned mean.
Existing setups retain their saved settings; changing an existing setup to fixed S
requires regenerating activity and dependent inputs, tuning and algorithm results.

`configs/experiment.yaml` defines grouped environment, activity, calibration,
communication, evaluation and algorithm settings.
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

The default experiment writes a `results/experiment/config.yaml` snapshot.
Named GUI setups keep their input definition in `results/<name>/config.yaml`; tasks
do not overwrite it. Each task has a
JSON manifest with its effective configuration and runtime metadata. Font sizes
and figure dimensions are defined locally in plotting functions, not YAML.

## Algorithm tuning and metrics

SBL > Tune selects its own tau on the saved tuning realizations. ANC > Tune uses
full Optuna TPE independently for each selected L/SNR condition: 50 trials, each
scored on the same first 100 saved tuning realizations. The ten startup trials
and zero-penalty baseline are included in the 50-trial budget. There is no
finalist stage.

The current continuous search bounds are alpha=0..10 and beta=0..0.1. The tau
range is 0..0.5 in steps of 0.001. Optimize one shared tau across all 100
realizations within each trial, without additional solver calls. TPE receives
that trial's mean F1. Select the winner by mean F1, lower true-active NMSE,
then higher recall. The default budget is 5,000 solver evaluations per condition;
identical repeated parameter pairs reuse saved solver results. Evaluation uses
only the saved winner on independent held-out realizations.
SBL tuning remains independent of the ANC search.

The optimizer uses a configured seed with condition-specific SeedSequence streams,
sequential trials, and no pruning. Trial history lives in a condition-local
`study.sqlite3`, with `trials.csv`, `sampler.pkl` and progress
metadata. Atomic checkpoint snapshots keep sampler state, trial history and any
pending trial together; Resume rebuilds the database from that durable state and
reuses completed solver calls. Optuna/numerical versions and source/input/settings
identities guard recovery. Search-history, sampled-pair and winner-threshold
figures regenerate directly from saved tables. ANC diagnostic plots display the
selected alpha, beta and tau, with the winning trial/pair highlighted.

Tune and Communication share the algorithm Run condition order: SNR sweep first,
then the pilot-length sweep without duplicate pairs, followed by any extra
required references and shared optional conditions. Sweep rows use Type Sweep;
extra required reference rows use Reference. Tune supports Complete all/selected,
Resume, and separate Rerun all/selected controls. Complete preserves compatible
finished conditions and reuses checkpoints; Rerun validates and resets only its
scope before solving, with a persistent reset journal for interruption recovery.

Per-condition Tune also provides Selected parameter sweeps in the graph generator:
ANC saves six alpha/beta/tau-versus-SNR/pilot-length figures; SBL saves its own two
tau sweep figures. They use compatible completed selections only, retain missing
conditions as gaps and show selected values as text. PNGs update under
`<algorithm>/figures/tuning/sweeps/`; interactive charts and PNG/PDF exports regenerate
from saved selections without solver calls. The existing condition diagnostics
remain available under their original figure type.

**Analysis > ANC parameter grid study** is a separate paper experiment with one
fixed L/SNR pair (default L=30, SNR=12 dB), 100 tuning realizations, and independently
editable alpha/beta/tau grids. It generates the F1, NMSE and selected-threshold
heatmaps under `anc_grid_study/`. It never writes an algorithm tuning selection;
ANC Run accepts full Optuna TPE selections only. Changing the paper grid does not
invalidate algorithm selections, and changing TPE budgets does not invalidate the
grid study. No preparation data is regenerated by either action.

OMP and CoSaMP use the feasible oracle sparsity budget B_q=min(S_q,L,N).
CoSaMP uses ridge regularization in both support fits, controlled by
`algorithms.cosamp.solver.cosamp_lambda_reg` (default `0.01`; `0` disables ridge).
The parameter is editable in experiment setup and CoSaMP task settings. Each fit
minimizes `||Y-A X||_F^2 + lambda_reg ||X||_F^2` using augmented SVD least squares.
The relative-residual stopping rule is unchanged. Earlier CoSaMP results are
preserved but incompatible with the revised solver and require an explicit Rerun.
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
  omp/run/L0025_snr8dB/
  cosamp/run/L0025_snr8dB/
  sbl/
    run/L0025_snr8dB/       # scores, timing, gamma data and condition figures
    tuning/                 # threshold curve and selection
  cgap_sbl_anc/
    run/L0025_snr8dB/
    tuning/                 # Optuna history, threshold curve and selection
  anc_grid_study/           # fixed-condition paper grid tables and heatmaps
  comparison/
  convergence/
```

Each `run/Lxxxx_snrXdB/` folder owns its `per_sample.csv`, `aggregate.csv`,
`runtime_summary.csv`, `summary.json`, and condition figures. SBL and ANC also
save condition-local gamma data. Algorithm-root sweep tables are combined views
of compatible completed conditions; the overall card completes when every configured
sweep condition finishes. Saved numerical outputs and figures are versioned.

Communication generation stores shared H, a and Z once per realization, with
Theta, Y, W and sigma2 for each condition. The first condition uses the shared
base archives; later conditions use separate condition input archives. Overlapping sweep conditions share inputs. Spatial dependence generation saves every C;
ANC loads those matrices and never constructs them during evaluation.

## Resume after interruption

Select **Resume** on the interrupted action. It preserves the original setup and
condition selection. Unfinished work saves Git-trackable checkpoints under its task/action or
condition folder, with portable requests in the task's `.recovery/` directory. Finished work is reused after a crash or process
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
numerical library versions and thread settings. Resume rejects incompatible work;
Explicit Start or Rerun begins a fresh calculation; completion actions reuse compatible progress. Generated communication and spatial files are checked against
saved hashes before reuse; missing or changed files are rebuilt. Unreadable
checkpoint units are recomputed. Old results produced before this feature are
not retroactively checkpointed.

JSON, YAML, CSV, NPZ, sparse matrices and figures are written through temporary
files, flushed, and atomically replaced. A failure before replacement leaves
the previous complete file intact. Checkpoints and portable recovery metadata are versioned while work is unfinished.
Each successful algorithm condition removes its checkpoints after saving results
and figures; other tasks clear theirs after full success. Pause, Stop and failure
retain unfinished progress. Local `.gui/` records and temporary files remain ignored.
Resume can recover after pulling the setup without its local GUI records.
Checkpoint additions and deletions are published only when you commit and push.

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

Normal tasks save PNG. Use **Regenerate publication exports** on the Activity
page to export the standard activity realization as PNG and vector PDF from
saved data. Native serif and Computer Modern math fonts are used with embedded
TrueType PDF fonts; bold Greek labels use local STIX math rendering.

## Code structure and validation

The GUI selects independent actions from the shared `tasks.py` registry. Data preparation lives in
`experiments/pipeline.py`; the estimator registry and shared tuning, scoring,
comparison and convergence workflow live in `experiments/estimators.py`.
Algorithm mathematics remains in `algorithms/`, data models in `models/`, and
plotting in `figures/`. A later GL method can supply its solver and tuning grid
through the same estimator interface, with a dedicated task and result folder.

```bash
make test-fast         # numerical/configuration unit checks
make test-gui          # Streamlit behavior
make test-integration  # saved data, workers, recovery
make test-plotting     # real PNG/PDF exports
make test-all          # full suite, two isolated processes
make test-profile      # full suite with the 20 slowest durations
```

Workflow-only tests use lightweight atomic artifact writers; dedicated plotting
tests and worker integration checks retain real rendering. Numerical solver tests
remain unchanged. Run affected tests while editing; scientific/tuning/recovery
changes require a full local run, and CI should run the full suite before merging.

See `VALIDATION.md` and `docs/theory.md` for validation scope and mathematical details.


CGAP-SBL-GL is available as the fifth algorithm, alongside CGAP-SBL-ANC.
It uses the saved spatial matrix with its diagonal retained, weighted degrees
`xi = C @ 1`, and the symmetric normalized graph penalty
`lambda = alpha * positive_part(gamma - r * (C @ (r * gamma)))`, where
`r = 1 / sqrt(xi)`. The posterior and gamma updates are shared with CGAP-SBL;
`alpha=0` recovers conventional SBL. Symmetric normalization has null vector
`sqrt(xi)`, so it does not treat uniform raw gamma as graph-smooth on irregular
graphs. Clipped lagged penalties do not establish global MAP convergence.

GL has its own Tune and Run pages, condition status controls, settings, checkpoints
and `results/<setup>/cgap_sbl_gl/` outputs. Default per-condition tuning uses
50 full TPE trials on the same 100 saved tuning realizations, tunes alpha in
`0..100`, and selects tau independently within each trial using the configured
threshold grid. These initial alpha bounds are editable and separate from ANC.
GL has no beta parameter. Saved-data figures include TPE history, sampled
alpha/threshold values, threshold scores, alpha/tau versus SNR and pilot length,
performance, gamma distributions and convergence, with interactive views and
PNG/PDF exports. No research runs are launched when adding this estimator.
