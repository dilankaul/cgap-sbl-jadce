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

On an algorithm's **Run** page, open **Condition status**, enter L/SNR in the empty
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

Choose either tuning policy independently for SBL and ANC:

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
  omp/run/L0025_snr8dB/
  cosamp/run/L0025_snr8dB/
  sbl/
    run/L0025_snr8dB/       # scores, timing, gamma data and condition figures
    tuning/                 # threshold curve and selection
  cgap_sbl_anc/
    run/L0025_snr8dB/
    tuning/                 # alpha/beta/tau grid, threshold curve and selection
  comparison/
  convergence/
```

Each `run/Lxxxx_snrXdB/` folder owns its `per_sample.csv`, `aggregate.csv`,
`runtime_summary.csv`, `summary.json`, and condition figures. SBL and ANC also
save condition-local gamma data. Algorithm-root sweep tables are combined views
of compatible completed conditions; the overall card completes when every configured
sweep condition finishes. Saved numerical outputs and figures are versioned.

Communication generation stores H, a and Z once, Theta once per L, and Y, W and sigma2 per unique
condition. Overlapping sweep conditions share inputs. Spatial dependence generation saves every C;
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
