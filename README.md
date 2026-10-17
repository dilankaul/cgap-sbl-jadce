# CGAP-SBL JADCE experiments

Reproducible event-driven joint activity detection and channel estimation with
MMV-OMP, MMV-CoSaMP, SBL and six CGAP-SBL spatial variants: GLC, NGLC,
RGL, GL, GD and AGA. All methods share the saved event-driven activity data.

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

The overview shows three preparation boxes, nine algorithm boxes, and Compare
and Convergence boxes. Click a box to open its task page, with a completion
indicator and current progress on the overview. Activity opens Calibrate κ and
Generate activity; SBL and all six spatial variants open Tune and Run. Run requires compatible saved
tuning selections. Requirements follow data dependencies: all algorithms
can run in any order once their inputs are ready. Spatial dependence can be
built before communication; SBL needs no spatial matrices.

Compare has a selector for algorithms completed at the requested conditions,
and can use even one algorithm. Convergence offers only completed SBL-family
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
communication, evaluation conditions and individual algorithms. GLC owns its
alpha/beta/tau ranges; SBL owns its threshold range. τ, the initial kappa scan and evaluation pilot-length/SNR sweeps use inclusive
**Start / Stop / Step** controls. Spatial TPE parameters use continuous lower/upper bounds and independent Logarithmic sampling controls for each algorithm. Kappa bracket-refinement steps remain separate. Each algorithm has its own
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
**Pilot length for the SNR sweep**. GLC has its own **Parameter tuning pilot length L**
and **Parameter tuning SNR (dB)**; these do not change either sweep's fixed value.
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
shared channels/activity. Condition-specific SBL-family policies require tuning
the added pair on their **Tune** page first; reference policies reuse the reference
selection. Completed optional conditions are available for comparison and SBL-family convergence.

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

Choose a tuning policy independently for SBL and all spatial variants:

- **Pooled selection (default):** one parameter set across an editable tuning grid;
  see the pooled protocol below. Outputs live under `tuning/pooled/`.
- **One reference selection:** tune at the reference L/SNR and reuse its selection
  across all selected evaluation conditions.
- **Separate selections:** tune SBL tau and GLC alpha/beta/tau
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
| CGAP-SBL-GLC/NGLC/RGL/GL/GD/AGA | Independently Tune spatial parameters and tau, then Run evaluation |
| Compare | Summarize selected completed algorithms |
| Convergence | Record selected completed SBL-family iteration histories |


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

The default `selection_mode: pooled` uses one shared parameter selection per
spatial method across `tuning_grid`. The default anchors are L={20,30,40} and
SNR={4,12,20} dB. The 100 independent tuning realization IDs are assigned once:
11 per cell and 12 at (L=30, SNR=12). Assignment uses a deterministic child stream
of the existing communication-tuning seed and is identical for every trial and
method. Other grid sizes distribute samples evenly, assigning remainders to cells
nearest the central grid index; there must be at least one sample per cell.

Each trial solves only its assigned 100 inputs, selects the largest configured S
final gamma values (stable ties by device index), and maximizes mean sample F1.
This is oracle-cardinality tuning under the default exact-S activity model. The
support budget is used only for scoring; SBL solvers receive neither true support
nor a sparsity budget. Winner ties use lower true-active NMSE, then higher recall.
All six spatial methods retain independent parameter domains and editable trial
budgets; identical parameter candidates reuse completed solves. A 50-trial study
needs at most 5,000 solves total. SBL has no parameter search in pooled mode: its
Tune action prepares its assigned final estimates and calibration curves once.
Reference/per-condition modes retain their previous within-trial threshold objective.

After the pooled winner is fixed, gamma and normalized posterior-row-norm
threshold curves are calibrated from its cached tuning outputs. These thresholds
do not affect the parameter search. `assignment.csv`, `condition_scores.csv` and
`final_estimates.npz` retain the allocation, cell scores and winning estimates;
PNG, interactive Plotly and native-font PDF diagnostics use only saved data.
Pooled selections apply to every evaluation condition, including optional pairs.

`communication.grid` independently specifies the full Cartesian input grid, by
default L=20:5:45 and SNR=4:4:24 dB (36 pairs), for both tuning and evaluation
splits. Evaluation keeps its own editable fixed-reference sweeps. All three sets
of ranges are editable in experiment settings and relevant task forms. Preparation
also includes any required references or tuning anchors outside the input grid.
Preparing inputs does not launch algorithm evaluations at every grid condition.

The optimizer uses a configured seed with condition-specific SeedSequence streams,
sequential trials, and no pruning. Trial history lives in the selected tuning folder’s
`study.sqlite3`, with `trials.csv`, `sampler.pkl` and progress
metadata. Atomic checkpoint snapshots keep sampler state, trial history and any
pending trial together; Resume rebuilds the database from that durable state and
reuses completed solver calls. Optuna/numerical versions and source/input/settings
identities guard recovery. Search-history, sampled-pair and winner-threshold
figures regenerate directly from saved tables. GLC diagnostic plots display the
selected alpha, beta and tau, with the winning trial/pair highlighted.

Tune and Communication share the algorithm Run condition order: SNR sweep first,
then the pilot-length sweep without duplicate pairs, followed by any extra
required references and shared optional conditions. Sweep rows use Type Sweep;
extra required reference rows use Reference. Tune supports Complete all/selected,
Resume, and separate Rerun all/selected controls. Complete preserves compatible
finished conditions and reuses checkpoints; Rerun validates and resets only its
scope before solving, with a persistent reset journal for interruption recovery.

Per-condition Tune also provides Selected parameter sweeps in the graph generator:
GLC saves six alpha/beta/tau-versus-SNR/pilot-length figures; SBL saves its own two
tau sweep figures. They use compatible completed selections only, retain missing
conditions as gaps and show selected values as text. PNGs update under
`<algorithm>/figures/tuning/sweeps/`; interactive charts and PNG/PDF exports regenerate
from saved selections without solver calls. The existing condition diagnostics
remain available under their original figure type.

**Analysis > GLC parameter grid study** is a separate paper experiment with one
fixed L/SNR pair (default L=30, SNR=12 dB), 100 tuning realizations, and independently
editable alpha/beta/tau grids. It generates the F1, NMSE and selected-threshold
heatmaps under `glc_grid_study/`. It never writes an algorithm tuning selection;
GLC Run accepts full Optuna TPE selections only. Changing the paper grid does not
invalidate algorithm selections, and changing TPE budgets does not invalidate the
grid study. No preparation data is regenerated by either action.

OMP and CoSaMP use the feasible oracle sparsity budget B_q=min(S_q,L,N).
CoSaMP uses ridge regularization in both support fits, controlled by
`algorithms.cosamp.solver.cosamp_lambda_reg` (default `0.01`; `0` disables ridge).
The parameter is editable in experiment setup and CoSaMP task settings. Each fit
minimizes `||Y-A X||_F^2 + lambda_reg ||X||_F^2` using augmented SVD least squares.
The relative-residual stopping rule is unchanged. Earlier CoSaMP results are
preserved but incompatible with the revised solver and require an explicit Rerun.
Their raw rows record S_q, B_q and whether capping occurs. SBL and GLC receive
neither the true support nor the oracle sparsity budget.

NMSE uses the true active rows in both numerator and reference-energy denominator:

`NMSE_q = sum_{i:a_i=1} ||z_i-z_hat_i||_2^2 / sum_{i:a_i=1} ||z_i||_2^2`.

Missed active devices contribute error; estimates on inactive rows are excluded.
The full posterior estimate is retained for SBL methods. Zero-reference-energy
NMSE is undefined and excluded from its average; detection scores still include
those realizations. The true activity mask is used only for scoring.

Every new algorithm run saves every final complex `Z_hat`, every SBL-family gamma,
activity masks and realization IDs in condition-local `final_estimates.npz`.
Ground truth remains in the shared communication inputs identified by the run
manifest. Raw scores retain `nmse_unmasked` and `nmse_support_masked` as well as the
selected NMSE. Greedy estimates have no gamma array.

**Comparison > Task settings > Detection and NMSE** controls gamma thresholding
(`gamma >= tau`), Z thresholding (`norm(Z_hat_i)/sqrt(M) >= tau_Z`), or top-K
ranking by gamma or Z row strength. Thresholds may use each method’s saved tuning
calibration or explicit values. Greedy methods use their native oracle support
when the common rule is gamma; Z thresholding needs an explicit value, and top-K
uses their row strengths. Optional method-specific overrides are available.
Select unmasked NMSE or NMSE after zeroing undetected rows. Both use true-active
rows and exclude zero-reference-energy samples from the NMSE average. Comparison
rescoring never calls a solver or modifies original results; it records the
analysis settings and both NMSE variants. Threshold/K changes leave solver runs
eligible and require only rerunning Comparison. Evaluation-based threshold
exploration is separate from tuning-calibrated reported results.

Each algorithm task records solver-call wall time with `time.perf_counter`,
excluding tuning, loading, metrics, plots and saving. Raw scores, aggregate scores,
and runtime statistics belong to that algorithm's folder. Algorithm comparison checks source
identity, setup, selected parameters, completeness and metric scope before comparison.
NMSE line plots and convergence use logarithmic y-axes; the GLC NMSE heatmap uses
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
  cgap_sbl_glc/
    run/L0025_snr8dB/
    tuning/                 # Optuna history, threshold curve and selection
  cgap_sbl_nglc/            # independent Tune/Run, same condition layout
  cgap_sbl_rgl/
  cgap_sbl_gl/
  cgap_sbl_gd/
  cgap_sbl_aga/
  glc_grid_study/           # fixed-condition paper grid tables and heatmaps
  comparison/
  convergence/
```

Each `run/Lxxxx_snrXdB/` folder owns its `per_sample.csv`, `aggregate.csv`,
`runtime_summary.csv`, `summary.json`, and condition figures. All SBL-family methods also
save condition-local gamma data. Algorithm-root sweep tables are combined views
of compatible completed conditions; the overall card completes when every configured
sweep condition finishes. Saved numerical outputs and figures are versioned.

Communication generation stores shared H, a and Z once per realization, with
Theta, Y, W and sigma2 for each condition. The first condition uses the shared
base archives; later conditions use separate condition input archives. Overlapping sweep conditions share inputs. Spatial dependence generation saves every C;
Every spatial variant loads those matrices and never constructs them during evaluation.

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
plotting in `figures/`. Spatial variants share the estimator interface, tuning engine and plotting workflow,
with independent method parameters and output folders.

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

## Spatial variants and notation

The nine independent estimators are OMP, CoSaMP, SBL and CGAP-SBL with
GLC, NGLC, RGL, GL, GD or AGA interactions. All six spatial variants use the
same posterior, penalized variance update, relative stopping rule and detection.
MGLC is not implemented.

Retain the diagonal of $\mathbf C$ and define

$$
\mathbf J=\frac{1}{N}\mathbf1_N\mathbf1_N^T,\qquad
\xi_i=\sum_j C_{ij},\qquad
\boldsymbol\Xi=\operatorname{diag}(\xi_1,\ldots,\xi_N),\qquad
\mathbf F=\boldsymbol\Xi^{-1/2}\mathbf C\boldsymbol\Xi^{-1/2},\qquad
\mathbf L_{\mathrm{norm}}=\mathbf I_N-\mathbf F.
$$

Require finite, real, symmetric, nonnegative graph weights and positive degrees.
The following interaction matrices define
$\boldsymbol\lambda^{[k]}=[\boldsymbol\Omega^{[k]}\boldsymbol\gamma^{[k-1]}]_+$.
Constant interactions omit the iteration superscript.

| Variant | Interaction $\boldsymbol\Omega$ | TPE parameters |
| --- | --- | --- |
| GLC — Global–Local Contrast | $\alpha(\beta\mathbf1_N\mathbf1_N^T-\mathbf C)$ | α, β |
| NGLC — Normalized Global–Local Contrast | $\alpha(\beta\mathbf J-\mathbf F)$ | α, β |
| RGL — Regularized Graph Laplacian | $\alpha(\mathbf I_N-\kappa_{\mathrm g}\mathbf F)$ | α, κ_g |
| GL — Normalized Graph Laplacian | $\alpha(\mathbf I_N-\mathbf F)$ | α |
| GD — Graph Diffusion | $\alpha(\mathbf I_N-\mathbf T_t)$, $\mathbf T_t=\exp[-t(\mathbf I_N-\mathbf F)]$ | α, t |
| AGA — Adaptive Graph Affinity | $\alpha(\beta\mathbf J-\mathbf F_{\mathrm{AGA}}^{[k]})$ | α, β, h |

Use $\kappa_{\mathrm g}$ for graph coupling, distinct from activity-model κ.
Strict RGL requires $0\leq\kappa_{\mathrm g}<1$; GL is its boundary at 1.
GD uses $t\geq0$; $\mathbf T_t$ avoids conflicting with the physical channel
$\mathbf H$. GD evaluates the sparse matrix-exponential action, without a dense
exponential. AGA uses affinity $\mathbf A^{[k]}$, distinct from noise $\mathbf W$:

$$
\bar\gamma_i=\max(\gamma_i^{[k-1]},\epsilon_\gamma),\qquad
A_{ij}^{[k]}=C_{ij}\exp\!\left[-\frac{(\log\bar\gamma_i-\log\bar\gamma_j)^2}{2h^2}\right],
\quad h>0,
$$

$$
\xi_{\mathrm{AGA},i}^{[k]}=\sum_j A_{ij}^{[k]},\qquad
\boldsymbol\Xi_{\mathrm{AGA}}^{[k]}=\operatorname{diag}(\xi_{\mathrm{AGA},i}^{[k]}),\qquad
\mathbf F_{\mathrm{AGA}}^{[k]}=(\boldsymbol\Xi_{\mathrm{AGA}}^{[k]})^{-1/2}
\mathbf A^{[k]}(\boldsymbol\Xi_{\mathrm{AGA}}^{[k]})^{-1/2}.
$$

AGA recomputes its affinity and degrees from the previous γ on every iteration.
The positive floor ε_γ (default `1e-12`) affects logarithms only. All methods keep

$$
\eta_i^{[k]}=\Sigma_{ii}^{[k]}+\frac{\|\boldsymbol\mu_{i,:}^{[k]}\|_2^2}{M},\qquad
\gamma_i^{[k]}=\frac{2\eta_i^{[k]}}
{1+\sqrt{1+4\lambda_i^{[k]}\eta_i^{[k]}/M}}.
$$

The penalty has no additional division by M. At α=0 all six recover conventional
SBL exactly; GD also does so at t=0. Recompute the final unmasked posterior mean
from final γ. Adaptive positive-part penalties do not assert a fixed global
MAP/MM objective or a new convergence guarantee.

Each variant owns Tune/Run settings, selections, condition folders, checkpoint
and recovery records under `cgap_sbl_<method>/`, using canonical keys
`glc`, `nglc`, `rgl`, `gl`, `gd`, `aga`. GLC replaces the former `anc` identifier.
Existing numerical files are preserved on disk, without migration; old
incompatible selections/checkpoints cannot supply new runs.

Every variant uses independent full Optuna TPE with editable trial budgets and
ten startup trials by default. Pooled mode scores the same assigned tuning inputs
with top-S gamma detection and calibrates thresholds only after selection.
Reference/per-condition modes retain threshold scoring per trial. Initial editable
linear bounds are:

| Method | α | Additional parameters |
| --- | --- | --- |
| GLC | 0.001–1000 (log) | β: 0–5 (linear) |
| NGLC | 0.001–1000 (log) | β: 0–5 (linear) |
| RGL | 0.001–1000 (log) | κ_g: 0–1 (linear, upper endpoint excluded) |
| GL | 0.001–1000 (log) | — |
| GD | 0.001–1000 (log) | t: 0.01–10 (log) |
| AGA | 0.001–1000 (log) | β: 0–5 (linear); h: 0.01–10 (log) |

Configuration uses `graph_kappa`, `diffusion_time`, `affinity_bandwidth` and the
AGA solver setting `gamma_log_floor`. Each tuned parameter has its own `_low`,
`_high` and boolean `_log` settings under that algorithm. Logarithmic sampling
requires a positive lower bound; choose linear sampling to include alpha=0 or t=0.
RGL accepts an upper setting of 1 but the effective Optuna upper bound is
`nextafter(1, 0)`, preserving κ_g<1 and GL as the distinct boundary. Actual
distributions are saved in `search_space.json` and the Optuna study. These initial bounds are search choices,
not claims of optimal experimental ranges. Seeds and dataset definitions remain
unchanged. Per-condition Tune pages use shared condition controls, Complete, Resume and
scoped Rerun; pooled pages show their allocation and one study’s progress. Keep the fixed-condition GLC paper grid study independently owned
under `glc_grid_study/`; it never supplies algorithm selections.

Each method retains TPE history, sampled parameter plots and the winning τ
curve. AGA includes both α–β and α–h samples. Every tuned parameter and τ also
has SNR and pilot-length sweep plots, with missing/incompatible conditions shown
as gaps, numerical point labels and selected values on diagnostic plots. Standard
performance, γ distributions, comparison and convergence support all variants.
PNG previews, interactive Plotly figures and native-font PDF exports use saved
data and do not rerun solvers.
