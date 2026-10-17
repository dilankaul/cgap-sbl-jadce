# AGENTS.md — CGAP-SBL JADCE Reproducibility Project

## Project purpose

This repository implements reproducible experiments for the paper
**Correlation-Guided Adaptive Penalized SBL for Joint Activity Detection and Channel Estimation in MTC**.
The implementation must remain mathematically consistent with the paper and
suitable for reproducible IEEE journal experiments.

Read `DECISIONS.md` before making algorithmic or experimental changes.

## Working rules

1. Work incrementally; do not redesign the entire pipeline unless explicitly requested.
2. Inspect the current implementation before changing mathematical code.
3. Do not silently change experiment definitions, seeds, dimensions, probability models, or hyperparameter meanings.
4. Preserve existing uncommitted work. Never run `git reset --hard`, `git clean`, or broad `git restore` commands unless explicitly requested.
5. Do not commit changes unless explicitly asked.
6. Run affected tests after each edit. Small GUI/configuration changes need targeted checks; run the full suite in CI before merging. Scientific, sampling, tuning and recovery changes require a full local `PYTHONPATH=src pytest -q` run before completion. Profile slow tests with `--durations=20`.
7. Prefer numerically stable linear algebra and avoid explicit matrix inverses.
8. If code behavior and paper equations disagree, identify the discrepancy instead of silently choosing one.

## Prerequisite workflow

Task identifiers and output folders are descriptive, never numbered.
The GUI overview contains preparation, algorithm and analysis boxes.

- Activity model: calibrate kappa, then generate activity.
- Communication and spatial dependence each depend on activity and can run independently.
- OMP/CoSaMP depend on evaluation communication only.
- SBL has Tune and Run; Run requires its saved tuning selection.
- GLC/NGLC/RGL/GL/GD/AGA each have independent Tune/Run actions and require C.
- Compare allows any selected completed algorithm, including just one.
- Convergence allows only completed SBL-family runs at the selected condition.

Tasks automatically save PNG artifacts; the GUI always constructs interactive
Plotly figures directly from saved numerical data. Publication exports regenerate
native-font PDF figures from data. Launch through cgap-sbl-lab (or python -m
cgap_sbl_jadce); the old curses menu, task_cli module and cgap-sbl command are removed.
The tasks.py registry owns orchestration. Dependencies determine availability,
not an ordinal sequence. Algorithm runs support independent L/SNR selections and merge
completed conditions. SBL and all spatial variants have separate Tune/Run actions and support either
reference tuning or condition-specific selections, chosen in a setup definition.
Default SBL and spatial-variant tuning uses one pooled selection across the configurable tuning grid; reference and per-condition modes remain available. See decision 124.
Per-condition Tune pages show the shared condition table with tuning-specific
completion, unfinished checkpoint and compatibility status, including shared Add/Delete
and optional-input preparation controls. All spatial variants show completed TPE trials. Tune and
Communication use Run sweep ordering and labels: SNR sweep, pilot-length sweep,
then required references and optional rows. Tune uses the same Complete/Resume/
Rerun controls, preserving completed conditions on Complete and resetting only
the explicit Rerun scope with recoverable cleanup.
The Analysis GLC parameter grid study has one fixed condition and cannot supply
algorithm selections. TPE and grid-study settings/results have independent ownership.
New GUI setups copy default/current settings into isolated results/<name>/config.yaml
before work starts. That one source config can be edited centrally or within
its task page; task manifests retain the effective settings that produced results.
Compatibility is recomputed from relevant values and inputs, and restoring matching
settings restores eligibility. Preserve incompatible results and block their
downstream use. Algorithm solver/tuning settings belong to that algorithm only.
Setup editors expose only fixed-reference sweeps. Algorithm Tune/Run and Communication pages manage shared optional
additional L/SNR pairs in the Condition status table, using an empty input row
with Add and per-row Delete controls, without a separate Save step. They require saved communication inputs and independent
per-condition tuning when selected. All/Complete all includes only setup sweeps.
Communication uses the same condition table and scope/run controls as algorithm Run.
Each condition has one combined status and is Completed only when both tuning and
evaluation inputs are saved. Complete selected/all skips completed inputs, resumes
compatible work and starts missing conditions; optional pairs remain outside All.
Preparation from either page always saves both splits, preserving shared inputs
and completed algorithm results. Selected conditions can start with no communication data.
Settings forms require explicit Save, disable Enter submission, and discard drafts
on Reset or navigation. Keep all setup datasets, results and checkpoints separate.
Detached GUI workers pause between completed units so timing excludes paused time;
private job records and generated exports live in ignored .gui/.

After each task inspect saved outputs before continuing. Long-running tasks must
show progress. Figures must remain regenerable from saved data without rerunning
expensive tasks. New scientific work must checkpoint completed units and use
atomic output writes. Label fresh actions Start or Rerun according to saved state.
Resume restores compatible checkpoints. Algorithm Complete all/selected skips
completed conditions, restores compatible unfinished work and starts untouched
conditions; only an explicit Rerun replaces completed numerical results.
Fresh algorithm runs validate the requested scope, then reset its known outputs
and owned checkpoints before the first solver call. Keep reset cleanup recoverable;
Resume retains new results and must never repeat a finished reset. Preserve input
archives, tuning selections, optional conditions outside the scope and unrelated files.
Track unfinished checkpoints and portable recovery
requests in Git; clear owned checkpoints after saving final results and figures.
Algorithm runs own separate L/SNR folders and accumulate overall completion;
never reuse work after incompatible settings, inputs or mathematical code changes.

## Reproducibility

Use `np.random.default_rng` and `np.random.SeedSequence`. Keep activity and
communication random streams separate. Do not silently change configured seeds.
All algorithms compared in one condition must receive the same underlying
realization. Tuning and final evaluation data must remain separate.

## Paper notation

Use paper notation in new code where practical:

- `N`: number of MTDs
- `M`: number of BS antennas
- `L`: pilot length
- `S`: target/realized active MTD count as defined by the experiment
- `a`: binary activity vector `(N,)`
- `H`: physical channel matrix `(M, N)`
- `Z`: sparse effective channel `(N, M)`
- `Theta`: pilot/sensing matrix `(L, N)`
- `Y`: received pilots `(L, M)`
- `gamma`: SBL row-variance vector `(N,)`
- `Gamma = diag(gamma)`
- `C`: spatial dependence matrix
- `Omega`: symmetric spatial interaction matrix
- `lambda_`: nonnegative adaptive penalty vector
- `sigma2`: noise variance
- `S_q`: realized active count; `B_q`: feasible greedy oracle budget

The system model is `Z = diag(a) H^T` and `Y = Theta Z + W`.
Do **not** replace `H^T` with `H^H`.

## Complex-MMV SBL theory

Rows satisfy `z_i | gamma_i ~ CN(0, gamma_i I_M)`.

Use

`eta_i = Sigma_ii + ||mu_i||_2^2 / M`.

Conventional SBL uses `gamma_i_new = eta_i`.

Use the observation covariance

`Pi = Theta diag(gamma) Theta^H + sigma^2 I`

and Cholesky solves rather than `inv(Pi)`. Do not construct full `N x N`
posterior covariance during normal runs. Compute `mu` and `diag(Sigma)` only.

## Revised CGAP-SBL theory

Follow `Publications___CGAP_SBL.pdf`, equations (29)–(41) and (65)–(69).
Use `lambda = positive_part(Omega @ gamma_previous)` without dividing lambda by M.
For each iteration freeze lambda and eta, then use
`gamma_new = 2*eta/(1 + sqrt(1 + 4*lambda*eta/M))`.
Zero penalty recovers conventional SBL exactly. Never upper-clip gamma.

The general interaction is finite and symmetric, not necessarily PSD. It is a
Gaussian-inspired design for iteration-specific penalization, not a proper
Gaussian hyperprior or a fixed-global MAP/MM convergence guarantee.
The GLC matrix is `Omega_GLC = alpha*(beta*1-C)` with its diagonal retained;
`1` means all ones. Compute `lambda = alpha*positive_part(beta*sum(gamma)-C@gamma)`
without constructing dense Omega. Match `rho=kappa`, `U=D` from saved Activity generation data.

Use the relative stopping criterion `norm(gamma_new-gamma)/norm(gamma) < delta`,
with `K=500`, `delta=1e-4`. Independently retune thresholds and GLC parameters;
reject old method/setup selections. The final channel estimate is the unmasked
posterior mean from final gamma.

Tuning: use saved tuning inputs. Default pooled tuning assigns the 100 unique realizations across configurable anchor conditions, with 11 per cell and 12 at the center for the default 3x3 grid. Each spatial variant independently tunes its parameters by top-S gamma mean F1 with its configured trial budget, then calibrates thresholds from cached winning outputs. SBL prepares its assigned outputs once. One pooled selection serves all evaluation pairs. Reference/per-condition modes retain their previous objectives. There is no finalist stage. Keep the fixed-condition paper grid study separate from algorithm selections. Threshold calibration uses tuning data only; analysis can explore alternative thresholds/K from saved final estimates. Preserve per-condition tuning diagnostics and annotate GLC selections. Save GLC alpha/beta/tau versus SNR and pilot length, plus independent SBL tau sweeps, from compatible completed selections; show missing conditions as gaps. Provide PNG previews, interactive plots and native-font PDF exports without solver calls.
Evaluation: Q=1000, L=20:5:45 at 12 dB, SNR=4:4:24 at L=30.
Convergence: L=30, 12 dB. Use the six explicit independent seeds.
Activity generation saves event-driven tuning and evaluation supports. Communication generation saves all
communication inputs; downstream tasks load them and never regenerate data.
Both communication splits cover the configurable full Cartesian communication grid, every configured sweep condition and
independent algorithm/preview reference conditions outside the sweeps,
regardless of tuning selection policy. Optional pairs retain explicit input preparation.
Spatial dependence generation owns spatial dependence matrices. OMP/CoSaMP need no tuning or C;
SBL tunes tau independently; each spatial variant tunes its own parameters and tau.
Save every final complex Z_hat and SBL-family gamma. Record unmasked and support-masked true-active-support NMSE and solver-call runtimes. Analysis detector/NMSE settings never invalidate solver results. Algorithm comparison compares only
compatible completed results.
Spatial methods are GLC, NGLC, RGL, GL, GD and AGA (no MGLC), each with independent
full TPE, condition-local outputs under cgap_sbl_<method>/, recovery and plots.
Use J=11^T/N, xi=C@1, Xi=diag(xi), F=Xi^(-1/2) C Xi^(-1/2), L_norm=I-F.
Retain C's diagonal; require finite real symmetric nonnegative C and positive degrees.
Interactions: GLC alpha*(beta*11^T-C), NGLC alpha*(beta*J-F),
RGL alpha*(I-graph_kappa*F), GL alpha*(I-F), GD alpha*(I-T_t),
AGA alpha*(beta*J-F_AGA_k). Graph coupling kappa_g satisfies 0<=kappa_g<1;
GL is the boundary at 1, distinct from activity-model kappa.
T_t=exp[-t*(I-F)] with t>=0; compute sparse exponential actions without dense T_t.
Use T_t rather than H_t (H denotes physical channel), and A_k rather than W_k
(W denotes noise) for AGA affinity C_ij*exp[-(log(gamma_bar_i)-log(gamma_bar_j))^2/(2*h^2)].
AGA recomputes degrees and normalized affinity from previous gamma every iteration,
h>0 and gamma_bar=max(gamma,epsilon_gamma) for logs only; never floor actual gamma.
All variants share unchanged E/M updates, relative stopping and final unmasked posterior.
Alpha=0 exactly recovers SBL; GD t=0 does too. No new global convergence claim.
Editable defaults for every spatial method: alpha 0.001..1000 logarithmic; beta 0..5 linear where applicable; RGL graph_kappa 0..1 linear with the upper endpoint excluded; GD t 0.01..10 logarithmic; AGA h 0.01..10 logarithmic, gamma_log_floor=1e-12. Bounds and boolean log sampling flags belong independently to each algorithm/parameter. Log sampling requires positive lower bounds; h remains strictly positive and graph_kappa remains below 1. See decision 125.
Every variant saves TPE history, parameter samples, threshold scores and sweeps
of every tuned parameter and tau versus SNR/pilot length, including selected values.
AGA includes alpha-beta and alpha-h diagnostics. PNG/Plotly/native-font PDF use saved data.

## Spatial activity model

For device/event distance `d_iv`,

`P_iv = (exp(-d_iv/kappa) - exp(-D/kappa)) / (1 - exp(-D/kappa))`

for `d_iv <= D`, and exactly zero for `d_iv > D`.

For multiple events use

`P_i = 1 - product_v(1 - P_iv)`.

Use the experiment dimensions from configs/experiment.yaml. This branch uses
N=500, fixed S=20, R=50 m and D=20 m. Use `activity.exact_sparsity: true`
by default: accept event-driven realizations only when their active count equals S.
Calibrate kappa automatically to the unconditioned target mean before applying
this exact-count conditioning. Retain explicit variable-activity support.
Activity generation must report realized-S mean, standard deviation, minimum, maximum, and
distribution/histogram.

## Baselines

SBL uses the same optimized posterior engine as CGAP-SBL where possible and an
independently tuned threshold.

MMV-OMP/SOMP is a joint-row-support oracle sparsity-budget baseline. MMV-CoSaMP is also
joint-support and uses explicit ridge regularization in both merged and pruned
support fits, solved as augmented SVD least squares. Configure
`algorithms.cosamp.solver.cosamp_lambda_reg` (default 0.01; zero is unregularized).
For realization `q`, OMP and CoSaMP use the feasible oracle sparsity budget
`B_q = min(S_q, L, N)`. Record whether capping occurs. CGAP-SBL and SBL solvers never
receive `S_q` or `B_q`; pooled top-S scoring uses configured S only.

## Activity figures

Selected-realization figures must be reproducible from saved Activity generation data and
may show the BS, cell boundary, inactive/active MTDs, events, optional cutoff
circles, activation-probability field, and a colorbar. Save PNG previews and
vector PDF publication output.

The displayed field must be the actual model probability

`P(x) = 1 - product_v(1 - P_v(x))`.

Never alter the probability values for visual purposes. Specifically do not:

- threshold low probabilities;
- replace zeros;
- add `1e-6` or machine epsilon;
- mask values merely because `P=0`;
- artificially whiten regions outside event influence;
- modify `probability_field` before plotting.

Masking only the region outside the physical circular BS cell is allowed.
The colormap itself maps `P=0` to pure white and increasing probability toward
dark red. Publication PDFs use vector `contourf`, with contour levels spanning
`np.linspace(0.0, 1.0, probability_levels)`.

## Results and tests

Prefer compressed NPZ for numerical arrays, CSV for tabular results, JSON/YAML
for summaries/manifests, PDF for publication figures, and PNG for previews. Do
not save enormous per-iteration histories for every realization.

Important tests include matrix dimensions, system-model identity, complex
Gaussian generation, optimized posterior vs a full reference, conventional SBL,
stable CGAP-SBL root, nonnegative lagged `lambda`, GLC/spatial dependence construction, activity
cutoff, multi-event probability, deterministic seeds, simple noiseless baseline
support recovery, and plotting that does not modify probability values.

Do not weaken tests merely to make an implementation pass.

## Fresh result format

Saved results and local GUI state were cleared at the user’s request. Do not
restore them or add result migration code. New preparation results require their
compatibility metadata; algorithm results require their condition-local files.
Keep setting/input/code guards and recoverable checkpoints for future runs.
The dashboard shows controls, settings, progress and necessary error messages;
avoid tutorials and repeated explanations. Keep workflow rationale and future
plans in these project documents rather than code comments. Scientific equations
and short notes on non-obvious numerical behavior may remain in code.


Canonical CGAP identifiers are glc, nglc, rgl, gl, gd and aga; GLC replaces anc.
Preserve existing result bytes without migration. New code/settings incompatibility
blocks old selections and checkpoints. See README spatial notation and decision 118.
