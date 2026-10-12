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
6. Run relevant tests after edits; before considering a change complete run `PYTHONPATH=src pytest -q`.
7. Prefer numerically stable linear algebra and avoid explicit matrix inverses.
8. If code behavior and paper equations disagree, identify the discrepancy instead of silently choosing one.

## Numbered task workflow

The scientific workflow has seven tasks (01–07). Utilities are intentionally
unnumbered because they do not represent extra experimental stages.

- Task 01: calibrate the activity model
- Task 02: generate activity realizations
- Task 03: generate and save full communication datasets
- Task 04: construct spatial dependence
- Task 05: tune CGAP-SBL alpha, beta and tau
- Task 06: run estimators, aggregate results and generate performance figures
- Task 07: convergence analysis

Numbered tasks automatically save PNG figures. The paper activity script
`scripts/paper_figures/activity_realization.py` exports one title-free vector PDF
from saved data. Numbered CLI options are only --config and --no-progress.
Plotting functions remain callable on saved data for regeneration.

After each task inspect saved outputs before continuing. Long-running tasks must
show progress. Figures must remain regenerable from saved data without rerunning
expensive tasks.

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
The ANC matrix is `Omega_ANC = alpha*(beta*1-C)` with its diagonal retained;
`1` means all ones. Compute `lambda = alpha*positive_part(beta*sum(gamma)-C@gamma)`
without constructing dense Omega. Match `rho=kappa`, `U=D` from saved Task-02 data.

Use the relative stopping criterion `norm(gamma_new-gamma)/norm(gamma) < delta`,
with `K=500`, `delta=1e-4`. Independently retune thresholds and ANC parameters;
reject old method/setup selections. The final channel estimate is the unmasked
posterior mean from final gamma.

Tuning: 100 samples at L=25, SNR=12 dB; alpha in 2:2:14, beta in .05:.05:.35.
Evaluation: Q=1000, L=20:5:45 at 12 dB, SNR=4:4:24 at L=30.
Convergence: L=30, 12 dB. Preserve the six existing seeds and use the seventh
explicit spatial_randomization seed for uniformly randomized paired supports.
Task 02 generates and saves paired randomized evaluation supports and spatial maps.
Task 03 saves full communication inputs with shared channels/pilots and
scenario-specific observations. Tasks 05–08 load these inputs and never regenerate
communication data. Task 06 loads saved supports and does not randomize activity. Randomized maps omit
event markers, cutoff circles, and event probability backgrounds.
Paired scenarios retain S_q, geometry, H, Theta, and C, with separately scaled
common standardized noise and fixed correlated-tuning selections. Evaluate only
CGAP-SBL-ANC and SBL in the randomized scenario. Do not import provisional
manuscript results as measured data or relabel historical estimator results.

## Spatial activity model

For device/event distance `d_iv`,

`P_iv = (exp(-d_iv/kappa) - exp(-D/kappa)) / (1 - exp(-D/kappa))`

for `d_iv <= D`, and exactly zero for `d_iv > D`.

For multiple events use

`P_i = 1 - product_v(1 - P_iv)`.

Current intended event influence radius is `D = 15 m`. The main experiment uses
variable realized activity. Do not force every realization to have exactly
`S=10`; instead calibrate `kappa` so that `E[S] ~= 10`. The current manuscript
and saved calibration use `kappa=3.78`.

Task 02 must report realized-S mean, standard deviation, minimum, maximum, and
distribution/histogram.

## Baselines

SBL uses the same optimized posterior engine as CGAP-SBL where possible and an
independently tuned threshold.

MMV-OMP/SOMP is a joint-row-support oracle sparsity-budget baseline. MMV-CoSaMP is also
joint-support, uses stable unregularized least squares, and has no hidden ridge
penalty. For realization `q`, OMP and CoSaMP use the feasible oracle sparsity budget
`B_q = min(S_q, L, N)`. Record whether capping occurs. CGAP-SBL and SBL never
receive `S_q` or `B_q`.

## Activity figures

Selected-realization figures must be reproducible from saved Task-02 data and
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
stable CGAP-SBL root, nonnegative lagged `lambda`, ANC/spatial dependence construction, activity
cutoff, multi-event probability, deterministic seeds, simple noiseless baseline
support recovery, and plotting that does not modify probability values.

Do not weaken tests merely to make an implementation pass.
