# Validation

Run `PYTHONPATH=src pytest -q`.

The suite covers complex-MMV dimensions, Gaussian generation, unit-norm QPSK
pilots, optimized Cholesky posterior calculations, stable CGAP-SBL updates,
zero-penalty SBL equivalence, sparse GLC, spatial cutoffs, independent seeds,
calibration, oracle greedy budgets and probability-preserving figures.

Workflow checks cover:

- Communication generation archives reproduce the deterministic communication model exactly.
- Independent estimator tasks use saved inputs and never regenerate communication.
- OMP/CoSaMP require neither tuning results nor spatial matrices.
- SBL tunes its own threshold without running GLC or loading C.
- GLC tunes alpha, beta and tau on tuning data; a separate Run action evaluates held-out data.
- Tune needs no evaluation archives; Run reuses selections unchanged and never retunes.
- Solver-call timers exclude archive loading and metric calculation.
- Failed and abruptly terminated writes preserve previous complete files.
- Fresh-process restart reuses committed scores and runtimes without duplicates.
- Calibration restores RNG state and exact running sums; activity retains its seed sequence.
- Generation validates output hashes before reuse; corrupted units are rebuilt.
- Tuning and convergence reuse completed solver outputs after injected failure.
- Comparison reads compatible completed outputs and executes no solver.
- Historical, incomplete or mismatched scores and tuning selections are rejected.
- True-active-support NMSE ignores inactive estimates, counts missed active rows,
  handles zero energy, and is used consistently in convergence traces.
- GUI navigation and setup creation never start an unrequested task.
- Named setups copy defaults, keep data/results isolated, and preserve one source config.
- Independent algorithm order and selected-condition runs retain prior scores and runtimes.
- Both tuning policies keep evaluation data held out; condition-specific runs require
  only requested selections and reject stale selection identities.
- Partial-condition comparisons validate all requested algorithms without solver execution.
- Interactive charts read arrays/tables rather than image files, expose exact HSV values,
  preserve zero activation probabilities and maintain log NMSE scales.
- Selected-condition views and publication exports regenerate from matching numerical inputs.
- Analysis selectors include only compatible completed conditions; single-algorithm comparison works.
- Convergence needs completed SBL/GLC evaluation, excludes greedy algorithms, and supports SBL alone without C.
- Paused tasks allow another independent worker to run before continuation.
- Paused workers reattach after controller restart; killed workers resume from checkpoints.
- Import-safe GUI/task entry points, selected PNG diagnostics, local fonts and paper PDFs.


Tests use small temporary datasets. The full publication experiment is not run
as part of this restructuring; historical measured results retain their metric
scope and are not promoted to current results.

- Individual condition folders retain scores and runtimes; completing conditions separately completes the algorithm.
- Explicit fresh reruns recalculate selected conditions and preserve other completed results.
- Successful condition data and figure saves clear only owned checkpoints; figure failure retains numerical progress.
- Portable recovery succeeds from a copied setup without `.gui/`, and a cleanup interruption repeats no calculations.
- Disjoint conditions of one algorithm can run while another condition is paused.

- Completion actions leave completed condition files and measured runtimes untouched, including when requested repeatedly.
- Complete selected restores a partial condition from an interrupted wider request; incompatible saved work requires explicit Rerun.
- GUI action labels follow saved state, and Complete all progress controls remain visible from a selected-condition view.

- Grouped setup edits preserve effective values and independently owned algorithm settings.
- Setup and task editors share one configuration and never launch a task when saving.
- Relevant edits mark results incompatible and block successors; restoring values restores eligibility.
- GLC tuning edits leave OMP, CoSaMP and SBL results usable; tuning policies can differ per algorithm.
- Adding L/SNR conditions preserves existing condition measurements, even after communication archives expand.
- Figure selections, YAML formatting and unrelated settings do not invalidate scoped recovery.
- Running tasks reject relevant edits until paused, while unrelated edits remain allowed.

- Inclusive range controls retain alpha/beta, kappa and evaluation candidates, validate endpoints and steps, support negative SNRs and singleton ranges, and preserve explicit custom grids.

- Fresh runs require preparation compatibility metadata and condition-local results.
- Old combined tables are not promoted into usable condition results.
- The dashboard contains no historical-result migration or tutorial captions.

- Cached communication identities exactly match direct hashing of the saved arrays.
- Each refresh shares file checks across conditions and restats inputs on the next refresh.
- Edited input/output files invalidate cached validation even when their modification time is restored.
- Restoring original arrays restores compatibility; unrelated conditions remain eligible.

On the existing 1,000-realization evaluation dataset, repeated checks for the
algorithm and analysis workflow cards took 0.75–0.93 s after the initial cache
fill, while an evaluation worker was running. The initial pass still hashes the
saved arrays. This measures status validation, not browser or graph rendering.

- Live logs handle nested tqdm cursor codes, partial writes and truncated UTF-8 tails without displaying broken bars; appended pause messages, warnings and errors remain visible.

- Run-all completion counts and table rows advance after each saved condition while the worker remains running; unchanged polls do not trigger a page rerun or read array/checkpoint contents. Result changes during a normal page render preserve navigation clicks.

- Fresh Run/Rerun retires all scoped condition results and old checkpoint units before the first solver call; optional/unselected results, inputs, tuning and unrelated files remain unchanged.
- Resume after a reset or mid-reset interruption uses only newly calculated units, including interruption before cleanup starts.
- Invalid prerequisites and linked condition paths prevent cleanup; combined numerical views exclude reset data immediately.

- Saved and interactive sweep lines use their sweep endpoints, including incomplete evaluation sweeps. GUI lines retain Plotly's default y-autorange and allow unrestricted zooming/panning; Reset restores the sweep view on linear and log axes. Optional Chromium tests exercise the actual toolbar.

- GUI spatial dependence and pilot cross-correlation matrices have square cells at responsive widths; complex communication matrices retain their existing proportions and color keys.


Optuna tuning checks cover the full trials-times-samples solver budget, identical
tuning samples for every trial, full-TPE-only selection, deterministic trial suggestions
after interruption (including the tell/checkpoint boundary), saved threshold
selection and independent paper-grid results/settings. Tune condition tables
support shared Add/Delete controls and preserve completed outputs. Full scientific
runs are not part of validation: tests use tiny saved datasets and explicit small
search budgets. Workflow logic tests substitute atomic artifact writers; dedicated
plotting tests retain real native-font PNG/PDF exports. Worker integration continues
to exercise the production renderer and detached execution path.

Use `make test-fast`, `make test-gui`, `make test-integration`, and
`make test-plotting` while editing. `make test-all` runs all checks with two isolated
workers; `make test-profile` reports slow tests. Scientific, sampling, tuning and
recovery changes require the full local suite before completion.


Condition-specific tuning validation also covers Run-equivalent row ordering and
Sweep labels, separate Resume/Rerun after Stop, worker Complete preserving finished
conditions, scoped resets before solver calls, interrupted-reset recovery and
linked-path protection. Parameter plots are checked against saved selections,
including missing-point gaps, incompatible-selection exclusion, independent SBL
thresholds, visible GLC parameter labels and real PNG/PDF export of both new sweep
plots and retained diagnostics. No research datasets are regenerated.


GL validation covers sparse penalties versus explicit normalized Laplacians,
lagged gamma, zero-alpha/isolated-self-loop SBL recovery, retained diagonals,
connectivity scaling and invalid graphs. Workflow coverage exercises independent
TPE alpha/tau selection, interruption before and after Optuna tell, deterministic
resume without repeated completed solver calls, scoped resets, separate GLC
ownership, the independent GL GUI card and condition controls, evaluation, comparison,
convergence, interactive plots and real PNG/PDF exports.


The canonical GLC name is used in task/configuration/solver/storage identifiers,
with no saved-result migration. Variant tests compare all six penalties to explicit
paper interactions, check lagged AGA recomputation, exact alpha-zero SBL recovery,
GD t=0, RGL's strict domain and GL limit, logs-only AGA flooring, retained diagonals,
sparse execution without dense spatial matrices/inverses and invalid graph rejection.
New-variant workflow tests cover independent parameter domains, shared saved tuning
samples, interrupted solver resume, scoped reruns, comparison/convergence, all parameter
sweeps, interactive figures and native-font PNG/PDF exports on tiny fixtures.


2026-10-17: pooled tuning, retained estimates and sampling-scale validation
passed the full sequential local suite: 454 passed, 1 skipped, in 705.21 s.
Command: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src pytest -q
--durations=20 --maxfail=1 -o addopts=''. Parallel execution was stopped after
two existing GUI-worker 30-second timeouts under concurrent load; their time
limits were not changed, and both passed in the full sequential run.

New coverage checks the 36-condition input grid, deterministic 11/12 allocation,
all seven pooled SBL-family workflows, top-S scoring without within-trial thresholds,
winning complex estimates/gamma, solver-free gamma/Z/top-K rescoring, unmasked and
support-masked true-active NMSE, analysis changes during checkpoint resume,
all tuning anchors in portable input signatures, saved-data PNG/Plotly/native-font
PDF diagnostics and actual realization-condition labels. Sampling checks cover
each algorithm's independent bounds/log flags, positive logarithmic domains,
strict RGL exclusion of 1, actual Optuna distributions and GUI Save behavior.
The default setup validates; seeds and solver equations were not changed.

Current experimental results were cleared as requested. A recoverable /tmp copy
was verified against all 16,069 inventoried files (537,780,036 bytes); the active
results tree contains no files. Source settings remain. No new research runs
were launched. The obsolete AGA worker was stopped after a completed-unit pause.
