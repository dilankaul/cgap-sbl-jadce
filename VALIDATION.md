# Validation

Run `PYTHONPATH=src pytest -q`.

The suite covers complex-MMV dimensions, Gaussian generation, unit-norm QPSK
pilots, optimized Cholesky posterior calculations, stable CGAP-SBL updates,
zero-penalty SBL equivalence, sparse ANC, spatial cutoffs, independent seeds,
calibration, oracle greedy budgets and probability-preserving figures.

Workflow checks cover:

- Communication generation archives reproduce the deterministic communication model exactly.
- Independent estimator tasks use saved inputs and never regenerate communication.
- OMP/CoSaMP require neither tuning results nor spatial matrices.
- SBL tunes its own threshold without running ANC or loading C.
- ANC tunes alpha, beta and tau on tuning data; a separate Run action evaluates held-out data.
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
- Convergence needs completed SBL/ANC evaluation, excludes greedy algorithms, and supports SBL alone without C.
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
- Setup and task editors share one configuration, preview changes, and never launch a task when saving.
- Relevant edits mark results incompatible and block successors; restoring values restores eligibility.
- ANC tuning edits leave OMP, CoSaMP and SBL results usable; tuning policies can differ per algorithm.
- Adding L/SNR conditions preserves existing condition measurements, even after communication archives expand.
- Figure selections, YAML formatting and unrelated settings do not invalidate scoped recovery.
- Running tasks reject relevant edits until paused, while unrelated edits remain allowed.

- Inclusive range controls retain alpha/beta, kappa and evaluation candidates, validate endpoints and steps, support negative SNRs and singleton ranges, and preserve legacy custom grids.
