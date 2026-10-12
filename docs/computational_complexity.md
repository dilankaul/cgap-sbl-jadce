# Complexity and runtime audit

Checked against Section VII, equations (75)–(79), of
`Publications___CGAP_SBL.pdf`. These equations agree with the implemented solvers.

Let

\[
P=L^3+L^2(N+M)+LNM.
\]

| Method | Per iteration / step | Complete solve |
|---|---|---|
| SBL | \(O(P)\), paper (76) | \(O((k+1)P)\) |
| General CGAP-SBL, dense Omega | \(O(P+N^2)\), paper (77) | \(O((k+1)P+kN^2)\) |
| CGAP-SBL-ANC, sparse C | \(O(P+\mathrm{nnz}(C)+N)\), paper (79) | \(O((k+1)P+k(\mathrm{nnz}(C)+N))\) |

Here k is the actual number of iterations, bounded by K. The extra posterior
calculation following termination produces the final unmasked channel estimate.
An initialization/validation pass is also included in measured solve time.
The paper gives per-iteration asymptotic costs, not execution times or fixed
iteration counts. Sparse ANC is not guaranteed to finish faster than SBL:
iteration counts and numerical overhead also matter.

## Posterior operations

`algorithms/posterior.py` constructs Pi in O(L²N), factorizes it in O(L³),
solves for Y in O(L²M), and applies one triangular solve to all pilot columns
in O(L²N). The squared whitened-column norms give the covariance diagonal in
O(LN). The mean needs O(LNM), and eta needs O(NM).

The paper describes observation-covariance solves for both M and N right-hand
sides. The implementation uses a triangular whitening solve for the N columns
rather than a full two-pass inverse application. This saves a constant factor
without changing equation (76). Normal runs never form an N×N posterior
covariance or explicitly invert Pi.

ANC uses a CSR matrix-vector product and does not form dense Omega_ANC. Its
aggregate gamma sum and positive-part update need O(N). If C becomes dense,
its interaction cost returns to O(N²), as noted in the paper.

Task 04's direct pairwise-distance construction costs O(N²) per realization,
with O(N²) temporary storage. This one-time preprocessing, sparse archive
writing and later archive loading are excluded from the solver timings,
consistent with Section VII's exclusion of spatial-matrix construction.

## Greedy baselines

The following bounds are derived from this implementation; Section VII does
not provide OMP/CoSaMP formulas. Write B=B_q=min(S_q,L,N), and assume B>0.

OMP takes B support-selection steps. Each evaluates a joint-MMV proxy in
O(LNM) and recomputes unregularized SVD least squares. Summing the increasing
support sizes gives an upper bound

\[
O(BLNM+LB^3+LMB^2).
\]

CoSaMP merges at most min(3B,N) candidates per iteration, performs two SVD
least-squares fits, prunes joint row energies and computes a dense residual.
With k_co actual iterations, an upper bound is

\[
O(k_{co}(LNM+LB^2+LBM)).
\]

For zero budget both return a zero (N,M) array, requiring O(NM) output storage
and initialization. Greedy methods use the oracle budget; SBL methods do not.
These bounds do not imply a performance ranking or replace measured runtimes.

## Recorded execution time

Task 07 measures each CGAP-SBL-ANC, SBL, MMV-OMP and MMV-CoSaMP call separately
using `time.perf_counter()`. Each raw metric row records `runtime_s`, grouped
by realization, scenario, L, SNR and sweep. CGAP-SBL-ANC and SBL are also timed
in the randomized scenario. General dense CGAP-SBL is implemented as a library
method but is not a separately evaluated baseline in the paper experiment.

The timer includes solver initialization, input validation, all numerical
iterations and the final channel estimate. Data generation, C construction,
archive loading, support selection from final estimates, metric computation,
plotting and saving are outside the timer. Execution is serial by default.
The Task-07 manifest records the platform, Python/NumPy/SciPy versions, CPU
information and explicit numerical thread settings.

Task 09 preserves mean runtime in aggregate.csv and produces runtime_summary.csv
with sample count, mean, standard deviation, median, 95th percentile, minimum,
maximum and total seconds for each scenario/method/condition/sweep. Standard
runtime figures show correlated-scenario algorithms versus SNR and pilot length.
Runtime statistics are measured wall-clock observations, not hardware-independent
constants. No provisional paper execution times are inserted into results.
