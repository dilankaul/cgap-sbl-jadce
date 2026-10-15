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

Spatial dependence generation's direct pairwise-distance construction costs O(N²) per realization,
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

The algorithms each measure their own algorithm using `time.perf_counter()`.
Every raw row records runtime_s by realization, L, SNR and sweep. General dense
CGAP-SBL is a library method, not a separately evaluated baseline.

The timer includes solver initialization, input validation, all iterations and
the final channel estimate. Tuning, data generation, C construction, archive
loading, support thresholding, metrics, plotting and saving are excluded.
Each algorithm's manifest records platform, library versions, CPU information
and numerical thread settings. Execution is serial.

Each algorithm saves runtime_summary.csv with count, mean, standard deviation,
median, 95th percentile, minimum, maximum and total seconds per condition.
Algorithm comparison compares their measured runtimes versus SNR and pilot length without
executing solvers. Wall-clock statistics depend on hardware and software;
no provisional manuscript execution times are inserted into results.
