# CGAP-SBL mathematics

This implementation follows **Correlation-Guided Adaptive Penalized SBL for Joint Activity Detection and Channel Estimation in MTC**, manuscript `Publications___CGAP_SBL.pdf`. Its simulation setup is authoritative; its provisional performance statements are not imported as results.

## System and posterior

`N` is the number of MTDs, `M` the BS antenna count, and `L` the pilot length.
The binary activity vector is `a`, the physical channel is `H` with shape `(M,N)`,
and the effective channel and received pilots are

\[
Z=\operatorname{diag}(a)H^T,\qquad Y=\Theta Z+W.
\]

The transpose is ordinary, not conjugate. Unit-norm QPSK pilots have shape `(L,N)`.
Rows satisfy \(z_i\mid\gamma_i\sim\mathcal{CN}(0,\gamma_i I_M)\).
With \(\Gamma=\operatorname{diag}(\gamma)\), the posterior uses

\[
\Pi=\Theta\Gamma\Theta^H+\sigma^2I_L,\qquad
\mu=\Gamma\Theta^H\Pi^{-1}Y,
\]
\[
\Sigma_{ii}=\gamma_i-\gamma_i^2\theta_i^H\Pi^{-1}\theta_i,
\qquad \eta_i=\Sigma_{ii}+\|\mu_{i,:}\|_2^2/M.
\]

The implementation uses Cholesky solves and computes only the covariance diagonal.
The inverse symbols above are mathematical notation, not explicit inverses in the solver.

## Iteration-specific penalization

For a finite symmetric interaction matrix \(\Omega\), equations (29)–(37) define

\[
\lambda^{[k]}=[\Omega\gamma^{[k-1]}]_+,
\qquad j_i(\gamma_i)=-M\log\gamma_i-M\eta_i/\gamma_i-\lambda_i\gamma_i.
\]

The penalty is frozen during each M-step. The stable optimizer is

\[
\gamma_i^{[k]}=\frac{2\eta_i^{[k]}}
 {1+\sqrt{1+4\lambda_i^{[k]}\eta_i^{[k]}/M}}.
\]

`lambda_` stores the unscaled nonnegative penalty. It replaces the former `phi`
notation; the factor `1/M` appears in the root. Zero penalty gives exactly
`gamma=eta`, the conventional SBL update. No upper clipping is applied.
The interaction matrix need not be positive semidefinite. The Gaussian-inspired
quadratic energy is not treated as a proper Gaussian hyperprior, nor as a fixed
global objective whose monotonic improvement or convergence is guaranteed.

Both SBL methods stop when

\[
\|\gamma^{[k]}-\gamma^{[k-1]}\|_2/\|\gamma^{[k-1]}\|_2<\delta,
\]

or after `K` iterations. Defaults are `K=500`, `delta=1e-4`, and the retained
positive initialization `gamma_init=0.1`. The final channel estimate is the
unmasked posterior mean recomputed from the final gamma. Detection uses
`gamma_i >= tau_A`, with independently tuned thresholds for CGAP-SBL and SBL.

## ANC implementation

The spatial dependence matrix uses the paper's finite-distance model,

\[
C_{ij}=\begin{cases}
\dfrac{e^{-u_{ij}/\rho}-e^{-U/\rho}}{1-e^{-U/\rho}},&u_{ij}\le U,\\
0,&u_{ij}>U.
\end{cases}
\]

Its diagonal remains one. The simulation matches `rho=kappa` and `U=D`, resolved
from the saved Activity generation activity parameters. This is a spatial dependence model,
not an empirical activity covariance.

Equations (65)–(69) specify

\[
\Omega_{\rm ANC}=\alpha(\beta\mathbf1-C),\qquad
\lambda_i=\alpha[\beta G-s_i]_+,
\quad G=\sum_j\gamma_j,\quad s=C\gamma.
\]

Here `1` is the all-ones matrix, not the identity; the ANC diagonal is retained.
`alpha,beta >= 0` are fixed tuning parameters. The ANC solver stores `C` sparsely
and evaluates this expression without forming dense `Omega_ANC`. A separate
general solver accepts any finite symmetric `Omega`.

## Evaluation

For realization `q`, `S_q=sum(a)` varies naturally. MMV-OMP and MMV-CoSaMP receive
only the feasible oracle count `B_q=min(S_q,L,N)`, never the true identities.
Their estimated supports use the top `B_q` recovered row energies. SBL methods
receive neither `S_q` nor `B_q`. Channel NMSE uses the unmasked final posterior
means for SBL and sparse least-squares estimates for greedy methods, restricted
in both numerator and denominator to the true active set
`A_q = {i : a_i=1}`:

`NMSE_q = sum_{i in A_q} ||z_i - z_hat_i||_2^2 / sum_{i in A_q} ||z_i||_2^2`.

Missed active MTDs contribute their estimation error; estimates on inactive rows
are excluded. The true activity mask is used only for scoring, never passed to
SBL estimators. Convergence traces use the same metric. Zero-energy realizations
have undefined NMSE and are excluded from that average. Detection
metrics still include them, with empty-denominator scores set to zero.

Stored q and figure indices are zero-based; the manuscript indexes realizations
from one.

Communication generation stores the event-driven communication inputs for all configured
realizations and conditions. H, a, Z and per-L Theta are saved once per
realization; Y, W and sigma2 are saved per unique condition. All algorithm tasks
read those same inputs. The data distribution and communication seeds are unchanged.

Spatial dependence generation constructs and saves each C in compressed CSR format. ANC loads the
saved matrix. OMP, CoSaMP and SBL do not need a spatial matrix.
SBL and ANC select their parameters independently on the tuning data before
held-out evaluation; all methods use the same true-active-support NMSE.
