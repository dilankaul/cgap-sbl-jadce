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

All SBL-family methods stop when

\[
\|\gamma^{[k]}-\gamma^{[k-1]}\|_2/\|\gamma^{[k-1]}\|_2<\delta,
\]

or after `K` iterations. Defaults are `K=500`, `delta=1e-4`, and the retained
positive initialization `gamma_init=0.1`. The final channel estimate is the
unmasked posterior mean recomputed from the final gamma. Detection uses
`gamma_i >= tau_A`, with independently tuned thresholds for CGAP-SBL and SBL.

## Spatial dependence and GLC implementation

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
\Omega_{\rm GLC}=\alpha(\beta\mathbf1_N\mathbf1_N^T-C),\qquad
\lambda_i=\alpha[\beta G-s_i]_+,
\quad G=\sum_j\gamma_j,\quad s=C\gamma.
\]

Here `1` is the all-ones matrix, not the identity; the GLC diagonal is retained.
`alpha,beta >= 0` are fixed tuning parameters. The GLC solver stores `C` sparsely
and evaluates this expression without forming dense `Omega_GLC`. A separate
general solver accepts any finite symmetric `Omega`.

## Six spatial interactions

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

See [README](../README.md#spatial-variants-and-notation) for independent TPE domains, storage and plots.

## Evaluation

For realization `q`, `S_q=sum(a)` equals S in fixed-count mode and varies in variable-activity mode. MMV-OMP and MMV-CoSaMP receive
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

Spatial dependence generation constructs and saves each C in compressed CSR format. Every spatial variant loads the
saved matrix. OMP, CoSaMP and SBL do not need a spatial matrix.
SBL and all six spatial variants select their parameters independently on the tuning data before
held-out evaluation; all methods use the same true-active-support NMSE.
