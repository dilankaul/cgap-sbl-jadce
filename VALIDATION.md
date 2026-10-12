# Validation of the revised CGAP-SBL implementation

Run the complete test suite with:

```bash
PYTHONPATH=src pytest -q
```

Final result: **102 tests passed**.

The revised implementation is checked against the local manuscript
`Publications___CGAP_SBL.pdf`. Validation covers:

- complex-MMV dimensions, ordinary channel transpose, Gaussian generation, and unit-norm QPSK pilots;
- optimized Cholesky posterior mean and covariance diagonal against a full reference;
- the stable nonnegative-penalty root, scalar stationarity, and maximization of each frozen penalized M-step;
- general symmetric indefinite interaction matrices, ANC equivalence, sparse/dense ANC agreement, and lagged penalty histories;
- exact SBL recovery when Omega or alpha is zero, relative stopping, and posterior recomputation from final gamma;
- ANC cutoff and diagonal, with rho=kappa and U=D taken from saved activity data;
- deterministic calibration, independent seeds, variable realized activity, and untouched probability-field values;
- oracle baseline support recovery and B_q=min(S_q,L,N) capping;
- paired spatial randomization preserving count, channels, pilots and standardized noise, with support fixed across conditions;
- scenario-specific aggregation, paired F1/NMSE advantages, and exclusion of undefined zero-energy NMSE;
- rejection of legacy tuning, configuration provenance, task wrappers, PNG outputs, and title-free vector paper PDFs.

A complete nine-task smoke run was performed using the small test configuration
in a temporary output folder, followed by a paper activity PDF export. Saved
outputs were inspected: Task 05 produces both F1 and NMSE contours; Task 07
produces 216 metric rows for six samples, six conditions, and six method/scenario
combinations; Task 09 produces the main performance plots, four spatial-dependence
comparisons, and paired advantages. Revised estimator stages were rechecked after
final metadata changes.

This smoke run verifies integration, not publication performance. The full
1,000-realization experiment has not been rerun for CGAP-SBL. Historical results
remain unchanged, and no provisional manuscript performance values are copied
into generated results.

Task-02 randomization ownership was additionally verified by rerunning Tasks 02, 07 and 09 on the small temporary dataset. All six randomized maps and saved paired counts were inspected; tests verify the unchanged seed/index sequence and omission of event-model overlays.

Full Task-03 dataset ownership is verified by exact comparisons with the previous
generator, shared paired H/Theta identity and standardized noise, zero-activity
samples, source/seed mismatch rejection, selected diagnostics, and downstream
Tasks 05–08 with generation functions deliberately disabled.

The full configured Task-03 dataset was generated and inspected: 100 tuning
realizations, 1,000 paired evaluation realizations, 11 unique evaluation conditions,
4,300 compressed archives (171.3 MiB), and six diagnostic PNGs. Sampled pairs
including the two high-activity realizations satisfy the system identity,
prescribed noise variance/SNR, and shared paired H/Theta/standardized noise.
Tasks 03 and 05–09 were also smoke-tested together on the small temporary dataset.
This generates full communication inputs, not new publication performance results.

Current figure validation additionally covers independent tuning/evaluation
selections, four-digit realization folders, compact CLI summaries, separate
scenario exports, native HSV phase/magnitude mapping, and code-local activity
presentation settings. Task-03 pilot heatmaps use automatic cell sizing; bold
Greek labels use native STIX MathText. The current selected diagnostic set
contains 91 figures.
