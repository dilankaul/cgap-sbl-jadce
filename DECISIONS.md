# Implementation decisions (later entries supersede earlier entries)

1. Use paper variables: `a`, `H`, `Z`, `Theta`, `Y`, `gamma`, `mu`, `Sigma_diag`, `C`, `Omega`, `eta`, `phi`.
2. Use proper complex-MMV SBL scaling with factor `M`, not the old real-Gaussian `1/2`.
3. Keep `Omega = alpha * (beta * 1 - C)` unchanged.
4. Do not insert an artificial `M` into the gamma correlation term.
5. Therefore use `phi = (Omega @ gamma) / M`.
6. Use the numerically stable positive root `2*eta/(sqrt(1+4*phi*eta)+1)` for `phi>0`.
7. Retain the legacy SBL fallback `gamma=eta` for `phi<=0`.
8. Do not clip gamma to an upper bound of 1. Only roundoff-level negative variances are projected to zero.
9. Retune CA-SBL alpha, beta and threshold under the corrected update. Tune the SBL threshold independently.
10. Use Cholesky solves and compute only `diag(Sigma)` during normal SBL/CA-SBL runs.
11. Use unit-norm QPSK pilots.
12. Implement the paper's event-distance cutoff exactly at `d <= D`.
13. The default event model is unconditioned: `system.S` is a target mean active count, not a hard per-realization constraint. MMV-OMP and MMV-CoSaMP use the feasible oracle sparsity budget `K_r = min(S_r, L, N)`, where `S_r` is the realized support size. They are described as oracle sparsity-budget baselines; capping is recorded whenever `S_r > L`.
14. CoSaMP uses SVD least squares with no hidden ridge regularizer.
15. All algorithms share identical Monte Carlo realizations per condition.
16. Tuning and final evaluation use separate deterministic seeds.
17. Large communication tensors are regenerated from saved seeds instead of stored, keeping outputs Git-friendly.

18. Use `D=15 m` for the revised activity model and calibrate `kappa` separately before activity generation. The initial configured value is `kappa=3.785`, subject to kappa calibration calibration.
19. Spatial activity figures may display the exact combined activation field `P(x)=1-prod_v(1-P_v(x))`; this is a visualization of the model probability, not a decorative gradient.
20. Replotting a saved activity generation realization uses the `D` and `kappa` recorded by activity generation, so later config edits cannot silently change the probability field shown for old data.
21. Probability-field data are never thresholded, offset, epsilon-shifted, or masked for visual effect. Only the region outside the physical BS cell may be masked.
22. Publication probability-field PDFs use vector `contourf` with levels spanning exactly `[0, 1]` and a custom colormap mapping `P=0` to pure white and `P=1` to dark red.

23. Earlier scientific stages used separate task launchers and a figures-only option. The GUI and descriptive task hierarchy supersede that interface.
24. Each scientific task saves its standard vector-PDF figures automatically by default; PNG is optional; `--figures-only` regenerates figures from saved results without computation.

25. activity generation saves spatial figures for every tuning and evaluation realization with dataset split, zero-based realization number, and active MTD count; the full experiment configuration is `configs/experiment.yaml`, with version-controlled outputs in `results/experiment/`.

26. Each task owns its output folder, with figures inside it. Activity data and figures distinguish tuning/evaluation subfolders; algorithm figures distinguish tuning and evaluation.

27. Name the first preparation action kappa calibration; its calibration procedure is unchanged.

28. With `activity.kappa_source: calibration`, activity generation automatically uses the compatible kappa calibration selected kappa, prints and records its provenance, and saves the effective run config. Missing/mismatched calibration is an error; configured mode remains explicit and seeds are unchanged.

29. kappa calibration uses an ascending broad kappa scan, bracket refinement ending at configured 0.005 spacing, and identical calibration geometries across stages. Independent deterministic validation reports expected-mean uncertainty and target gap without retuning. Candidate ranges, refinement steps, tolerance, and validation seed offset are explicit configuration; unbracketed calibration is a saved failure.

30. Both supplied configs use automatic kappa calibration calibration and omit fixed kappa values; calibration reports only the selected value. Replotting uses saved activity generation kappa, and fresh previews resolve the calibrated value unless explicitly overridden.

31. `configs/experiment.yaml` is the sole user-facing configuration and all scripts default to it. A small test fixture is retained under `tests/fixtures/small_experiment.yaml` for fast verification.

32. All scientific tasks save local effective config.yaml and environment manifest.json. kappa calibration records overrides and selected kappa; downstream snapshots retain activity-data kappa and runtime estimator selections. Root snapshots belong to activity generation, and input configs remain unchanged. Snapshots are created by actual runs, never fabricated for historical results.

33. Remove standalone utility scripts; each scientific task supports --figures-only, retaining regeneration from saved data without rewriting task configurations or rerunning experiments.

34. All task launchers are import-safe wrappers around a shared task CLI registry. Every task's computation is a pipeline stage; plotting is orchestrated after saved outputs. kappa calibration follows the same structure, with no changes to its calibration mathematics, seeds, candidate grids, or saved result definitions.

35. Consolidate generated YAML to one root config.yaml per run. Exact task configs are retained in JSON manifests under configuration; activity generation owns the root snapshot after activity generation. Figure dimensions and typography are defined inside plotting functions, not experimental YAML. This supersedes per-task YAML and YAML font overrides.

36. Use six explicit, distinctly named seeds under seeds: kappa_calibration, kappa_validation, activity_tuning, activity_evaluation, communication_tuning, and communication_evaluation. Remove base-seed offsets. Preserve five existing effective streams; change communication tuning from 20260817 to 20260818 as requested to make all six seeds distinct. Existing results remain historical; communication tuning and dependent results must be regenerated. Calibration means selecting kappa to match the target expected active MTD count; validation independently checks the selected kappa.

37. At the user's request, simplify the six explicit seeds to 1 through 6 in configuration order: kappa calibration, kappa validation, activity tuning, activity evaluation, communication tuning, communication evaluation. This supersedes the numerical values in decision 36 and changes all streams; regenerate the experiment starting with kappa calibration. Historical results are not rewritten.

38. Replace seeds 1–6 with distinct four-digit values at the user's request: kappa_calibration=1379, kappa_validation=8641, activity_tuning=2953, activity_evaluation=7069, communication_tuning=4283, communication_evaluation=9517. This supersedes decision 37 seed values; regenerate from kappa calibration and preserve historical result provenance.

39. Add a separate presentation-only configs/paper_figures.yaml for selected activity generation publication exports via --figures-only --paper-config. Export title-free PDFs with configurable typography and dimensions to a separate destination. Standard activity generation exports remain unchanged; reuse saved activity and recorded model parameters without computation or snapshot writes.

40. Supersede decision 39's separate paper YAML and task CLI option: scripts/paper_activity_figures.py exports selected saved activity generation realizations using the existing experiment config. Define publication font sizes and dimensions locally inside plot_paper_activity_figures, in the same style as task plot functions. Keep title-free PDFs and independent output folders; normal task plots remain unchanged.

41. Simplify algorithm tasks without changing their scientific outcomes: reuse normalized calibration settings, generate/save tuning and evaluation through one split loop, and resolve common activity plot options once per export. Keep standard and paper typography local and independent. Remove inactive max_attempts from supplied variable-activity configurations; preserve its default and explicit support for exact-sparsity runs. Preserve existing user-edited paper dimensions and fonts, saved paths, titles, probability values, and seeds.

42. At the user's request, scientific tasks always run their stage and automatically save standard PNG figures for repository viewing. Remove --figures-only and --figures format switches; standard plotting defaults are PNG. Keep the dedicated paper_activity_figures.py export as vector PDF with code-local typography and no separate configuration. Existing saved files are not deleted.

43. Keep only --config and --no-progress (and standard help) across all scripts. Remove calibration, activity split/field, worker, and paper selection CLI overrides and the unused override-copy helper. Paper selections stay editable in the plotting function. ANC parameter tuning and estimator evaluation use their existing single-worker default; keep their internal worker support for library callers. Remove obsolete unreferenced run_all and make_figures orchestrators. Preserve scientific configurations, random streams, standard PNG output, and paper PDF styling.

44. Organize source modules by responsibility: models contains activity, communication, realizations, and correlation; experiments contains calibration, tuning, and the pipeline; figures separates activity/paper plots, diagnostics, and common rendering. Keep algorithms and shared infrastructure in their existing roles. Update all imports and documented paths without compatibility facades or changes to equations, seeds, saved data, task commands, or figure styling.

45. Make figure modules navigable by the plotted quantity; keep shared rendering in common.py. Split broad activity.py and diagnostics.py modules into dedicated calibration, realization, distribution, communication, spatial dependence, tuning and performance modules. Move plotting functions without changing calculations.

46. Keep paper scripts under scripts/paper_figures/. The activity_realization.py paper script exports exactly one title-free PDF, selected by required --split tuning|evaluation and zero-based --index, with optional --config. Replace the batch paper export with plot_paper_activity_figure; preserve publication typography, saved-model parameters, and output paths. Numbered-task CLI options remain unchanged.

47. Name the single paper activity PDF event_driven_activation_model_[split]_[index].pdf, using the split name and unpadded zero-based index. Preserve its tuning/evaluation output subfolder.

48. Use four-digit index padding for paper activity PDF filenames, e.g. event_driven_activation_model_evaluation_0005.pdf. This supersedes decision 47's unpadded index.


49. Adopt the revised manuscript `Publications___CASBL (7).pdf` at the user's request. Rename the methods CGAP-SBL and CGAP-SBL-ANC, exposing `cgap_sbl` and `cgap_sbl_anc`. Retain the existing `casbl_jadce` Python namespace for import stability and rename the distribution `cgap-sbl-jadce`. This supersedes the former CA-SBL mathematical interpretation in decisions 1–10.

50. General CGAP-SBL uses a finite symmetric interaction Omega, with `lambda = positive_part(Omega @ gamma_previous)`. Lambda is unscaled and frozen during the current penalized M-step. Update `gamma=2*eta/(1+sqrt(1+4*lambda*eta/M))`, recovering SBL exactly for lambda=0. The Gaussian-inspired interaction is not a proper Gaussian hyperprior; no fixed-global MAP/MM monotonicity or convergence claim is made. Keep the final posterior mean unmasked and recompute it from final gamma.

51. ANC uses `Omega_ANC=alpha*(beta*1-C)` without zeroing its diagonal, and computes `lambda=alpha*positive_part(beta*sum(gamma)-C@gamma)` with sparse C rather than constructing dense Omega. Match `rho=kappa`, `U=D` to saved activity generation parameters, replacing rho=7 and U=20. The current saved experiment has kappa=3.78 and D=15. Automatic calibration and activity probabilities remain unchanged.

52. Adopt the revised setup: K=500 and relative gamma change < delta=1e-4; 100 tuning realizations at L=25, SNR 12 dB; alpha=[2,4,6,8,10,12,14], beta=[.05,.10,.15,.20,.25,.30,.35]. Independently tune both activity thresholds. Evaluate Q=1000 with L=20:5:45 at 12 dB and SNR=4:4:24 at L=30. Convergence uses L=30, 12 dB. Use `sigma2`, `lambda_`, `S_q`, `B_q=min(S_q,L,N)` notation. Preserve zero-based stored indices.

53. Integrate the paper's paired spatial-randomization comparison into estimator evaluation and performance evaluation. Uniformly select S_q distinct device identities independently of geometry, retain device/event positions, H, Theta and C, and rescale identical standardized noise to each scenario's prescribed SNR. Keep randomized support fixed across conditions. Compare CGAP-SBL-ANC and SBL using correlated-tuning selections unchanged. Add explicit seed spatial_randomization=6031; preserve all six prior four-digit seeds. Record scenarios and within-scenario paired F1 gain and NMSE reduction. Main correlated comparisons retain all four estimators.

54. Save F1 and NMSE tuning contours and four-curve spatial comparisons. Reject legacy or mismatched tuning definitions rather than reuse them. Preserve existing numerical results, figures and historical manifests; no manuscript performance numbers are fabricated. Existing algorithm tasks remain compatible, while algorithm tasks should be regenerated for the revised schemas and estimator setup. The only standard formats remain PNG and separate native-font paper PDF, as requested in earlier presentation decisions.


55. Move paired support randomization from estimator evaluation into activity generation at the user's request. Save randomized evaluation samples and all their spatial PNG maps in separate randomized_evaluation data/figure folders. Preserve the exact SeedSequence([spatial_randomization,q]) choice sequence, counts and geometry; store conditional marginal probabilities S_q/N. Randomized maps omit event markers, event-influence circles and the event probability field. Tuning remains event-driven only. estimator evaluation loads saved pairs, checks pairing, and generates only communication observations; missing pairs require rerunning activity generation. Summarize randomized figure output by folder, and record the actual saved randomization seed in downstream provenance.


56. At the user's explicit request, replace the remaining active legacy package names: rename src/casbl_jadce to src/cgap_sbl_jadce, migrate imports in source, scripts and tests, and rename local ca/ca_* variables to cgap/cgap_*. Rename the manuscript file to Publications___CGAP_SBL.pdf and update current documentation references. This supersedes decision 49's namespace-retention choice. Preserve recorded historical decisions, results, manifests, equations, seeds, and experiment folder names. The workspace checkout path and remote repository identity are external to this Python package rename.


57. At the user's request, communication generation now owns full communication-data generation and saving. Store shared H and scenario-specific a/Z once per realization, Theta once per L, and Y/W/sigma2 once per unique condition and scenario in compressed NPZ archives under communication/{tuning,evaluation}/{q:04d}/. Preserve existing deterministic streams and paired standardized noise exactly. algorithm tasks load saved inputs and do not generate communication observations. Source activity hashes, communication seeds, dimensions and conditions identify the dataset; incompatible or missing archives require rerunning communication generation. Keep historical preview outputs untouched. This supersedes decision 17 and the communication-generation ownership in decision 55.

58. communication generation automatically saves only selected-realization diagnostics: pilot cross-correlation with diagonal removed for display, effective-channel row energy, and received-signal magnitude. Save each diagnostic for tuning and evaluation; evaluation channel/signal diagnostics show correlated and randomized panels with the same condition, and received magnitudes use a shared color scale. Include q, S_q, L and SNR in titles. Keep code-local typography, PNG defaults, existing --config/--no-progress arguments, and progress during generation. No per-realization diagnostic flood or additional configuration files.


59. At the user's request, use figures.tuning_indices and figures.evaluation_indices as independent zero-based lists selecting activity generation spatial maps and communication generation communication diagnostics. Each evaluation selection includes both correlated and randomized scenarios. Empty lists skip realization figures; indices are validated and duplicates are omitted. Remove selected_sample from supplied configs. Preserve all numerical data, seeds, and full-sample active-count histogram; do not remove previously saved images. This supersedes automatic all-realization spatial figure generation in earlier decisions.


60. Group activity generation and communication generation realization figures into tuning_{index:04d}/ and evaluation_{index:04d}/ folders under each task's figures/. Pair correlated and randomized activity generation maps in the same evaluation folder; preserve distinct filenames and communication generation paired panels. All task CLIs report one figures/ folder and a compact file/realization count summary instead of per-image paths. Keep numerical archive paths, selections, equations and existing saved figures unchanged.


61. Change the main experiment spatial_randomization seed from 6031 to 9097. Preserve the other six seeds, active counts and paired geometry. Regenerate activity and communication and dependent estimator/evaluation outputs; historical manifests remain unchanged. The small test fixture retains its independent 6031 reference stream.

62. At the user's request, replace paired multi-panel communication generation diagnostics with separate evaluation and randomized_evaluation figures for all six diagnostics. Preserve shared scales for Z magnitude, row energy and received magnitude, and group filenames by evaluation index. Add figure-saving progress controlled by --no-progress. Numerical datasets and algorithms are unchanged.


63. Save shared H, Theta and pilot cross-correlation diagnostics only once per selected evaluation pair. Keep separate correlated/randomized Z, row-energy and received-magnitude figures with shared scales. This yields six figures per tuning index and nine per evaluation index, or 81 figures for the current 3/7 selections; supersedes duplicate common exports in decision 62.


64. Add communication generation noise-magnitude heatmaps per scenario with common paired magnitude scales and noise variance/SNR labels. Save a standardized-noise magnitude diagnostic once per pair, since W/sqrt(sigma2) is shared. Preserve all noise arrays and seeds. Counts become eight per tuning index and twelve per evaluation pair (108 for current selections).


65. Simplify communication generation diagnostics to HSV H, Theta, Z, Y and shared standardized W/sqrt(sigma2), plus channel row energy and pilot cross-correlation. H, Theta, standardized noise and pilot cross-correlation are shared per evaluation pair; Z, Y and row energy are separate per scenario. Remove magnitude-only noise/received exports. Raw noise variances remain scenario-specific and unchanged. Theta HSV uses equal aspect for square cells. Counts become seven per tuning index and ten per evaluation pair (91 for current selections).


66. spatial dependence generation owns full C generation and saving, using one compressed sparse CSR matrix per tuning/evaluation geometry and one shared C per evaluation pair. algorithm tasks load saved matrices and reject missing/incompatible source identity, N, rho or U. Keep the existing matrix formula and matched rho=kappa/U=D unchanged. spatial dependence generation heatmaps follow the common selected tuning/evaluation index lists and padded realization folders, with split generation/figure progress and compact CLI reporting. Supersedes the earlier single-matrix preview and downstream recomputation.


67. Use spatial dependence matrix for C throughout active code, documentation and artifacts. Rename C-related modules, functions and metadata consistently, and migrate existing outputs without changing matrix values. Pilot cross-correlation and the correlation-guided algorithm name remain correct.


68. Recheck paper Section VII, equations (75)–(79): implemented Cholesky/diagonal-posterior SBL, dense general CGAP-SBL and sparse CSR ANC match the stated per-iteration costs. Record the full-call runtime scope explicitly, including initialization/validation and final posterior, excluding data/C generation, loading, metrics, plots and saving. Preserve existing per-algorithm runtime_s measurements. Add per-condition/scenario runtime statistics, CPU/thread provenance, and runtime-vs-pilot figure alongside runtime-vs-SNR. Complexity audit distinguishes per-iteration costs, actual iteration counts and complete solves; no paper performance numbers are copied into results.


69. Restrict channel NMSE in algorithm tasks to the true active MTD rows in both the numerator and reference-energy denominator. Score missed active rows, exclude estimates on inactive rows, and keep zero-reference-energy realizations undefined (NaN). Use saved activity masks for tuning and evaluation; convergence history infers support from the true effective channel. This affects scoring only, not solver inputs or detection metrics. Record nmse_scope=true_active_support and invalidate older tuning definitions; do not relabel historical full-matrix scores. data preparation remain reusable; regenerate algorithm tasks for the new metric.


70. On feature/graph-laplacian, remove the randomized-support experiment. Keep the six original seeds and only event-driven tuning/evaluation datasets. Preserve the numerical event-driven communication inputs while converting archives to H/a/Z, per-L pilots and per-condition observations. Retain the original saved files in a verified backup outside the repository during migration. Split historical event-driven scores without changing NMSE values or claiming compatibility.

71. algorithm tasks independently run OMP, CoSaMP, SBL and CGAP-SBL-ANC. SBL owns tau tuning; ANC owns alpha/beta/tau tuning. Shared estimator orchestration handles saved inputs, scoring, per-call runtime and output ownership, with an interface for a later GL method. OMP/CoSaMP have no tuning or C dependency; SBL needs no C. algorithm comparison compares compatible completed results without solver execution; convergence analysis analyzes convergence. Keep only --config and --no-progress. ANC equations, true-active-support NMSE and existing experiment settings remain unchanged.


72. Replace numbered scientific launcher scripts with the cgap-sbl interactive terminal menu (also python -m cgap_sbl_jadce). Up/Down select a registered task, Enter runs only that task, and Q/Esc exits. Restore the normal terminal for task output and progress, return to the menu afterward, highlight the next task on success, and retain the selection on expected failure. Keep --config and --no-progress and the separate paper exporter. Use standard-library curses without adding a dependency.


73. Name plotting modules by their content rather than task numbers: kappa_calibration, activity_realizations, active_mtd_count_distribution, communication_diagnostics, spatial_dependence, anc_parameter_tuning, algorithm_performance and algorithm_convergence. Update imports throughout CLI, paper export and tests; keep plot calculations and rendering unchanged.


74. Group estimator runs under an Algorithms submenu in cgap-sbl. Derive its entries from the estimator registry so a future GL variant is grouped automatically. Keep comparison/convergence in the main menu, preserve independent algorithm execution and result ownership, and provide Back/Left/Q/Esc navigation to the parent menu.


75. Expose Tune and Run as separate submenu actions for tunable estimators (SBL and ANC, and future registered variants). Tune reads only tuning inputs, saves selections and produces tuning figures. Run requires compatible saved selections, performs only evaluation and produces evaluation figures. OMP/CoSaMP retain direct Run actions. After Tune succeeds, highlight Run for the same estimator; preserve action selection after failure. Record the action in task provenance without adding configuration options.


76. Add automatic persistent restart checkpoints for new task runs and Tune/Run actions. Commit calibration batch state/RNG, activity samples, generated communication/spatial output hashes, tuning gamma/metrics, evaluation rows/runtimes after each condition, and completed convergence traces. Resume only compatible settings, source identities, scientific code and numerical environments; repeat the unfinished unit and regenerate final tables/plots from committed numerical work. Write outputs atomically with file/directory fsync and preserve prior complete files after failed writes. Private checkpoints and unfinished temporary files stay outside Git. Historical pre-checkpoint results are not promoted to resumable units.


77. The Streamlit GUI replaces the cgap-sbl terminal task menu. A shared tasks.py registry runs only explicitly selected actions. Named setups start from default/current settings, use isolated result roots and one stable config.yaml, and never overwrite existing setups. Data dependencies permit independent algorithm order and communication/spatial generation order. Selected L/SNR evaluations accumulate compatible completed scores/runtimes and checkpoints; overlapping sweep views share one solver calculation and measured runtime; comparison can use selected conditions without requiring full sweeps.

78. Provide both reference and per-condition SBL/ANC tuning policies at setup creation. Reference tuning preserves the existing default. Per-condition tuning saves the extra tuning communication inputs, keeps tuning/evaluation streams separate, and writes condition-local selections/tables/figures. Runs require only requested selections; changed selections invalidate their cached evaluation rows. Provide fixed-reference sweeps or an explicit full L×SNR grid without changing default simulation settings or solver equations.

79. The GUI renders interactive Plotly charts from numerical data rather than reading saved figures. Preserve actual activity probabilities and HSV magnitude/phase, provide hover/zoom/pan/legend interaction, maintain true-active-support NMSE and log scales, and label historical score scope. Native-font publication exports are regenerated from numerical data by the existing Matplotlib functions. Existing saved figures are retained as artifacts and are not GUI inputs.


80. Replace ordinal task identifiers and output-folder prefixes with descriptive names throughout the active workflow. Migrate existing folders, figure filenames and provenance references without changing numerical arrays or measured scores. The GUI overview uses clickable preparation, algorithm and analysis cards with completion indicators and current progress; SBL/ANC have Tune/Run child cards. Task pages include an interactive graph generator using numerical data.

81. Allow one computing worker and multiple cooperatively paused workers. Wait for confirmed pause before starting another task, return to the original action to Continue, and reject continuation if required inputs, configuration or scientific code changed while paused. Restart uses compatible checkpoints after interruption.

82. Compare selects any subset of algorithms with compatible completed results for the requested conditions, including one algorithm. Convergence selects only completed SBL/CGAP-SBL-ANC runs at the chosen evaluation condition, requires their tuning selections, and requires C only for spatial methods. OMP/CoSaMP are excluded. Analysis no longer requires all algorithms or comparison before convergence.
