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


83. Unify Continue and Restart as Resume: continue a live paused worker or recover a stopped, failed or interrupted action using compatible checkpoints. Keep Pause and Stop, input guards, and one computing worker.

84. Split algorithm evaluation results by L/SNR under `<method>/run/Lxxxx_snrXdB/`; retain algorithm-root sweep tables as derived combined views. Individual completed conditions accumulate to full algorithm completion. Keep All conditions and Selected conditions. Start/Rerun calculates requested conditions afresh; only Resume reuses unfinished units and skips conditions already completed by its request. Preserve unrelated completed data, reference/per-condition tuning choices, true-active-support NMSE, solver timing, numerical definitions and seeds.

85. Version unfinished checkpoints and portable recovery requests, excluding local PIDs/logs/exports and atomic temporary files. Clear only verified checkpoint owners after numerical results and required figures commit; preserve unfinished recovery after pause, stop, failure and interruption. Commit final-output metadata before cleanup so a shutdown during cleanup can recover without recalculation. Disjoint algorithm conditions may run while another condition is paused. Compatible legacy combined results migrate without running solvers; historical scores remain historical.


86. Distinguish Start, Resume and explicit Rerun labels using saved state. Algorithm Run offers Complete all and Complete selected, which skip compatible finished conditions, restore unfinished checkpoints and start untouched conditions; show counts before launching and reject incompatible progress rather than silently rerun it. Keep separate explicit Rerun selected/all actions and shared Pause/Resume/Stop controls. A broader completion request adopts confirmed paused work at its durable boundary; an exact Resume continues its live worker. Retain condition selection, one computing worker, numerical definitions and existing checkpoint identities.


87. Group editable setup settings by task and algorithm. ANC owns alpha/beta/tau tuning, SBL owns tau tuning, and each algorithm has independent solver options, tuning policy and reference condition. Preserve flat-config reading and migrate layouts only when saving an edit or creating a setup. Use fixed-reference sweeps or explicit L/SNR pairs; omit the GUI Cartesian-grid option. Central and task-local editors share one source config, preview dependency-specific effects and save atomically without deleting outputs.

88. Recompute compatibility from relevant settings, actual condition input arrays, scientific method identities and recorded selections. Ignore YAML formatting, unrelated algorithm options, figure selections and other condition lists when validating a condition. Restore eligibility when matching values and original data return. Show incompatible results while preventing downstream scientific use; retain historical figures from recorded data. Protect active worker settings, permit edits at confirmed pause boundaries, and validate scoped semantic identities before Resume. No publication experiments are run by the setup restructuring.


89. At the user’s request, use 1e-4 as the default stopping tolerance for SBL, CGAP-SBL-ANC and CoSaMP. CoSaMP retains its relative-residual criterion, with its default iteration limit increased to 500; SBL-family methods retain their relative-gamma-change criterion and configured iteration limits. Existing named setups and recorded result provenance retain their explicitly saved values.


90. Use inclusive start/stop/step controls for ANC alpha/beta, the initial kappa scan, evaluation pilot lengths and evaluation SNRs, matching tau. Preserve existing default candidate values exactly; retain separate kappa bracket-refinement steps and explicit L/SNR condition selection. Read legacy explicit grids without changing their values; convert regular grids exactly, and retain nonuniform grids until the user explicitly replaces their range. Decimal-step expansion avoids floating endpoint drift; pilot-length ranges require integers.


91. Experiment setup and task settings editors expose only fixed-reference pilot-length/SNR sweeps. Optional additional L/SNR pairs are registered on an algorithm Run page, outside the source YAML. Explicit Prepare inputs extends compatible saved communication data using condition-local archives, preserves base arrays and bytes, and checkpoints each realization. Reference tuning reuses its selection; per-condition tuning separately tunes the added pair before evaluation. All/Complete all and parent completion count only setup sweeps; optional runs require explicit selection and remain available to comparison/convergence. Supersedes selected-pair setup controls in decision 87; retain legacy config reading.

92. Setup creation/editing and task settings require explicit Save buttons with Enter submission disabled. Reset restores saved/source values without writing configuration or launching tasks. Discard unsaved form drafts when navigating away, changing setup or changing editor context. Preserve active-worker settings guards and reversible result compatibility. Clearly label ANC alpha-beta tuning L/SNR independently of the SNR held fixed during the pilot-length sweep and the L held fixed during the SNR sweep; scientific values and solver equations are unchanged.


93. Optional-condition entry fields start empty. Save/delete on any algorithm Run page updates one shared optional-condition registry and all algorithms’ condition options/status tables. Label these rows Optional and exclude them from parent sweep completion and All/Complete all. Keep algorithm outputs independently owned. Removing an option retains existing numerical data and results; prevent removal while a live/paused task uses it. Read earlier method-local registrations as shared options.


94. Configure optional conditions directly in the algorithm Run page’s Condition status table: an empty L/SNR input row with Add, and Delete on each optional row. Add/Delete apply immediately, with no additional Save control. Required sweep rows remain read-only and have no Delete action. Preserve shared options, input preparation, active-worker guards, saved numerical data and sweep-only completion totals. Setup and task-settings Save/Reset forms are unchanged.


95. Clear all saved experiment runs, named result setups, figures, checkpoints, recovery requests and local GUI job/export state for a fresh start. Retain the source experiment YAML and documentation. Remove legacy result migration, legacy preparation-metadata fallback, old combined-result eligibility, and historical-score UI handling; current condition results and strict input/settings/code compatibility remain mandatory. Saved tuning outputs from current runs remain viewable after later setting changes. Shorten the dashboard to controls, relevant settings/progress and necessary error messages. Remove narrative code comments/docstrings and keep only concise scientific notes; rationale and future work belong in project documents.

96. Reuse unchanged communication-array digests, output hashes and condition tables with bounded in-memory caches keyed by file size, modification time and change time. Share validation only within a GUI refresh, and stat the inputs again on the next refresh. Preserve existing content identities and setting/code compatibility checks; changed files invalidate caches, and restoring the original content restores eligibility. Reuse a task page's status and selected-condition plan rather than validating them again for graph controls. No scientific settings or calculations change.

97. The GUI log reader strips terminal cursor/color sequences before extracting complete tqdm frames for the graphical progress display. Discard partial frames at the bounded tail/read boundary, and preserve warnings or pause/status messages appended to a frame. Keep raw worker logs intact; do not change scientific loops or worker calculations for display formatting.

98. Poll saved preparation summaries, algorithm condition summaries, tuning selections and shared condition registration alongside worker status. Refresh the page when those file versions change, including during a still-running multi-condition request, so completion counts, status rows, controls and available graphs advance after each saved condition. Inspect metadata only on the timer; do not scan numerical arrays or per-realization checkpoints to detect UI updates. Continue validating numerical compatibility when rendering results. During a normal page render, update the watcher baseline without requesting another rerun, so result updates cannot swallow navigation clicks.

99. Fresh algorithm Run/Rerun validates every requested condition before clearing its known result files, native condition plots and owned checkpoint units. Retire the entire requested scope before the first solver call; All excludes optional conditions. Rebuild combined tables from surviving compatible results and preserve input archives, tuning selections, other algorithms, unselected conditions and unrelated files. Validate condition paths and checkpoint ownership, and reject linked paths outside that scope. Persist reset_pending in portable recovery and a scoped reset journal during cleanup; Resume finishes interrupted cleanup before calculating, then retains all newly completed work. Supersede overlapping older recovery requests without deleting unrelated recovery records. Conditions without results display Not started without placeholder result files; incomplete resets require Resume of their original request, or a fresh rerun covering that entire scope.

100. At the user's request, make fixed-S activity the default research setup: set `activity.exact_sparsity: true` in configs/experiment.yaml, with system.S=20. Use the existing rejection sampler to condition event-driven tuning and evaluation realizations on exactly S active MTDs. Kappa calibration continues to target the unconditioned mean before conditioning. This supersedes the default unconditioned activity policy in decision 13; retain explicit variable-activity support and feasible oracle budgets. Preserve seeds, solver definitions, existing setup settings and saved provenance. Regenerate activity and dependent communication, spatial dependence, tuning and algorithm outputs before using fixed-S results.

101. Configure dashboard realization defaults with figures.tuning_main_index and figures.evaluation_main_index, initially 4 and 458. Expose both in central and preparation-task Figure selections settings with explicit Save. Share defaults across activity, paper activity, communication and spatial previews; retain manual per-view index changes until the configured default changes. Fall back to the first selected figure index, or zero for empty/all selections, and bound display indices to the saved dataset. Keep these presentation settings outside scientific compatibility and checkpoint code identities. Automatic export index lists remain independent.

102. At the user's request, use explicit ridge regularization for CoSaMP in both merged-support and final pruned-support fits. Configure algorithms.cosamp.solver.cosamp_lambda_reg, default 0.01 as in the linked casbl-for-jadce-in-mtc solver; zero restores unregularized fitting. Expose the nonnegative finite parameter in experiment setup and CoSaMP task settings, and retain it in method definitions and manifests. Solve the ridge objective ||Y-A X||_F^2 + lambda_reg ||X||_F^2 by augmented SVD least squares, avoiding normal equations and explicit inverses. Retain joint-row support selection, feasible oracle budgets, relative residual tolerance and iteration limits. This supersedes decision 14's unregularized default. Preserve old results and reject their downstream use through existing settings/code compatibility checks; do not rerun experiments automatically.

103. At the user's request, communication generation always saves independent tuning inputs for every configured evaluation sweep L/SNR pair, plus the algorithm tuning references and communication preview reference. This input coverage no longer depends on selecting reference versus per-condition tuning. Preserve both tuning policies, explicit optional-condition preparation, activity and communication seeds, deterministic per-condition realizations and numerical definitions. Use one condition planner for generation and worker settings guards. Preserve saved datasets and algorithm outputs; older datasets remain usable for their present compatible conditions and report missing coverage until communication generation is explicitly run. Do not regenerate experiments automatically.

104. Mirror the shared optional-condition registry on the Communication page, alongside setup conditions, with tuning/evaluation input status and Complete selected/all missing inputs controls. Communication-owned selected preparation has no estimator selection and prepares both splits where required; algorithm-page shortcuts retain their method-specific split requirements. Extend inputs in condition-local archives, skip compatible completed input conditions, support worker pause/resume and checkpoints, and leave base archives, completed algorithm outputs and solver runtimes untouched. Keep optional registration on algorithm Run pages and optional pairs outside default sweep completion; no automatic estimator runs.


105. At the user's request, give Communication the algorithm Run condition table, shared Add/Delete row controls, All/Selected scope, Start/Complete/Resume/Rerun actions and one combined condition status. A completed communication condition includes both independent tuning and evaluation datasets, including setup sweep and tuning/preview reference pairs. Optional input preparation from either page always creates both splits without running estimators; optional pairs remain outside All/Complete all. Support selected starts from an empty communication dataset, condition-local incremental inputs, per-realization atomic checkpoints, compatible completion and scoped PNG artifacts. Preserve shared input arrays, unselected conditions and completed algorithm results. Explicit full reruns retain incompatible shared datasets under communication/.retired before replacement. Clear current experiment results and local GUI state for the authorized fresh start, preserving the source YAML; do not run experiments automatically. Supersedes the split status and method-specific input shortcuts in decision 104.


106. At the user's request, remove the additional paper activity realization figure, its dedicated renderer/export script, dashboard category, and feature-specific tests. Retain the standard activity realization and active-count figures, interactive activity chart, and standard saved-data PNG/vector-PDF publication exports. No activity probabilities, experiment settings, seeds, or scientific results change. This supersedes the separate paper activity export in earlier figure decisions.


107. At the user's request, set algorithms.sbl.tuning.selection_mode and algorithms.anc.tuning.selection_mode to per_condition in the source experiment YAML and current results/experiment/config.yaml. Use the existing condition-specific tuning workflow: SBL selects tau independently and ANC selects alpha, beta and tau independently for each configured L/SNR pair, with optional pairs tuned when explicitly selected. The current sweeps contain 11 unique conditions. Retain parameter grids, solver settings, seeds, independent tuning/evaluation data, and reference-policy support. Do not run tuning or evaluation automatically. This supersedes the reference default in earlier tuning-policy decisions.


108. At the user's request, show the shared Condition status table on per-condition SBL/ANC Tune pages. List setup and shared optional L/SNR pairs with tuning-specific Completed, Unfinished, Incompatible and Not started states, plus setup tuning completion counts. Validate completed selections and saved output hashes; distinguish compatible unfinished checkpoints/recoveries from incompatible progress. Refresh when condition tuning summaries commit. Keep the Tune table read-only, with shared condition registration on Run/Communication pages; preserve existing tuning scope/actions, solver behavior and saved results. No experiments run automatically.


109. Implement the approved two-stage ANC tuning plan: seeded sequential Optuna TPE, 50 total Stage-1 trials on the same first 20 saved tuning realizations, top 10 distinct alpha/beta pairs reevaluated on the first 100, and one tau optimized across each stage's realizations. Reuse first-stage gamma, NMSE and solver timings in Stage 2. Use continuous linear alpha=0..10 and beta=0..0.1, existing tau=0..0.5/0.001, ten startup trials included in the 50, no pruning, and an alpha=0 baseline included in the budget. Add explicit optimizer seed 17389 under ANC settings; preserve all six activity/communication seeds. Select by mean F1, lower true-active-support NMSE, then recall. ANC Run accepts only Stage-2 Optuna selections. Persist SQLite history and an atomic checkpoint containing sampler state, full trial history and any pending trial; rebuild the database from that snapshot on Resume so a crash between ask/tell and commit preserves the suggestion sequence. Save candidate-level scoring checkpoints as well as individual solver units. Include Optuna version and tuning source identities in compatibility. Keep old data without migration or automatic reruns.

110. Retain grid search as the independent Analysis task ANC parameter grid study, with one configurable fixed L/SNR pair, default 30/12 dB, 100 tuning realizations, its own alpha/beta/tau grids, checkpoints, numerical tables and heatmaps under anc_grid_study/. It does not write selected.json for algorithms. Keep paper-grid settings outside ANC TPE/run compatibility and TPE settings outside paper-grid compatibility. Generate each tuning/study PNG once and collect already committed paths in task orchestration.

111. Enable the shared Add/Delete condition row and optional-input preparation on per-condition Tune pages, using action-specific widget keys while sharing the condition registry across algorithms and Communication. Preserve optional conditions outside All and overall sweep counts. This supersedes the read-only Tune table choice in decision 108.

112. Adopt affected-test checks for small GUI/configuration edits and full local checks for scientific/tuning/recovery changes, with full CI checks before merging. Register unit/gui/integration/plotting groups, add Make targets and pytest-xdist with two isolated workers, and profile full runs with --durations=20. Replace image encoding only in workflow logic tests; dedicated plotting tests retain real PNG/PDF exports, and worker integration still exercises production rendering. Preserve all numerical/reference correctness and recovery checks.


113. At the user's request, replace two-stage ANC tuning with full Optuna TPE: 50 total trials, each scored on the same first 100 saved tuning realizations. Optimize one tau per trial using the existing threshold grid and mean F1, without additional solver calls. Keep ten startup trials and the zero-penalty baseline within the budget, continuous alpha/beta bounds, optimizer seed, condition-specific streams and winner tie rules. Remove finalist settings, reevaluation, files and plots; show trial progress and retain per-solver/per-trial atomic checkpoints, sampler/history replay and all compatibility guards. ANC Run accepts only full-TPE selections. Keep the independent fixed-condition paper grid unchanged. Remove current ANC tuning results, figures, checkpoints and tuning recovery/job records while preserving preparation data and other algorithm outputs; do not launch research experiments automatically. This supersedes decision 109's two-stage search.


114. Match Tune and Communication condition tables to algorithm Run: SNR sweep first, then pilot-length sweep, one row per shared condition, required references afterward and optional rows last. Use Sweep for sweep rows, Reference only for required pairs outside the sweeps, and Optional for shared additions. Preserve canonical full communication-generation/archive order independently of the display order. Per-condition Tune uses the shared Start/Complete/Resume/Rerun controls and worker completion dispatch. Complete skips compatible finished selections and reuses solver checkpoints; explicit Rerun validates the complete scope before clearing only owned tuning outputs/checkpoints/figures. Journal unfinished reset cleanup, block downstream selection use during reset, preserve unrelated files and conditions, and supersede only overlapping recovery requests.

115. Retain the three ANC tuning diagnostic plots and independently save six alpha/beta/tau-versus-SNR/pilot-length plots from compatible completed selections, plus SBL's two independent tau sweeps. Use configured sweep axes, gaps for incomplete/incompatible points, and numeric point labels. Update PNGs after each completed tuning condition; provide separate interactive sweep figures and native-font PNG/PDF exports without solver calls. Annotate ANC diagnostic plots with selected alpha/beta/tau and condition, and highlight the winning trial/pair. Keep paper-grid outputs separate. At the user's request clear the current unfinished ANC tuning condition, its checkpoints, recovery and stopped job record, preserving preparation data and completed results. No research experiments run automatically.


116. Add CGAP-SBL-GL as the fifth estimator alongside OMP, CoSaMP, SBL and ANC.
Use xi_i=sum_j C_ij including C_ii=1, L_norm=I-Xi^(-1/2) C Xi^(-1/2), and
lambda=alpha*positive_part(L_norm@gamma_previous). Precompute inverse square-root
degrees and use one sparse matvec per iteration. Reject non-finite, asymmetric,
negative-weight or zero-degree graphs; saved model graphs have positive self
connections. Preserve the shared posterior, stable gamma root, relative stopping
criterion and final unmasked posterior mean. The normalized null vector is
sqrt(xi), not generally 1; positive-part lagging is adaptive shrinkage, without a
new fixed-global objective or convergence guarantee. Do not change ANC solver,
ANC TPE implementation, existing seeds, preparation data or recorded selections.
GL owns alpha/tau tuning (no beta), separate full TPE with 50 trials x the same
100 tuning realizations, ten startup trials, optimizer seed 17389, linear alpha
bounds initially 0..100 and a zero-penalty first trial when alpha_low=0. Retain
independent solver/tuning settings and condition-specific selection by default.
Use existing ranking, tau scoring and crash-consistent sampler replay. Store GL
outputs under cgap_sbl_gl/, with Tune/Run condition Add/Delete, Complete, Resume
and scoped Rerun. Include comparison and convergence, three diagnostic plots,
four alpha/tau sweep plots, standard performance/gamma plots, interactive views,
PNG previews and saved-data native-font PDF exports. Implement and test outside
the live workspace while ANC is computing, then apply only after it completes.


117. Rename the public ANC algorithm to CGAP-SBL-GLC (Global–Local Contrast),
distinct from the fifth estimator CGAP-SBL-GL. Update dashboard cards, task and
settings pages, selectors, diagnostic/performance/convergence figures, progress
views and reports. Retain the `anc` task/configuration keys, solver API, storage
paths and recorded scientific method identity. This is a presentation rename;
no numerical code, seeds, parameters, saved result bytes, selection digests,
checkpoint identities or preparation data change. Existing stored figures retain
their original provenance; interactive views and regenerated exports use GLC.
Do not migrate results or rerun research experiments for a naming change.


118. At the user's request implement six CGAP spatial variants GLC/NGLC/RGL/GL/GD/AGA,
excluding optional MGLC. Canonicalize ANC to glc in APIs, task/configuration keys and
future cgap_sbl_glc paths; this supersedes decision 117's presentation-only identity
and decision 116's restriction on changing ANC search code. Preserve old saved numerical
files on disk without migration and reject incompatible old selections/checkpoints.
Reuse the unchanged posterior, penalized M-step, convergence rule and detection.
Use J=11^T/N, xi=C@1, Xi=diag(xi), F=Xi^(-1/2) C Xi^(-1/2), retaining C's diagonal.
GLC uses alpha*(beta*11^T-C); NGLC alpha*(beta*J-F); RGL alpha*(I-kappa_g*F);
GL alpha*(I-F); GD alpha*(I-T_t), T_t=exp[-t*(I-F)]; AGA alpha*(beta*J-F_AGA_k).
Use kappa_g for graph coupling (0<=kappa_g<1), distinct from activity kappa; GL is
the boundary at 1. Use T_t, not channel H, for diffusion, and A_k, not noise W,
for AGA affinity. AGA rebuilds weighted affinities and normalized degrees from preceding
gamma; floor gamma only inside logarithms, epsilon_gamma=1e-12 by default. GD uses
sparse expm_multiply, no dense exponential. Zero alpha recovers SBL exactly; GD also
at t=0. Preserve shared E/M updates and do not assert a fixed-global convergence guarantee.
All variants independently use full TPE 50 x 100, ten startup trials, the existing
within-trial tau grid, deterministic condition streams and durable optimizer/solver
checkpoints. Keep GLC's paper grid separately owned, canonical glc_grid_study.
Editable initial bounds: GLC alpha 0..10 beta 0..0.1; NGLC alpha 0..100 beta 0..2;
RGL alpha 0..100 kappa_g 0..0.999; GL alpha 0..100; GD alpha 0..100 t 0..10;
AGA alpha 0..100 beta 0..2 h 0.1..5. These are declared search choices, not tuned claims.
Keep seeds, dimensions and saved data definitions unchanged. Nine algorithm cards use
three columns; every spatial method has its own settings and full condition controls.
Retain all previous diagnostics and provide each tuned parameter/tau sweep, AGA alpha-h
samples, selected-value annotations, performance/gamma/convergence, interactive views
and PNG/native-font PDF exports from saved data. No research runs or result deletion.


119. Add a GLC tuning-only posterior row-norm threshold diagnostic after selecting the
winning alpha/beta using the existing gamma-based full TPE. Sweep tau_Z=0..1
inclusive at 0.001 spacing on norm(mu_i)/sqrt(M), with >= detection and the same
saved tuning realizations. Save mean precision/recall/F1 and the best diagnostic
threshold independently; leave Run detection and selected gamma tau unchanged.
Cache row norms with solver units to avoid additional solves. Save CSV/JSON, PNG,
interactive Plotly and saved-data native-font PDF exports; include diagnostic
outputs in completion hashes and scoped tuning cleanup. At the user's request
remove current algorithm-related results and job records, preserving preparation
data, setup configuration and shared optional conditions. No experiments launched.

120. For the selected GLC alpha/beta and tuning condition, save original and
estimated effective-channel row energies sum_m |Z_im|^2 for configured
figures.tuning_indices within the tuning sample count. Cache estimated energies
with solver units and save selected-realization arrays for PNG, Plotly and
native-font PDF regeneration without additional solves. Annotate realization,
condition and selected parameters; use final unmasked posterior estimates.

121. Extend original/estimated effective-channel row-energy comparisons to all
algorithm Run results at each condition using figures.evaluation_indices. Save
selected energies with atomic solver checkpoints and condition-local NPZ output;
regenerate PNG, Plotly and native-font PDF without solver calls. Include condition
and saved method parameters; use unmasked posterior means for SBL-family methods
and native channel estimates for greedy baselines. Own energy files/figures in
scoped Run cleanup and completion hashes. Increment evaluation format revision.

122. Extend tuning energy comparisons to SBL and all six spatial methods, using
each method's final winning parameters on configured tuning realizations. The
additional 0..1 posterior row-norm threshold diagnostic remains GLC-only.

123. Extend the tuning-only 0..1 row-norm threshold sweep to SBL and all six
spatial variants, each at its own winning tuning parameters. Preserve existing
gamma-based tuning selection and Run detection. Greedy OMP/CoSaMP retain oracle
support budgets and have no tuning stage. Supersedes the GLC-only scope above.


124. At the user's request add pooled top-S tuning as the default experiment
policy, superseding the per-condition default and within-trial threshold objective
for this policy. Independently editable communication.grid prepares the Cartesian
L=20:5:45, SNR=4:4:24 grid for both saved splits. Evaluation retains independent
fixed-reference sweeps and optional conditions. Independently editable tuning_grid
uses L={20,30,40}, SNR={4,12,20}; assign the existing 100 unique tuning indices once,
11 per cell and 12 at the center. Use an explicit deterministic child of the
existing communication tuning SeedSequence; do not alter activity/communication
input streams or seeds. General grids balance sample counts with center-first
remainders and require at least one sample per condition. Every method/trial uses
the same assignment. Each spatial method has one independent full TPE study and
one parameter set for all evaluation pairs. Retain configured trial budgets and
bounds; pooled sample count is the setup tuning count, not a method subset.
Select top configured integer S by final gamma for mean-F1 scoring (stable device
index ties); use NMSE then recall for winner ties. Known S is scoring-only, never
an SBL solver input. With variable activity this is a fixed target-S ranking rule,
not access to each realization's true active count. SBL prepares its final tuning
outputs once without a parameter search. Preserve solver equations and stopping.
After choosing pooled spatial parameters, calibrate gamma and normalized Z-row
thresholds from cached winner outputs; thresholds never select trial parameters.
Keep reference/per-condition policies and the independently owned paper grid.
Save assignment, per-cell scores and winning complex estimates/gamma, plus TPE
history and calibration diagnostics. Pooled folders own separate recoverable
resets/checkpoints; preserve existing result bytes without migration.
All new evaluation conditions save every final unmasked complex Z_hat and every
SBL-family gamma, masks and realization IDs; shared saved inputs supply ground
truth and their identities remain in manifests. Save unmasked NMSE and NMSE after
zeroing undetected rows, always on true-active support. Comparison can rescore
from these archives using gamma >= tau, norm(Z_hat_i)/sqrt(M) >= tau_Z, or top-K
(gamma or Z ranking), and can choose either estimate for NMSE, without solvers.
Record shared/per-method analysis settings. Greedy methods have no gamma and keep
native oracle support under the common gamma rule; Z thresholding requires an
explicit threshold, while top-K uses their channel rows. Threshold/K/estimate-mode
changes invalidate only analysis, never completed solver runs or checkpoints.
GUI settings retain explicit Save/reset/navigation behavior. PNG/Plotly/native-font
PDF diagnostics regenerate from saved data. Version new result contracts and
reject old incompatible selections/checkpoints; do not launch research runs.


125. At the user's request use alpha=0.001..1000 logarithmic for every spatial
method; beta=0..5 linear where present; RGL graph_kappa=0..1 linear; GD diffusion
time=0.01..10 logarithmic; AGA affinity bandwidth=0.01..10 logarithmic. Every tuned
parameter has independent editable lower/upper bounds and a boolean <parameter>_log
under its algorithm, exposed in setup/task forms with explicit Save. Validate boolean
flags and strictly positive lower bounds for logarithmic domains. Linear alpha=0
and diffusion time=0 remain valid scientific choices; h must always be positive.
Existing explicit zero lower bounds without a log flag imply linear sampling.
RGL's mathematical domain remains 0<=kappa_g<1, distinct from GL at 1. Accept the
user's nominal upper setting 1 and sample with effective high=nextafter(1,0);
never send kappa_g=1 to its solver. Save actual FloatDistribution bounds/log flags
in search_space.json and the Optuna database, guarded by owned completion hashes,
recoverable cleanup and selection/checkpoint identities. Parameter-sample figures
use logarithmic axes where the selected sampling flags request them. Preserve
independent trial budgets, seeds, posterior/penalty equations and convergence.

126. At the user's request clear all current generated experimental results,
including preparation data, algorithm results, plots and checkpoints. Retain the
source configuration and any named setup definitions. Clear the active results
folders through a recoverable /tmp backup; do not restore results or migrate them.
GUI job history remains separate from experimental numerical results. Do not
launch new research computations as part of implementing or validating the changes.
