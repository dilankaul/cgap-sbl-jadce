Saved experiment outputs are versioned in Git. `experiment/` contains experiment data, summaries, configuration snapshots, and task figures, generated with `configs/experiment.yaml`.


The source now implements the revised CGAP-SBL manuscript. Existing outputs are
historical and retain the method/configuration recorded in their manifests;
they are not relabeled as CGAP-SBL measurements. Task-01/02 activity data remain
compatible. Regenerate Tasks 03–09 with `configs/experiment.yaml` for the revised
noise/penalty notation, matched ANC parameters, relative stopping criterion,
and paired spatial-randomization comparison. Task-local manifests record the
actual effective configuration; the root YAML remains the activity snapshot.
