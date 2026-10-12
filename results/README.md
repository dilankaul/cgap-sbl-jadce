Saved experiment outputs are versioned in Git. `experiment/` contains experiment data, summaries, configuration snapshots, and task figures, generated with `configs/experiment.yaml`.


The source now implements the revised CGAP-SBL manuscript. Existing outputs are
historical and retain the method/configuration recorded in their manifests;
they are not relabeled as CGAP-SBL measurements. Task-01/02 activity data remain
compatible. Regenerate Tasks 03–09 with `configs/experiment.yaml` for the revised
noise/penalty notation, matched ANC parameters, relative stopping criterion,
and paired spatial-randomization comparison. Task-local manifests record the
actual effective configuration; the root YAML remains the activity snapshot.


Task 03 now saves full communication data under 03_communication_data/, with
per-realization shared channels/pilots and scenario observation archives. The
older 03_communication_preview/ folder is historical. Tasks 05–08 require the
full dataset. Task 03 saves six selected-sample diagnostic PNGs, not figures for
every communication realization.


Task 04 stores complete sparse C datasets in 04_spatial_dependence/tuning/ and
evaluation/, with only selected heatmaps under padded realization figure folders.
The prior preview_C.npz artifact is historical; consumers require the full dataset.
