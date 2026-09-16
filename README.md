# Temporal EEG State Knowledge Graphs for Patient-Specific Seizure Forecasting

Code and summary results for the study:

**Temporal EEG State Knowledge Graphs for Patient-Specific Seizure Forecasting: A Chronological Continual-Learning Study**

## Overview

The study evaluates whether patient-specific temporal relationships between recurring EEG connectivity states add useful information to strictly chronological seizure forecasting. The pipeline combines:

- 10-s multichannel EEG windows with a 5-s stride;
- five log-band-power node features per channel;
- signed Pearson functional connectivity across 22 bipolar channels;
- a compact signed-connectivity GNN;
- chronological fine-tuning and replay baselines;
- patient-specific temporal EEG state knowledge graphs;
- knowledge-graph-guided replay and direct prediction guidance.

The reported analysis uses three eligible CHB-MIT patients (chb06, chb09, chb10), ten future seizures, and 42.077778 h of clean future EEG.

## Repository structure

```text
notebooks/
  01_protocol_audit.ipynb
  02_experiment_pipeline.ipynb
results/
  cohort_summary.csv
  continual_and_kg_replay_summary.csv
  replay_memory_diversity.csv
  direct_kg_summary.csv
  direct_kg_patient_summary.csv
  kg_ablation_differences.csv
requirements.txt
.gitignore
```

Notebook outputs are cleared in the public copies to keep the repository compact and avoid mixing exploratory logs with the final pipeline. The paper-level numerical summaries are provided in `results/`.

## Data

The CHB-MIT Scalp EEG Database is not redistributed in this repository. Download the original dataset from PhysioNet and update `DATA_ROOT` in the notebooks to match its location.

The notebooks were developed for Google Colab and Google Drive. The project paths can be changed to local paths if the same directory structure is retained.

## Run order

1. Run `notebooks/01_protocol_audit.ipynb`.
   - Audits recording coverage and channel compatibility.
   - Produces the frozen `protocol_v2` artifacts.

2. Run `notebooks/02_experiment_pipeline.ipynb`.
   - Builds the graph-feature cache.
   - Performs full-cohort screening and the final chronological split.
   - Fits patient-specific normalization using initial training data only.
   - Runs the GNN continual-learning baselines.
   - Builds the temporal EEG state knowledge graphs.
   - Runs KG-guided replay and direct KG-guidance experiments.

Future episodes follow a strict **predict -> evaluate -> reveal -> update** order.

## Reproducibility checks

Expected hashes from the reported experiments:

- Protocol v2: `0f82325c0bf85962ee7bdc92a7bc5d25c4fa80d404f625a8b4f199b42c966c90`
- Preprocessing: `3999e8ee6592657ce0a7c20a682b7c29b95420926bc6fbe0962ba9763f67a107`
- Final split v3: `a0db400259564448fd17d5f71547c1431455d4498999faf5bdb96788abab64a2`

The final fixed montage contains 22 bipolar channels. The seizure prediction horizon is 5 min, the seizure occurrence period is 30 min, and clean interictal EEG is at least 4 h from every annotated seizure.

## Main reported results

The standalone GNN achieved the highest pooled future AUPRC (`0.130025`). Direct temporal KG guidance achieved `0.117821`, compared with `0.100170` for state-only KG guidance. The temporal-minus-state AUPRC difference was positive in all five direct-guidance runs, with a mean difference of `0.017651`.

The KG-guided methods did not outperform the standalone GNN overall, and the false-alarm burden remained high. See the CSV files in `results/` for the complete paper-level summaries.

## Notes

- Five optimization seeds quantify training variability and are not independent clinical replicates.
- The 128-example replay value is an update/replay-selection budget. Candidate selection can access the accumulated revealed history.
- Knowledge-graph state centroids are learned from the initial historical training data and remain fixed during future evaluation.
- Temporal transitions are counted only between consecutive windows in the same EDF file at the expected 5-s stride.
