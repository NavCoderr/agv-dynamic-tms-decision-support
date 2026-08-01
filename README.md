# Planning-to-Execution Runtime Observability for Manufacturing AGVs

This repository contains the datasets, processing scripts, validation
pipelines, result tables, and figures supporting the study:

**Planning-to-Execution Runtime Observability and Supervisory State
Recognition for AGV Traffic Management in Manufacturing Intralogistics**

## Overview

The project studies how an upstream AGV transport plan can be used as a
runtime reference during physical execution.

The evaluated workflow combines:

- a directed shop-floor graph;
- a selected AGV route and time-indexed planned trajectory;
- physical AGV telemetry acquired through OPC UA;
- one software-simulated traffic participant;
- current spatial and corridor relations;
- short-term interaction dynamics;
- TMS, safe-node, scanner, urgent-control, and operator evidence used for
  offline supervisory-state reconstruction.

The principal task is binary supervisory-state recognition.

```text
Non-risk:
    normal
    resume

Requires supervisory attention:
    operator_stop
    deadlock
    fallback_hold
    safe_node_hold
    preentry_block
    conflict_risk
```

The principal evaluation uses a corrected 21-variable event-free
representation. It excludes direct TMS, scanner, safety, urgent-control,
and operator-event variables. It also excludes the binary
`close_distance_flag`, `very_close_flag`, and `opposite_edge_flag`
variables involved directly in deterministic conflict-risk
reconstruction.

The model output is intended as execution-state evidence for an existing
TMS or operator. It is not a certified collision-avoidance function and
does not issue vehicle commands, change traffic reservations, release
held AGVs, or override PLC, scanner, safety, or operator authority.

## Evaluated Environment

The repository contains data from three controlled hybrid
physical--simulated AGV runtime experiments.

```text
Physical participant:
    One Navitrol-controlled laboratory AGV

Physical data interface:
    OPC UA

Virtual participant:
    One software-simulated AGV using the same coordinate frame and
    shop-floor graph identifiers

Runtime resolution:
    1 Hz

Complete runtime runs:
    3

Total synchronized samples:
    3279

Combined recorded duration:
    3276 s
```

The experiments support controlled manufacturing-intralogistics
feasibility and complete-run validation. They do not constitute a fully
physical multi-AGV fleet deployment or factory-scale validation.

## Repository Structure

```text
dataset1/                                  Runtime logs for Dataset 1
dataset2/                                  Runtime logs for Dataset 2
dataset3/                                  Runtime logs for Dataset 3
dynamic_outputs/                           Processed datasets, results, and figures
dynamic_tms.py                             Main dataset-construction pipeline
dynamic_analysis.py                        Secondary result and figure generation
extra_validation.py                        Secondary bootstrap and calibration analysis
jim_final_validation.py                    Principal corrected 21-variable validation
planning_reference_observability_audit.py  Planning-reference observability analysis
requirements.txt                           Fixed Python package versions
```

## Runtime Data

Each dataset folder contains logs from one hybrid
physical--simulated runtime experiment.

Typical files include:

```text
agent_live_trajectory_log_v145.csv
dynamic_decision_log_v99.csv
dynamic_training_log_v99.csv
dynamic_edge_traversals_v111.csv
safe_node_tms_log_v146.csv
tms_runtime_action_log_v143.csv
series_leg_trajectory_sample_log_v98.csv
series_leg_plan_log_v99.csv
series_mission_log_v99.csv
hard_stop_log_v141.csv
emergency_event_log_v99.csv
virtual_scanner_deadlock_log_v145.csv
urgent_control_log_v107.csv
opc_write_log_v99.csv
settings_log_v108.csv
```

File availability may differ across runs. Each run is processed
independently, and its identifier is preserved during synchronization,
feature construction, and evaluation.

The main processed tables are stored in:

```text
dynamic_outputs/data/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NavCoderr/agv-dynamic-tms-decision-support.git
cd agv-dynamic-tms-decision-support
```

Create and activate a virtual environment.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository was prepared for Python 3.11. Exact package versions are
listed in `requirements.txt`.

## Execution

### 1. Build the synchronized datasets

```bash
python dynamic_tms.py
```

This script:

- reads the three raw dataset folders;
- synchronizes streams independently within each run;
- constructs the common 1 Hz analysis timeline;
- generates the event-aware diagnostic representation;
- generates the earlier 24-variable event-free representation;
- evaluates the initial models;
- saves processed datasets and initial results.

Main processed outputs:

```text
dynamic_outputs/data/combined_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/combined_model_table_v4.csv
dynamic_outputs/data/dataset1_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/dataset2_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/dataset3_dynamic_tms_dataset_1hz_v4.csv
```

### 2. Generate secondary analyses

```bash
python dynamic_analysis.py
```

This script generates the earlier 24-variable model comparisons,
feature-group analyses, short-horizon classification, distance
regression, threshold analyses, and related figures.

Outputs are stored in:

```text
dynamic_outputs/journal_analysis/
```

### 3. Generate secondary validation outputs

```bash
python extra_validation.py
```

This script generates the earlier 24-variable bootstrap summaries,
feature importance, probability calibration, confusion analysis, and
additional binary metrics.

Outputs are stored in:

```text
dynamic_outputs/journal_extra_validation/
```

The bootstrap percentiles are exploratory sample-level uncertainty
summaries. Adjacent per-second samples are temporally dependent, so they
are not independent-run or independent-episode confidence intervals.

### 4. Run the principal corrected validation

```bash
python jim_final_validation.py
```

This script performs:

- corrected 21-variable leave-one-run-out validation;
- engineering and learned baseline comparison;
- corrected representation ablation;
- episode-level evaluation;
- representative runtime case selection;
- probability calibration;
- permutation importance;
- confusion analysis.

Outputs are stored in:

```text
dynamic_outputs/jim_final_validation/
```

Important result files include:

```text
dynamic_outputs/jim_final_validation/tables/table_corrected_21_feature_list.csv
dynamic_outputs/jim_final_validation/tables/table_21feature_loro_metrics.csv
dynamic_outputs/jim_final_validation/tables/table_21feature_ablation.csv
dynamic_outputs/jim_final_validation/tables/table_engineering_and_learned_baselines.csv
dynamic_outputs/jim_final_validation/tables/table_episode_level_summary.csv
dynamic_outputs/jim_final_validation/tables/table_21feature_calibration.csv
dynamic_outputs/jim_final_validation/tables/table_21feature_confusion_matrix.csv
```

### 5. Run the planning-reference observability audit

Run this script after `jim_final_validation.py`:

```bash
python planning_reference_observability_audit.py
```

This script evaluates:

- planning-reference coverage;
- planned-versus-observed execution deviation;
- planned-only corridor-exposure episodes;
- recognition stratified by planning-reference availability.

Outputs are stored in:

```text
dynamic_outputs/planning_reference_observability/
```

## Recommended Run Order

To regenerate all outputs:

```bash
python dynamic_tms.py
python dynamic_analysis.py
python extra_validation.py
python jim_final_validation.py
python planning_reference_observability_audit.py
```

If the processed files under `dynamic_outputs/data/` already exist and
only the principal analyses are required:

```bash
python jim_final_validation.py
python planning_reference_observability_audit.py
```

## Principal Evaluation Protocol

```text
Principal representation:
    Corrected 21-variable event-free execution context

Direct event and intervention variables:
    Excluded

Binary rule-defining indicators:
    Excluded

Validation:
    Leave one complete runtime run out at a time

Row-level random split across runs:
    Not used

Primary task:
    Binary supervisory-attention recognition

Principal reference test:
    Dataset 3, fitted using Dataset 1 and Dataset 2

Secondary task:
    Fine-grained multiclass recognition using the earlier
    24-variable representation

Diagnostic task:
    Event-aware reconstruction using direct event and intervention
    evidence
```

## Principal Results

### Corrected 21-variable leave-one-run-out evaluation

| Held-out run | Accuracy | Balanced accuracy | Macro-F1 | Risk recall | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dataset 1 | 0.8869 | 0.8367 | 0.8333 | 0.7491 | 0.9277 | 0.7481 | 0.0917 |
| Dataset 2 | 0.8641 | 0.8801 | 0.8566 | 0.9308 | 0.9369 | 0.8735 | 0.1032 |
| Dataset 3 | 0.8760 | 0.8348 | 0.8409 | 0.7441 | 0.9338 | 0.8176 | 0.0920 |

These results demonstrate transfer across the three complete held-out
controlled runtime runs. They do not establish factory-scale or
fleet-scale generalization.

### Principal Dataset 3 result

```text
Accuracy:           0.8760
Balanced accuracy:  0.8348
Macro-F1:           0.8409
Risk precision:     0.7893
Risk recall:        0.7441
ROC-AUC:            0.9338
PR-AUC:             0.8176
Brier score:        0.0920
```

Confusion matrix:

```text
                    Predicted non-risk    Predicted risk/action

True non-risk              733                    59
True risk/action            76                   221
```

The output is not suitable for direct safety control because 76 of the
297 risk/action samples were missed.

## Engineering and Learned Baselines

Selected Dataset 3 results:

| Method | Accuracy | Macro-F1 | Risk recall | PR-AUC |
|---|---:|---:|---:|---:|
| ExtraTrees, corrected 21 | 0.8760 | 0.8409 | 0.7441 | 0.8176 |
| Logistic regression, corrected 21 | 0.8127 | 0.7833 | 0.8148 | 0.8053 |
| Same corridor and closing | 0.7539 | 0.5261 | 0.1111 | 0.3319 |
| Current distance below 0.30 m | 0.7401 | 0.4692 | 0.0471 | 0.3070 |
| Current distance below 0.10 m | 0.7309 | 0.4352 | 0.0135 | 0.2825 |
| Same corridor with current or planned proximity | 0.7309 | 0.4352 | 0.0135 | 0.2869 |
| Always non-risk | 0.7273 | 0.4211 | 0.0000 | 0.2727 |

ExtraTrees achieved the highest macro-F1 and PR-AUC. Logistic regression
achieved higher risk/action recall but lower accuracy, macro-F1, and
PR-AUC.

## Episode-Level Evaluation

| Run | True episodes | Predicted episodes | Detected true episodes | False predicted episodes | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Dataset 1 | 30 | 18 | 16 | 1 | 0.5333 | 0.9444 |
| Dataset 2 | 8 | 8 | 8 | 1 | 1.0000 | 0.8750 |
| Dataset 3 | 20 | 11 | 12 | 1 | 0.6000 | 0.9091 |
| Combined | 58 | 37 | 36 | 3 | 0.6207 | 0.9189 |

The overlap-based matching procedure can associate one predicted
interval with more than one reconstructed true episode. Precision and
recall should therefore be taken directly from the released output
table.

## Planning-Reference Results

```text
Valid planned-reference rows:          911 / 3279
Planning-reference coverage:           27.78%
Planned edge transitions:              302
Planned--SIM same-corridor rows:         42

Mean planned--REAL deviation:         1.501 m
Median planned--REAL deviation:       0.970 m

Planned-only corridor episodes:           7
Linked later current corridor events:     7
Median retrospective offset:             10 s
Offset range:                           3–22 s
```

The corridor-exposure analysis is retrospective. It does not establish
prospective warning accuracy, collision prediction, or avoided-conflict
performance.

## Result Hierarchy

Principal results:

```text
dynamic_outputs/jim_final_validation/
dynamic_outputs/planning_reference_observability/
```

Secondary results:

```text
dynamic_outputs/journal_analysis/
dynamic_outputs/journal_extra_validation/
```

The secondary folders use the earlier 24-variable representation. That
representation excludes direct event variables but retains three binary
indicators involved directly in deterministic conflict-risk
reconstruction.

## Main Output Folders

```text
dynamic_outputs/data/
    Processed synchronized datasets and model-ready tables

dynamic_outputs/results/
    Initial model results and latency tables

dynamic_outputs/journal_analysis/
    Secondary 24-variable analyses

dynamic_outputs/journal_extra_validation/
    Secondary bootstrap, importance, and calibration analyses

dynamic_outputs/jim_final_validation/
    Principal corrected 21-variable validation

dynamic_outputs/planning_reference_observability/
    Planning-reference observability results
```

## Reproducibility Notes

The scripts use a fixed random seed where applicable:

```text
GLOBAL_SEED = 42
```

The final validation also fixes Python and NumPy random states and passes
a fixed `random_state` to stochastic estimators and permutation
importance.

For exact reproduction, use the same:

- raw and processed CSV files;
- input row ordering;
- Python version;
- package versions from `requirements.txt`;
- feature definitions;
- thresholds;
- model configurations;
- random seed.

The repository includes generated `dynamic_outputs/`, so the results can
be inspected directly or regenerated.

## Archival Snapshot

A stable archival snapshot is available on Zenodo:

https://doi.org/10.5281/zenodo.20773984

The Zenodo version used for the manuscript should contain the same
scripts, outputs, and README as the corresponding GitHub release.


## License

This repository is provided for academic research and reproducibility
purposes.

The repository does not provide or certify a safety-rated AGV traffic
control or collision-avoidance function.
