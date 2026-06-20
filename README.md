# Runtime Traffic-Management Decision Support for Industrial AGVs

This repository contains the code, runtime datasets, processed outputs, and analysis scripts for the manuscript:

**Runtime Traffic-Management Decision Support for Industrial AGVs Using Learned Trajectory Priors and Hybrid Physical–Simulated Validation**

## Overview

The project builds a 1 Hz runtime state representation for AGV traffic-management decision support. It combines physical AGV telemetry, software-simulated traffic trajectories, planned trajectory priors, graph-relation features, safe-node decisions, TMS actions, virtual-scanner events, urgent-control events, and operator-stop evidence.

The main predictive task is binary risk/action recognition using event-free features. Fine-grained multiclass TMS-state recognition is reported as a diagnostic and exploratory task. The framework is intended as a supervisory decision-support layer and does not claim certified collision-free AGV control.

## Repository Structure

```text
dataset1/                 Runtime logs for dataset 1
dataset2/                 Runtime logs for dataset 2
dataset3/                 Runtime logs for dataset 3
dynamic_outputs/          Generated processed data, results, figures, and validation outputs
dynamic_tms.py            Main pipeline for dataset construction and model evaluation
dynamic_analysis.py       Script for result table and figure preparation
extra_validation.py       Bootstrap, feature importance, calibration, and extra validation
requirements.txt          Python package requirements with fixed versions
```

## Runtime Dataset Folders

Each dataset folder contains logs collected during one hybrid physical–simulated AGV runtime experiment. The main raw log types include:

```text
agent_live_trajectory_log_v145.csv            Physical and simulated AGV trajectory logs
dynamic_decision_log_v99.csv                  Runtime decision-support log
dynamic_training_log_v99.csv                  Runtime training/state log
dynamic_edge_traversals_v111.csv              Edge traversal log
safe_node_tms_log_v146.csv                    Safe-node TMS action log
tms_runtime_action_log_v143.csv               TMS runtime action log
series_leg_trajectory_sample_log_v98.csv      Planned trajectory-prior samples
series_leg_plan_log_v99.csv                   Planned leg information
series_mission_log_v99.csv                    Mission-level runtime log
hard_stop_log_v141.csv                        Hard-stop and intervention log
emergency_event_log_v99.csv                   Emergency-event log
virtual_scanner_deadlock_log_v145.csv         Virtual-scanner/deadlock log
urgent_control_log_v107.csv                   Urgent-control log, available in dataset 2 and dataset 3
opc_write_log_v99.csv                         OPC UA write/action log
settings_log_v108.csv                         Runtime settings log
```

The synchronized and processed research-ready datasets are stored in `dynamic_outputs/data/`.

## Reproducibility and Execution Steps

### 1. Clone the repository

```bash
git clone https://github.com/NavCoderr/agv-dynamic-tms-decision-support.git
cd agv-dynamic-tms-decision-support
```

### 2. Create a Python environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main runtime TMS pipeline

```bash
python dynamic_tms.py
```

This script constructs the synchronized 1 Hz runtime datasets, prepares event-aware and event-free feature tables, evaluates the main machine-learning models, and saves the processed outputs under `dynamic_outputs/`.

Expected main outputs include:

```text
dynamic_outputs/data/combined_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/combined_model_table_v4.csv
dynamic_outputs/data/dataset1_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/dataset2_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/data/dataset3_dynamic_tms_dataset_1hz_v4.csv
dynamic_outputs/results/model_results_event_aware_v5.csv
dynamic_outputs/results/model_results_event_free_v5.csv
dynamic_outputs/results/model_results_v5_event_aware_and_event_free.csv
dynamic_outputs/results/leave_one_run_out_results_v4.csv
dynamic_outputs/results/feature_group_ablation_v4.csv
dynamic_outputs/results/runtime_latency_v4.csv
```

### 5. Generate journal tables and figures

```bash
python dynamic_analysis.py
```

This script prepares journal-ready result tables and figures from the generated outputs.

Expected outputs include:

```text
dynamic_outputs/journal_analysis/table_main_results.csv
dynamic_outputs/journal_analysis/table_best_event_free_binary_model.csv
dynamic_outputs/journal_analysis/table_feature_ablation.csv
dynamic_outputs/journal_analysis/table_leave_one_run_out.csv
dynamic_outputs/journal_analysis/table_future_prediction.csv
dynamic_outputs/journal_analysis/table_distance_regression.csv
dynamic_outputs/journal_analysis/table_leakage_audit.csv
dynamic_outputs/journal_analysis/table_policy_threshold.csv
dynamic_outputs/journal_analysis/fig_future_prediction.png
dynamic_outputs/journal_analysis/fig_distance_regression.png
dynamic_outputs/journal_analysis/fig_leakage_audit.png
```

### 6. Run extra validation analyses

```bash
python extra_validation.py
```

This script prepares additional validation outputs used for robustness analysis, including bootstrap confidence intervals, feature importance, probability calibration, and event-free binary risk/action evaluation.

Expected outputs include:

```text
dynamic_outputs/journal_extra_validation/tables/journal_extra_bootstrap_ci.csv
dynamic_outputs/journal_extra_validation/tables/journal_extra_bootstrap_raw.csv
dynamic_outputs/journal_extra_validation/tables/journal_extra_event_free_metrics.csv
dynamic_outputs/journal_extra_validation/tables/journal_extra_feature_importance.csv
dynamic_outputs/journal_extra_validation/tables/journal_extra_calibration_curve.csv
dynamic_outputs/journal_extra_validation/figures/fig_event_free_extratrees_confusion_matrix.png
dynamic_outputs/journal_extra_validation/figures/fig_event_free_feature_importance_top15.png
dynamic_outputs/journal_extra_validation/figures/fig_event_free_probability_calibration.png
```

## Main Output Folders

```text
dynamic_outputs/data/                         Processed 1 Hz datasets and feature tables
dynamic_outputs/results/                      Main model result tables
dynamic_outputs/figures/                      Generated figures
dynamic_outputs/extra_outputs/                Classification reports and audit outputs
dynamic_outputs/journal_analysis/             Journal-ready tables and summaries
dynamic_outputs/journal_extra_validation/     Extra validation outputs
```

## Main Evaluation Protocol

```text
Training runs: Dataset 1 and Dataset 2
Independent test run: Dataset 3
Main predictive setting: Event-free binary risk/action recognition
Secondary setting: Fine-grained multiclass TMS-state recognition
Diagnostic setting: Event-aware recognition using direct event/intervention features
```

The event-free setting removes direct TMS and safety-event indicators and uses runtime spatial state, graph-relation variables, trajectory-prior variables, and short-term motion trends. This setting is treated as the main predictive evaluation because it avoids over-claiming from direct event-log evidence.

## Key Reported Results

The main event-free binary risk/action model is ExtraTrees. In the independent run-wise test, it achieved:

```text
Accuracy:      0.8797
Macro-F1:      0.8452
Risk recall:   0.7475
ROC-AUC:       0.9391
PR-AUC:        0.8742
Brier score:   0.0866
```

Bootstrap analysis produced a 95% confidence interval of 0.8203–0.8684 for macro-F1.

## Reproducibility Notes

The scripts use a fixed random seed where applicable:

```text
GLOBAL_SEED = 42
```

The main pipeline is designed to reproduce the processed 1 Hz datasets, event-aware and event-free model tables, feature-group ablation, leave-one-run-out validation, threshold sensitivity, latency analysis, leakage audit, future-horizon prediction, future distance regression, bootstrap confidence intervals, feature importance, and calibration outputs.

Because the repository already includes generated `dynamic_outputs/`, the user can either inspect the existing processed outputs directly or rerun the scripts to regenerate them.

## Software Dependencies

The repository was prepared for Python 3.11. The exact Python package versions used for reproducibility are listed in `requirements.txt`.

## Citation

If you use this repository, please cite the associated manuscript:

Naveen Sharma, Wojciech Klein, and Rafał Cupek, “Runtime Traffic-Management Decision Support for Industrial AGVs Using Learned Trajectory Priors and Hybrid Physical–Simulated Validation,” manuscript under review.

## Archival Snapshot

A stable archival snapshot of this repository will be deposited on Zenodo before journal submission. The Zenodo DOI will be added here after the snapshot is created.

DOI: To be added.

## License

This repository is provided for research and reproducibility purposes. Please cite the associated manuscript and repository if you use the data, code, or processed outputs.
