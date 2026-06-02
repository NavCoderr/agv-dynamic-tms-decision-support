# Knowledge-Informed Runtime Decision Support for AGV Traffic Management

This repository contains the code, runtime datasets, processed outputs, and analysis scripts:

**Knowledge-Informed Runtime Decision Support for AGV Traffic Management Using Learned Trajectory Priors**

## Overview

The project builds a 1 Hz runtime knowledge representation for AGV traffic-management decision support. It combines physical AGV telemetry, simulated AGV trajectories, planned trajectory priors, graph-relation features, safe-node decisions, TMS actions, virtual-scanner events, urgent-control events, and operator-stop evidence.

The main predictive task is binary risk/action recognition using event-free features. Fine-grained multiclass TMS-state recognition is reported as a diagnostic and exploratory task.

## Repository Structure

```text
dataset1/                 Runtime logs for dataset 1
dataset2/                 Runtime logs for dataset 2
dataset3/                 Runtime logs for dataset 3
dynamic_outputs/          Generated processed data, results, figures, and validation outputs
dynamic_tms.py            Main pipeline for dataset construction and model evaluation
dynamic_analysis.py       Script for result table and figure preparation
extra_validation.py       Bootstrap, feature importance, calibration, and extra validation
requirements.txt          Python package requirements
