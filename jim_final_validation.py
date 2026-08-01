# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


                                                                               
               
                                                                               
GLOBAL_SEED = 42
N_TREES = 500
MIN_SAMPLES_LEAF = 2
PRINCIPAL_TRAIN_RUNS = ("dataset1", "dataset2")
PRINCIPAL_TEST_RUN = "dataset3"
PROBABILITY_THRESHOLD = 0.50
EPISODE_MIN_PERSISTENCE_S = 2
EPISODE_MERGE_GAP_S = 1
CASE_WINDOW_BEFORE_S = 15
CASE_WINDOW_AFTER_S = 20
PERMUTATION_REPEATS = 10

EVENT_AWARE_FEATURES = [
    "tms_conflict_count",
    "tms_first_severity",
    "tms_horizon_s",
    "tms_hold_s",
    "operator_event",
    "hard_stop_event",
    "hard_stop_operator",
    "hard_stop_scanner",
    "urgent_control_event",
    "virtual_scanner_event",
]

DIRECT_RULE_INDICATORS = [
    "close_distance_flag",
    "very_close_flag",
    "opposite_edge_flag",
]

CORE_COLUMNS = ["run_id", "sec", "state_label", "risk_binary"]

                                                                            
LIVE_SPATIAL_FEATURES = [
    "real_x",
    "real_y",
    "real_speed",
    "real_node",
    "sim1_x",
    "sim1_y",
    "sim1_speed",
    "real_sim_distance_m",
]

GRAPH_CONTEXT_FEATURES = [
    "same_physical_edge",
]

PLANNED_TRAJECTORY_FEATURES = [
    "planned_real_x",
    "planned_real_y",
    "planned_real_u",
    "planned_real_v",
    "planned_real_tau",
    "planned_real_available",
    "planned_to_sim_distance_m",
    "planned_same_sim_edge",
]

TEMPORAL_INTERACTION_FEATURES = [
    "distance_delta_1s",
    "distance_delta_3s",
    "closing_speed_est",
    "planned_distance_delta_1s",
]

CORRECTED_21_FEATURES = (
    LIVE_SPATIAL_FEATURES
    + GRAPH_CONTEXT_FEATURES
    + PLANNED_TRAJECTORY_FEATURES
    + TEMPORAL_INTERACTION_FEATURES
)

RISK_STATES = [
    "operator_stop",
    "deadlock",
    "preentry_block",
    "fallback_hold",
    "safe_node_hold",
    "conflict_risk",
]


                                                                               
                   
                                                                               
def set_seed(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def locate_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        Path.cwd(),
        script_dir / "dynamic_outputs_extracted",
        script_dir / "dynamic_outputs_extracted" / "dynamic_outputs",
    ]

                                                                           
    try:
        candidates.extend([p.parent.parent for p in script_dir.glob("**/dynamic_outputs/data/combined_model_table_v4.csv")])
    except Exception:
        pass

    checked = []
    for candidate in unique_preserve_order([str(p.resolve()) for p in candidates]):
        root = Path(candidate)
        checked.append(str(root))

        direct = root / "dynamic_outputs" / "data" / "combined_model_table_v4.csv"
        if direct.exists():
            return root

                                                      
        if (root / "data" / "combined_model_table_v4.csv").exists() and root.name == "dynamic_outputs":
            return root.parent

    raise FileNotFoundError(
        "Could not locate dynamic_outputs/data/combined_model_table_v4.csv.\n"
        "Place jim_final_validation.py in the project root beside dynamic_outputs/.\n"
        f"Checked: {checked}"
    )


def ensure_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[SAVE] {path}")


def coerce_model_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ensure_columns(out, CORE_COLUMNS + CORRECTED_21_FEATURES, "corrected 21-feature analysis")

    out["run_id"] = out["run_id"].astype(str)
    out["state_label"] = out["state_label"].astype(str)
    out["sec"] = pd.to_numeric(out["sec"], errors="coerce")
    out["risk_binary"] = pd.to_numeric(out["risk_binary"], errors="coerce")

    for c in CORRECTED_21_FEATURES + DIRECT_RULE_INDICATORS + EVENT_AWARE_FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["run_id", "sec", "risk_binary"]).copy()
    out["sec"] = out["sec"].astype(int)
    out["risk_binary"] = out["risk_binary"].astype(int)
    out = out.sort_values(["run_id", "sec"]).reset_index(drop=True)
    return out


def audit_feature_definition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, features in [
        ("live_spatial", LIVE_SPATIAL_FEATURES),
        ("graph_context", GRAPH_CONTEXT_FEATURES),
        ("planned_trajectory", PLANNED_TRAJECTORY_FEATURES),
        ("temporal_interaction", TEMPORAL_INTERACTION_FEATURES),
    ]:
        for feature in features:
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "present": int(feature in df.columns),
                    "missing_fraction": float(df[feature].isna().mean()) if feature in df.columns else np.nan,
                    "n_unique": int(df[feature].nunique(dropna=True)) if feature in df.columns else 0,
                }
            )

                               
    for feature in DIRECT_RULE_INDICATORS:
        rows.append(
            {
                "group": "excluded_direct_rule_indicator",
                "feature": feature,
                "present": int(feature in df.columns),
                "missing_fraction": float(df[feature].isna().mean()) if feature in df.columns else np.nan,
                "n_unique": int(df[feature].nunique(dropna=True)) if feature in df.columns else 0,
            }
        )
    for feature in EVENT_AWARE_FEATURES:
        rows.append(
            {
                "group": "excluded_event_or_action",
                "feature": feature,
                "present": int(feature in df.columns),
                "missing_fraction": float(df[feature].isna().mean()) if feature in df.columns else np.nan,
                "n_unique": int(df[feature].nunique(dropna=True)) if feature in df.columns else 0,
            }
        )
    return pd.DataFrame(rows)


                                                                               
                    
                                                                               
def make_extratrees() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=N_TREES,
                    random_state=GLOBAL_SEED,
                    class_weight="balanced",
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=GLOBAL_SEED,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    row: Dict[str, float] = {
        "n": int(len(yt)),
        "non_risk_support": int(np.sum(yt == 0)),
        "risk_support": int(np.sum(yt == 1)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "risk_precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "risk_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "risk_f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
    }

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    row.update(
        {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )

    if y_prob is not None:
        pr = np.asarray(y_prob, dtype=float)
        if len(np.unique(yt)) == 2:
            row["roc_auc"] = float(roc_auc_score(yt, pr))
            row["pr_auc"] = float(average_precision_score(yt, pr))
            row["brier_score"] = float(brier_score_loss(yt, pr))
        else:
            row["roc_auc"] = np.nan
            row["pr_auc"] = np.nan
            row["brier_score"] = np.nan
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
        row["brier_score"] = np.nan

    return row


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    model_kind: str = "extratrees",
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray]:
    if not features:
        raise ValueError("Feature list is empty.")
    ensure_columns(train, list(features) + ["risk_binary"], "training")
    ensure_columns(test, list(features) + ["risk_binary"], "testing")

    model = make_extratrees() if model_kind == "extratrees" else make_logistic()
    X_train = train[list(features)]
    X_test = test[list(features)]
    y_train = train["risk_binary"].astype(int).to_numpy()
    y_test = test["risk_binary"].astype(int).to_numpy()

    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(int)
    prob = model.predict_proba(X_test)[:, 1].astype(float)
    return model, y_test, pred, prob


                                                                               
                                           
                                                                               
def run_loro(df: pd.DataFrame, tables_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header("1. Corrected 21-feature leave-one-run-out validation")
    runs = sorted(df["run_id"].unique().tolist())
    if len(runs) < 3:
        raise RuntimeError(f"Expected at least three runs; found {runs}")

    metric_rows = []
    prediction_parts = []

    for test_run in runs:
        train_runs = [r for r in runs if r != test_run]
        train = df[df["run_id"].isin(train_runs)].copy()
        test = df[df["run_id"] == test_run].copy()

        model, y_true, pred, prob = fit_predict(train, test, CORRECTED_21_FEATURES, "extratrees")
        row = binary_metrics(y_true, pred, prob)
        row.update(
            {
                "representation": "corrected_21_feature",
                "model": "ExtraTrees",
                "train_runs": "+".join(train_runs),
                "test_run": test_run,
                "n_features": len(CORRECTED_21_FEATURES),
            }
        )
        metric_rows.append(row)

        pred_df = test[["run_id", "sec", "state_label", "risk_binary"]].copy()
        pred_df["predicted_risk"] = pred
        pred_df["risk_probability"] = prob
        pred_df["evaluation"] = "LORO"
        prediction_parts.append(pred_df)

        print(
            f"[LORO] test={test_run}: macro-F1={row['macro_f1']:.4f}, "
            f"risk recall={row['risk_recall']:.4f}, PR-AUC={row['pr_auc']:.4f}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    save_csv(metrics_df, tables_dir / "table_21feature_loro_metrics.csv")
    save_csv(predictions_df, tables_dir / "predictions_21feature_loro_all_runs.csv")
    return metrics_df, predictions_df


                                                                               
                                  
                                                                               
def build_ablation_sets() -> Dict[str, List[str]]:
    full = list(CORRECTED_21_FEATURES)
    sets = {
        "live_spatial_only": list(LIVE_SPATIAL_FEATURES),
        "live_plus_graph": list(LIVE_SPATIAL_FEATURES + GRAPH_CONTEXT_FEATURES),
        "live_plus_planned_trajectory": list(LIVE_SPATIAL_FEATURES + PLANNED_TRAJECTORY_FEATURES),
        "live_plus_temporal_interaction": list(LIVE_SPATIAL_FEATURES + TEMPORAL_INTERACTION_FEATURES),
        "live_graph_planned": list(LIVE_SPATIAL_FEATURES + GRAPH_CONTEXT_FEATURES + PLANNED_TRAJECTORY_FEATURES),
        "full_corrected_21": full,
        "full_minus_planned_trajectory": [f for f in full if f not in PLANNED_TRAJECTORY_FEATURES],
        "full_minus_graph_context": [f for f in full if f not in GRAPH_CONTEXT_FEATURES],
        "full_minus_temporal_interaction": [f for f in full if f not in TEMPORAL_INTERACTION_FEATURES],
        "full_minus_current_separation": [f for f in full if f != "real_sim_distance_m"],
    }
    return {name: unique_preserve_order(features) for name, features in sets.items()}


def run_ablation(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    print_header("2. Corrected 21-feature feature-group and removal ablation")
    train = df[df["run_id"].isin(PRINCIPAL_TRAIN_RUNS)].copy()
    test = df[df["run_id"] == PRINCIPAL_TEST_RUN].copy()

    rows = []
    ablation_sets = build_ablation_sets()

    for name, features in ablation_sets.items():
        _, y_true, pred, prob = fit_predict(train, test, features, "extratrees")
        row = binary_metrics(y_true, pred, prob)
        row.update(
            {
                "feature_set": name,
                "n_features": len(features),
                "features": "|".join(features),
                "train_runs": "+".join(PRINCIPAL_TRAIN_RUNS),
                "test_run": PRINCIPAL_TEST_RUN,
                "model": "ExtraTrees",
            }
        )
        rows.append(row)
        print(f"[ABLATION] {name:36s} n={len(features):2d} macro-F1={row['macro_f1']:.4f}")

    result = pd.DataFrame(rows)
    full_f1 = float(result.loc[result["feature_set"] == "full_corrected_21", "macro_f1"].iloc[0])
    result["macro_f1_change_vs_full"] = result["macro_f1"] - full_f1
    save_csv(result, tables_dir / "table_21feature_ablation.csv")

    ordered = [
        "live_spatial_only",
        "live_plus_graph",
        "live_plus_planned_trajectory",
        "live_plus_temporal_interaction",
        "live_graph_planned",
        "full_corrected_21",
    ]
    plot_df = result.set_index("feature_set").loc[[x for x in ordered if x in result["feature_set"].values]].reset_index()
    plt.figure(figsize=(11, 6))
    x = np.arange(len(plot_df))
    width = 0.38
    plt.bar(x - width / 2, plot_df["accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, plot_df["macro_f1"], width, label="Macro-F1")
    plt.xticks(x, plot_df["feature_set"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Corrected 21-feature representation-development ablation")
    plt.legend()
    plt.tight_layout()
    out = figures_dir / "fig_21feature_ablation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out}")
    return result


                                                                               
                                          
                                                                               
def safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def run_baselines(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header("3. Engineering-rule and learned baselines")
    train = df[df["run_id"].isin(PRINCIPAL_TRAIN_RUNS)].copy()
    test = df[df["run_id"] == PRINCIPAL_TEST_RUN].copy()
    y_true = test["risk_binary"].astype(int).to_numpy()

    distance = safe_numeric(test["real_sim_distance_m"], default=np.inf)
    same_corridor = safe_numeric(test["same_physical_edge"], default=0.0) > 0.5
    closing = safe_numeric(test["closing_speed_est"], default=0.0)
    planned_distance = safe_numeric(test["planned_to_sim_distance_m"], default=np.inf)
    planned_available = safe_numeric(test["planned_real_available"], default=0.0) > 0.5

    baseline_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "always_non_risk": (
            np.zeros(len(test), dtype=int),
            np.full(len(test), float(train["risk_binary"].mean())),
        ),
        "distance_below_0_30m": (
            (distance < 0.30).astype(int),
            np.clip(1.0 - distance / 0.30, 0.0, 1.0),
        ),
        "distance_below_0_10m": (
            (distance < 0.10).astype(int),
            np.clip(1.0 - distance / 0.10, 0.0, 1.0),
        ),
        "same_corridor_and_closing": (
            (same_corridor & (closing > 0.0)).astype(int),
            np.where(same_corridor, np.clip(closing / 0.30, 0.0, 1.0), 0.0),
        ),
        "same_corridor_close_or_planned_close": (
            (same_corridor & ((distance < 0.30) | (planned_available & (planned_distance < 0.30)))).astype(int),
            np.maximum(
                np.where(same_corridor, np.clip(1.0 - distance / 0.30, 0.0, 1.0), 0.0),
                np.where(planned_available, np.clip(1.0 - planned_distance / 0.30, 0.0, 1.0), 0.0),
            ),
        ),
    }

    rows = []
    pred_table = test[["run_id", "sec", "state_label", "risk_binary"]].copy()

    for name, (pred, prob) in baseline_predictions.items():
        row = binary_metrics(y_true, pred, prob)
        row.update(
            {
                "baseline": name,
                "category": "engineering_rule",
                "n_features": 0,
                "train_runs": "+".join(PRINCIPAL_TRAIN_RUNS),
                "test_run": PRINCIPAL_TEST_RUN,
            }
        )
        rows.append(row)
        pred_table[f"pred_{name}"] = pred
        pred_table[f"prob_{name}"] = prob

    for name, kind in [("logistic_regression_corrected_21", "logistic"), ("extratrees_corrected_21", "extratrees")]:
        _, yt, pred, prob = fit_predict(train, test, CORRECTED_21_FEATURES, kind)
        row = binary_metrics(yt, pred, prob)
        row.update(
            {
                "baseline": name,
                "category": "learned_model",
                "n_features": len(CORRECTED_21_FEATURES),
                "train_runs": "+".join(PRINCIPAL_TRAIN_RUNS),
                "test_run": PRINCIPAL_TEST_RUN,
            }
        )
        rows.append(row)
        pred_table[f"pred_{name}"] = pred
        pred_table[f"prob_{name}"] = prob

    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    save_csv(result, tables_dir / "table_engineering_and_learned_baselines.csv")
    save_csv(pred_table, tables_dir / "predictions_principal_test_all_baselines.csv")

    plt.figure(figsize=(11, 6))
    plot_df = result.sort_values("macro_f1", ascending=True)
    y = np.arange(len(plot_df))
    plt.barh(y, plot_df["macro_f1"])
    plt.yticks(y, plot_df["baseline"])
    plt.xlim(0, 1)
    plt.xlabel("Macro-F1")
    plt.title("Engineering-rule and learned-model comparison on Dataset 3")
    plt.tight_layout()
    out = figures_dir / "fig_engineering_baselines_macro_f1.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out}")
    return result, pred_table


                                                                               
                                         
                                                                               
@dataclass
class Episode:
    run_id: str
    start_sec: int
    end_sec: int
    duration_s: int
    label: str
    first_index: int
    last_index: int


def binary_series_to_episodes(
    run_df: pd.DataFrame,
    positive_col: str,
    label_col: Optional[str] = None,
    min_persistence_s: int = 1,
    merge_gap_s: int = 0,
) -> List[Episode]:
    if run_df.empty:
        return []

    g = run_df.sort_values("sec").reset_index(drop=True).copy()
    positive = pd.to_numeric(g[positive_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    secs = g["sec"].astype(int).to_numpy()

    raw: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(positive):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(positive) - 1):
            end = i if flag and i == len(positive) - 1 else i - 1
            raw.append((start, end))
            start = None

                                                                      
    merged: List[Tuple[int, int]] = []
    for s, e in raw:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        gap = secs[s] - secs[pe] - 1
        if gap <= merge_gap_s:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    episodes = []
    for s, e in merged:
        duration = int(secs[e] - secs[s] + 1)
        if duration < min_persistence_s:
            continue

        label = "risk/action"
        if label_col is not None and label_col in g.columns:
            labels = g.loc[s:e, label_col].astype(str)
            labels = labels[~labels.isin(["normal", "resume", "nan"])]
            if not labels.empty:
                label = str(labels.value_counts().index[0])

        episodes.append(
            Episode(
                run_id=str(g.loc[s, "run_id"]),
                start_sec=int(secs[s]),
                end_sec=int(secs[e]),
                duration_s=duration,
                label=label,
                first_index=int(s),
                last_index=int(e),
            )
        )
    return episodes


def overlap_seconds(a: Episode, b: Episode) -> int:
    if a.run_id != b.run_id:
        return 0
    return max(0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec) + 1)


def episode_rows(episodes: Sequence[Episode], episode_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_type": episode_type,
                "run_id": e.run_id,
                "start_sec": e.start_sec,
                "end_sec": e.end_sec,
                "duration_s": e.duration_s,
                "label": e.label,
            }
            for e in episodes
        ]
    )


def evaluate_episodes_for_run(run_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    true_eps = binary_series_to_episodes(
        run_df,
        positive_col="risk_binary",
        label_col="state_label",
        min_persistence_s=1,
        merge_gap_s=EPISODE_MERGE_GAP_S,
    )
    pred_eps = binary_series_to_episodes(
        run_df,
        positive_col="predicted_risk",
        label_col=None,
        min_persistence_s=EPISODE_MIN_PERSISTENCE_S,
        merge_gap_s=EPISODE_MERGE_GAP_S,
    )

    detail_rows = []
    detected_true = 0
    onset_offsets = []

    for i, true_ep in enumerate(true_eps, start=1):
        overlaps = [(j, pred_ep, overlap_seconds(true_ep, pred_ep)) for j, pred_ep in enumerate(pred_eps, start=1)]
        overlaps = [x for x in overlaps if x[2] > 0]
        if overlaps:
            best_id, best_pred, ov = max(overlaps, key=lambda x: x[2])
            detected_true += 1
            offset = int(best_pred.start_sec - true_ep.start_sec)
            onset_offsets.append(offset)
            detected = 1
            matched_pred_id = best_id
            overlap_s = ov
        else:
            detected = 0
            matched_pred_id = np.nan
            overlap_s = 0
            offset = np.nan

        detail_rows.append(
            {
                "run_id": true_ep.run_id,
                "true_episode_id": i,
                "state_label": true_ep.label,
                "true_start_sec": true_ep.start_sec,
                "true_end_sec": true_ep.end_sec,
                "true_duration_s": true_ep.duration_s,
                "detected": detected,
                "matched_predicted_episode_id": matched_pred_id,
                "overlap_s": overlap_s,
                "prediction_onset_offset_s": offset,
            }
        )

    false_pred = 0
    for pred_ep in pred_eps:
        if not any(overlap_seconds(pred_ep, true_ep) > 0 for true_ep in true_eps):
            false_pred += 1

    duration_h = max((run_df["sec"].max() - run_df["sec"].min() + 1) / 3600.0, 1e-12)
    true_n = len(true_eps)
    pred_n = len(pred_eps)
    episode_recall = detected_true / true_n if true_n else np.nan
    matched_predictions = pred_n - false_pred
    episode_precision = matched_predictions / pred_n if pred_n else np.nan

    summary = {
        "run_id": str(run_df["run_id"].iloc[0]),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "prediction_min_persistence_s": EPISODE_MIN_PERSISTENCE_S,
        "merge_gap_s": EPISODE_MERGE_GAP_S,
        "runtime_hours": duration_h,
        "true_episode_count": true_n,
        "predicted_episode_count": pred_n,
        "detected_true_episode_count": detected_true,
        "missed_true_episode_count": true_n - detected_true,
        "false_predicted_episode_count": false_pred,
        "episode_recall": episode_recall,
        "episode_precision": episode_precision,
        "false_alert_episodes_per_hour": false_pred / duration_h,
        "median_prediction_onset_offset_s": float(np.median(onset_offsets)) if onset_offsets else np.nan,
        "mean_prediction_onset_offset_s": float(np.mean(onset_offsets)) if onset_offsets else np.nan,
    }

    return summary, pd.DataFrame(detail_rows), pd.concat(
        [episode_rows(true_eps, "true"), episode_rows(pred_eps, "predicted")],
        ignore_index=True,
    )


def run_episode_analysis(
    df: pd.DataFrame,
    loro_predictions: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_header("4. Episode-level operational evaluation")

    merged = df.merge(
        loro_predictions[["run_id", "sec", "predicted_risk", "risk_probability"]],
        on=["run_id", "sec"],
        how="inner",
        validate="one_to_one",
    )
    merged["predicted_risk"] = (
        pd.to_numeric(merged["risk_probability"], errors="coerce").fillna(0.0) >= PROBABILITY_THRESHOLD
    ).astype(int)

    summary_rows = []
    detail_parts = []
    episode_parts = []

    for run_id, run_df in merged.groupby("run_id", sort=True):
        summary, details, episodes = evaluate_episodes_for_run(run_df.copy())
        summary_rows.append(summary)
        detail_parts.append(details)
        episode_parts.append(episodes)
        print(
            f"[EPISODES] {run_id}: true={summary['true_episode_count']}, "
            f"detected={summary['detected_true_episode_count']}, false={summary['false_predicted_episode_count']}, "
            f"recall={summary['episode_recall']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    episodes_df = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()

                                                                                            
    total_runtime_h = summary_df["runtime_hours"].sum()
    total_true = int(summary_df["true_episode_count"].sum())
    total_pred = int(summary_df["predicted_episode_count"].sum())
    total_detected = int(summary_df["detected_true_episode_count"].sum())
    total_false = int(summary_df["false_predicted_episode_count"].sum())
    aggregate = {
        "run_id": "combined",
        "probability_threshold": PROBABILITY_THRESHOLD,
        "prediction_min_persistence_s": EPISODE_MIN_PERSISTENCE_S,
        "merge_gap_s": EPISODE_MERGE_GAP_S,
        "runtime_hours": total_runtime_h,
        "true_episode_count": total_true,
        "predicted_episode_count": total_pred,
        "detected_true_episode_count": total_detected,
        "missed_true_episode_count": total_true - total_detected,
        "false_predicted_episode_count": total_false,
        "episode_recall": total_detected / total_true if total_true else np.nan,
        "episode_precision": (total_pred - total_false) / total_pred if total_pred else np.nan,
        "false_alert_episodes_per_hour": total_false / total_runtime_h if total_runtime_h else np.nan,
        "median_prediction_onset_offset_s": details_df.loc[details_df["detected"] == 1, "prediction_onset_offset_s"].median() if not details_df.empty else np.nan,
        "mean_prediction_onset_offset_s": details_df.loc[details_df["detected"] == 1, "prediction_onset_offset_s"].mean() if not details_df.empty else np.nan,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([aggregate])], ignore_index=True)

                                       
    if not details_df.empty:
        state_summary = (
            details_df.groupby("state_label", dropna=False)
            .agg(
                true_episodes=("true_episode_id", "count"),
                detected_episodes=("detected", "sum"),
                median_duration_s=("true_duration_s", "median"),
                median_onset_offset_s=("prediction_onset_offset_s", "median"),
            )
            .reset_index()
        )
        state_summary["episode_recall"] = state_summary["detected_episodes"] / state_summary["true_episodes"]
    else:
        state_summary = pd.DataFrame()

    save_csv(summary_df, tables_dir / "table_episode_level_summary.csv")
    save_csv(details_df, tables_dir / "table_episode_detection_details.csv")
    save_csv(episodes_df, tables_dir / "table_true_and_predicted_episodes.csv")
    save_csv(state_summary, tables_dir / "table_episode_detection_by_state.csv")

    if not state_summary.empty:
        plt.figure(figsize=(10, 6))
        plot_df = state_summary.sort_values("episode_recall", ascending=True)
        y = np.arange(len(plot_df))
        plt.barh(y, plot_df["episode_recall"])
        plt.yticks(y, plot_df["state_label"])
        plt.xlim(0, 1)
        plt.xlabel("Episode recall")
        plt.title("LORO episode detection by supervisory state")
        plt.tight_layout()
        out = figures_dir / "fig_episode_recall_by_state.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVE] {out}")

    return summary_df, details_df, merged


                                                                               
                                                    
                                                                               
def choose_case_episodes(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame()

    candidates = details_df.copy()
    candidates["priority"] = candidates["state_label"].map(
        {
            "conflict_risk": 1,
            "preentry_block": 2,
            "deadlock": 3,
            "fallback_hold": 4,
            "safe_node_hold": 5,
            "operator_stop": 6,
        }
    ).fillna(99)

    selected = []
    desired_groups = [
        ["conflict_risk", "preentry_block"],
        ["deadlock", "fallback_hold"],
        ["safe_node_hold", "operator_stop"],
    ]

    used = set()
    for group in desired_groups:
        subset = candidates[candidates["state_label"].isin(group)].copy()
        subset = subset[~subset.index.isin(used)]
        if subset.empty:
            continue
        subset = subset.sort_values(["detected", "true_duration_s"], ascending=[False, False])
        idx = subset.index[0]
        used.add(idx)
        selected.append(candidates.loc[idx])

    if len(selected) < 3:
        remaining = candidates[~candidates.index.isin(used)].sort_values(
            ["detected", "true_duration_s"], ascending=[False, False]
        )
        for idx, row in remaining.iterrows():
            selected.append(row)
            if len(selected) >= 3:
                break

    return pd.DataFrame(selected[:3]).reset_index(drop=True)


def state_to_numeric(state: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    order = ["normal", "resume"] + RISK_STATES
    mapping = {label: idx for idx, label in enumerate(order)}
    values = state.astype(str).map(mapping).fillna(len(mapping)).to_numpy(dtype=float)
    return values, mapping


def run_case_studies(
    merged_predictions: pd.DataFrame,
    details_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    print_header("5. Representative manufacturing case-study figures")
    selected = choose_case_episodes(details_df)
    save_csv(selected, tables_dir / "table_selected_case_studies.csv")
    if selected.empty:
        print("[WARN] No true risk episodes available for case-study plots.")
        return selected

    for case_no, case in selected.iterrows():
        run_id = str(case["run_id"])
        start = int(case["true_start_sec"])
        end = int(case["true_end_sec"])
        lo = start - CASE_WINDOW_BEFORE_S
        hi = end + CASE_WINDOW_AFTER_S
        window = merged_predictions[
            (merged_predictions["run_id"] == run_id)
            & (merged_predictions["sec"] >= lo)
            & (merged_predictions["sec"] <= hi)
        ].sort_values("sec")

        if window.empty:
            continue

        state_numeric, mapping = state_to_numeric(window["state_label"])
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(window["sec"], window["risk_probability"], label="Corrected 21-feature probability")
        axes[0].axhline(PROBABILITY_THRESHOLD, linestyle="--", label="Decision threshold")
        axes[0].axvspan(start, end, alpha=0.18, label="True episode")
        axes[0].set_ylabel("Risk probability")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].legend(loc="upper right")

        axes[1].plot(window["sec"], window["real_sim_distance_m"], label="REAL-SIM distance")
        if "planned_to_sim_distance_m" in window.columns:
            axes[1].plot(window["sec"], window["planned_to_sim_distance_m"], label="Planned-to-SIM distance")
        axes[1].axhline(0.30, linestyle="--", label="0.30 m reference")
        axes[1].set_ylabel("Distance (m)")
        axes[1].legend(loc="upper right")

        axes[2].plot(window["sec"], window["real_speed"], label="REAL speed")
        axes[2].plot(window["sec"], window["sim1_speed"], label="SIM1 speed")
        axes[2].set_ylabel("Speed")
        axes[2].legend(loc="upper right")

        axes[3].step(window["sec"], state_numeric, where="post", label="Reconstructed state")
        axes[3].step(window["sec"], window["predicted_risk"], where="post", label="Predicted risk (0/1)")
        axes[3].set_yticks(list(mapping.values()))
        axes[3].set_yticklabels(list(mapping.keys()))
        axes[3].set_ylabel("State")
        axes[3].set_xlabel("Synchronized runtime second")
        axes[3].legend(loc="upper right")

        fig.suptitle(
            f"Case {case_no + 1}: {case['state_label']} in {run_id} "
            f"({start}-{end} s, detected={int(case['detected'])})"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = figures_dir / f"fig_case_{case_no + 1}_{run_id}_{case['state_label']}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] {out}")

    return selected


                                                                               
                                                                
                                                                               
def run_calibration_and_importance(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header("6. Corrected 21-feature calibration and permutation importance")
    train = df[df["run_id"].isin(PRINCIPAL_TRAIN_RUNS)].copy()
    test = df[df["run_id"] == PRINCIPAL_TEST_RUN].copy()
    model, y_true, pred, prob = fit_predict(train, test, CORRECTED_21_FEATURES, "extratrees")

                                   
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    calibration_df = pd.DataFrame(
        {
            "mean_predicted_probability": mean_pred,
            "observed_risk_fraction": frac_pos,
        }
    )
    save_csv(calibration_df, tables_dir / "table_21feature_calibration.csv")

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="Corrected 21-feature ExtraTrees")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed risk/action fraction")
    plt.title("Probability calibration on held-out Dataset 3")
    plt.legend()
    plt.tight_layout()
    out = figures_dir / "fig_21feature_calibration.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out}")

                                                                            
    perm = permutation_importance(
        model,
        test[CORRECTED_21_FEATURES],
        y_true,
        scoring="f1_macro",
        n_repeats=PERMUTATION_REPEATS,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": CORRECTED_21_FEATURES,
            "importance_mean_macro_f1_drop": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_macro_f1_drop", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    save_csv(importance_df, tables_dir / "table_21feature_permutation_importance.csv")

    top = importance_df.head(15).sort_values("importance_mean_macro_f1_drop", ascending=True)
    plt.figure(figsize=(9, 7))
    y = np.arange(len(top))
    plt.barh(y, top["importance_mean_macro_f1_drop"], xerr=top["importance_std"])
    plt.yticks(y, top["feature"])
    plt.xlabel("Mean decrease in test macro-F1 after permutation")
    plt.title("Corrected 21-feature permutation importance")
    plt.tight_layout()
    out = figures_dir / "fig_21feature_permutation_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out}")

                                 
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_non_risk", "true_risk_action"], columns=["pred_non_risk", "pred_risk_action"])
    save_csv(cm_df.reset_index().rename(columns={"index": "true_class"}), tables_dir / "table_21feature_confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks([0, 1], ["non-risk", "risk/action"])
    plt.yticks([0, 1], ["non-risk", "risk/action"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Corrected 21-feature ExtraTrees on Dataset 3")
    plt.tight_layout()
    out = figures_dir / "fig_21feature_confusion_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out}")

    return calibration_df, importance_df


                                                                               
             
                                                                               


def main() -> int:
    set_seed()
    print_header("JIM FINAL VALIDATION: corrected 21-feature representation")

    project_root = locate_project_root()
    dynamic_outputs = project_root / "dynamic_outputs"
    data_dir = dynamic_outputs / "data"
    model_table_path = data_dir / "combined_model_table_v4.csv"

    out_dir = dynamic_outputs / "jim_final_validation"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    for d in [out_dir, tables_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[PROJECT] {project_root}")
    print(f"[INPUT]   {model_table_path}")
    print(f"[OUTPUT]  {out_dir}")

    df = pd.read_csv(model_table_path)
    df = coerce_model_table(df)

    if len(CORRECTED_21_FEATURES) != 21:
        raise AssertionError(f"Corrected feature list must contain 21 features; found {len(CORRECTED_21_FEATURES)}")
    if set(CORRECTED_21_FEATURES) & set(DIRECT_RULE_INDICATORS):
        raise AssertionError("A direct rule indicator entered the corrected feature list.")
    if set(CORRECTED_21_FEATURES) & set(EVENT_AWARE_FEATURES):
        raise AssertionError("An event/action feature entered the corrected feature list.")

    feature_audit = audit_feature_definition(df)
    save_csv(feature_audit, tables_dir / "table_feature_definition_and_exclusion_audit.csv")
    save_csv(pd.DataFrame({"corrected_21_feature": CORRECTED_21_FEATURES}), tables_dir / "table_corrected_21_feature_list.csv")

    run_summary = (
        df.groupby("run_id")
        .agg(
            rows=("sec", "size"),
            start_sec=("sec", "min"),
            end_sec=("sec", "max"),
            non_risk_samples=("risk_binary", lambda s: int((s == 0).sum())),
            risk_action_samples=("risk_binary", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )
    save_csv(run_summary, tables_dir / "table_input_run_summary.csv")

    loro_metrics, loro_predictions = run_loro(df, tables_dir)
    ablation = run_ablation(df, tables_dir, figures_dir)
    baselines, _ = run_baselines(df, tables_dir, figures_dir)
    episode_summary, episode_details, merged_predictions = run_episode_analysis(
        df, loro_predictions, tables_dir, figures_dir
    )
    run_case_studies(merged_predictions, episode_details, tables_dir, figures_dir)
    run_calibration_and_importance(df, tables_dir, figures_dir)


    summary_payload = {
        "project_root": str(project_root),
        "input_model_table": str(model_table_path),
        "output_directory": str(out_dir),
        "seed": GLOBAL_SEED,
        "corrected_feature_count": len(CORRECTED_21_FEATURES),
        "corrected_features": CORRECTED_21_FEATURES,
        "excluded_direct_rule_indicators": DIRECT_RULE_INDICATORS,
        "excluded_event_action_features": EVENT_AWARE_FEATURES,
        "principal_train_runs": list(PRINCIPAL_TRAIN_RUNS),
        "principal_test_run": PRINCIPAL_TEST_RUN,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "episode_min_persistence_s": EPISODE_MIN_PERSISTENCE_S,
        "episode_merge_gap_s": EPISODE_MERGE_GAP_S,
    }
    save_json(summary_payload, out_dir / "run_configuration.json")

    print_header("COMPLETED SUCCESSFULLY")
    print(f"All new JIM validation outputs are in:\n{out_dir}")
    print("The original dynamic_outputs folders and the three existing scripts were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n[FATAL ERROR]", exc)
        traceback.print_exc()
        raise
