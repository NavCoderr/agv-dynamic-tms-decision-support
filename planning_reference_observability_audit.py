#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline


                                                                               
               
                                                                               

@dataclass(frozen=True)
class Config:
    random_seed: int = 42
    n_estimators: int = 500
    min_samples_leaf: int = 2
    probability_threshold: float = 0.50

                                                                   
    deviation_bands_m: tuple[float, ...] = (0.25, 0.50, 1.00)

                                                                              
                                                  
    prospective_link_horizon_s: int = 60

                                                                             
    exposure_merge_gap_s: int = 1

                                                                  
                                      
    supervisory_followup_horizon_s: int = 60


CFG = Config()

DIRECT_EVENT_FEATURES = {
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
}

DIRECT_RULE_FEATURES = {
    "close_distance_flag",
    "very_close_flag",
    "opposite_edge_flag",
}

PLANNED_FEATURES = {
    "planned_real_x",
    "planned_real_y",
    "planned_real_u",
    "planned_real_v",
    "planned_real_tau",
    "planned_real_available",
    "planned_to_sim_distance_m",
    "planned_same_sim_edge",
}


                                                                               
           
                                                                               

def banner(text: str) -> None:
    print("\n" + "=" * 96)
    print(text)
    print("=" * 96)


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "dynamic_outputs" / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find dynamic_outputs/data. Place this script in the project "
        "root or run it from the project root."
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}")


def safe_metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "n": 0,
            "non_risk_support": 0,
            "risk_support": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "risk_precision": np.nan,
            "risk_recall": np.nan,
            "risk_f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier_score": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    labels_present = np.unique(y_true)
    tn = fp = fn = tp = 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc = roc_auc_score(y_true, y_prob) if len(labels_present) == 2 else np.nan
    pr = average_precision_score(y_true, y_prob) if len(labels_present) == 2 else np.nan

    return {
        "n": int(len(y_true)),
        "non_risk_support": int((y_true == 0).sum()),
        "risk_support": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, y_pred)
            if len(labels_present) == 2
            else np.nan
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "risk_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "risk_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "risk_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier_score_loss(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def binary_episodes(
    sec: np.ndarray,
    flag: np.ndarray,
    merge_gap_s: int = 1,
) -> list[dict]:
    sec = np.asarray(sec, dtype=int)
    flag = np.asarray(flag, dtype=bool)
    active = sec[flag]
    if len(active) == 0:
        return []

    episodes: list[dict] = []
    start = prev = int(active[0])
    for current in active[1:]:
        current = int(current)
        if current - prev <= merge_gap_s + 1:
            prev = current
            continue
        episodes.append(
            {
                "start_sec": start,
                "end_sec": prev,
                "duration_s": prev - start + 1,
            }
        )
        start = prev = current
    episodes.append(
        {
            "start_sec": start,
            "end_sec": prev,
            "duration_s": prev - start + 1,
        }
    )
    return episodes


def make_model(feature_names: list[str]) -> Pipeline:
                                                                                 
                                                       
    pre = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_names,
            )
        ],
        remainder="drop",
    )
    clf = ExtraTreesClassifier(
        n_estimators=CFG.n_estimators,
        random_state=CFG.random_seed,
        class_weight="balanced",
        min_samples_leaf=CFG.min_samples_leaf,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", clf)])


def infer_corrected_features(model_df: pd.DataFrame, root: Path) -> list[str]:
    explicit = (
        root
        / "dynamic_outputs"
        / "jim_final_validation"
        / "tables"
        / "table_corrected_21_feature_list.csv"
    )
    if explicit.exists():
        feature_df = pd.read_csv(explicit)
        candidate_column = feature_df.columns[0]
        features = feature_df[candidate_column].dropna().astype(str).tolist()
    else:
        excluded = {
            "run_id",
            "sec",
            "state_label",
            "risk_binary",
            *DIRECT_EVENT_FEATURES,
            *DIRECT_RULE_FEATURES,
        }
        features = [
            c
            for c in model_df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(model_df[c])
        ]

    invalid = sorted((DIRECT_EVENT_FEATURES | DIRECT_RULE_FEATURES).intersection(features))
    if invalid:
        raise RuntimeError(f"Leakage audit failed. Excluded features found: {invalid}")
    if len(features) != 21:
        warnings.warn(
            f"Expected 21 corrected features, found {len(features)}: {features}",
            RuntimeWarning,
        )
    return features


                                                                               
               
                                                                               

def build_plan_coverage_table(runtime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, g in runtime_df.groupby("run_id", sort=True):
        g = g.sort_values("sec")
        avail = g["planned_real_available"].fillna(0).astype(int).eq(1)
        valid_xy = avail & g["planned_real_x"].notna() & g["planned_real_y"].notna()
        valid_edge = avail & g["planned_real_u"].notna() & g["planned_real_v"].notna()

        edge_changes = (
            g.loc[valid_edge, ["planned_real_u", "planned_real_v"]]
            .astype(str)
            .agg("->".join, axis=1)
            .ne(
                g.loc[valid_edge, ["planned_real_u", "planned_real_v"]]
                .astype(str)
                .agg("->".join, axis=1)
                .shift()
            )
            .sum()
        )
        rows.append(
            {
                "run_id": run_id,
                "runtime_rows": len(g),
                "runtime_duration_s": int(g["sec"].max() - g["sec"].min()),
                "planned_state_rows": int(avail.sum()),
                "planned_coverage_percent": 100.0 * avail.mean(),
                "planned_position_rows": int(valid_xy.sum()),
                "planned_edge_rows": int(valid_edge.sum()),
                "planned_edge_transitions_observed": int(max(edge_changes - 1, 0)),
                "planned_same_sim_edge_rows": int(
                    ((g["planned_same_sim_edge"].fillna(0) == 1) & avail).sum()
                ),
            }
        )

    out = pd.DataFrame(rows)
    total = {
        "run_id": "combined",
        "runtime_rows": int(out["runtime_rows"].sum()),
        "runtime_duration_s": int(out["runtime_duration_s"].sum()),
        "planned_state_rows": int(out["planned_state_rows"].sum()),
        "planned_coverage_percent": (
            100.0 * out["planned_state_rows"].sum() / out["runtime_rows"].sum()
        ),
        "planned_position_rows": int(out["planned_position_rows"].sum()),
        "planned_edge_rows": int(out["planned_edge_rows"].sum()),
        "planned_edge_transitions_observed": int(
            out["planned_edge_transitions_observed"].sum()
        ),
        "planned_same_sim_edge_rows": int(out["planned_same_sim_edge_rows"].sum()),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def add_plan_real_deviation(runtime_df: pd.DataFrame) -> pd.DataFrame:
    df = runtime_df.copy()
    df["plan_real_deviation_m"] = np.sqrt(
        (df["planned_real_x"] - df["real_x"]) ** 2
        + (df["planned_real_y"] - df["real_y"]) ** 2
    )
    valid = (
        df["planned_real_available"].fillna(0).astype(int).eq(1)
        & df["planned_real_x"].notna()
        & df["planned_real_y"].notna()
        & df["real_x"].notna()
        & df["real_y"].notna()
    )
    df.loc[~valid, "plan_real_deviation_m"] = np.nan
    return df


def summarize_deviation(g: pd.DataFrame) -> dict:
    x = g["plan_real_deviation_m"].dropna()
    if x.empty:
        result = {
            "n_valid_planned_rows": 0,
            "mean_deviation_m": np.nan,
            "median_deviation_m": np.nan,
            "p90_deviation_m": np.nan,
            "p95_deviation_m": np.nan,
            "max_deviation_m": np.nan,
        }
    else:
        result = {
            "n_valid_planned_rows": int(len(x)),
            "mean_deviation_m": x.mean(),
            "median_deviation_m": x.median(),
            "p90_deviation_m": x.quantile(0.90),
            "p95_deviation_m": x.quantile(0.95),
            "max_deviation_m": x.max(),
        }
    for threshold in CFG.deviation_bands_m:
        result[f"rows_above_{str(threshold).replace('.', '_')}m"] = int(
            (x > threshold).sum()
        )
        result[f"share_above_{str(threshold).replace('.', '_')}m"] = (
            float((x > threshold).mean()) if len(x) else np.nan
        )
    return result


def build_deviation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for run_id, g in df.groupby("run_id", sort=True):
        run_rows.append({"run_id": run_id, **summarize_deviation(g)})
    run_rows.append({"run_id": "combined", **summarize_deviation(df)})
    by_run = pd.DataFrame(run_rows)

    state_rows = []
    valid = df[df["plan_real_deviation_m"].notna()].copy()
    for (run_id, state), g in valid.groupby(["run_id", "state_label"], sort=True):
        state_rows.append(
            {
                "run_id": run_id,
                "state_label": state,
                **summarize_deviation(g),
            }
        )
    for state, g in valid.groupby("state_label", sort=True):
        state_rows.append(
            {
                "run_id": "combined",
                "state_label": state,
                **summarize_deviation(g),
            }
        )
    return by_run, pd.DataFrame(state_rows)


def first_followup_supervisory_state(
    g: pd.DataFrame, start_sec: int, horizon_s: int
) -> tuple[str, Optional[int]]:
    window = g[
        (g["sec"] >= start_sec)
        & (g["sec"] <= start_sec + horizon_s)
        & (~g["state_label"].isin(["normal", "resume"]))
    ]
    if window.empty:
        return "no_supervisory_state", None
    row = window.iloc[0]
    return str(row["state_label"]), int(row["sec"] - start_sec)


def build_prospective_exposure_table(runtime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_id = 0

    for run_id, g in runtime_df.groupby("run_id", sort=True):
        g = g.sort_values("sec").reset_index(drop=True)
        planned_only = (
            g["planned_real_available"].fillna(0).astype(int).eq(1)
            & g["planned_same_sim_edge"].fillna(0).astype(int).eq(1)
            & g["same_physical_edge"].fillna(0).astype(int).eq(0)
        )

        episodes = binary_episodes(
            g["sec"].to_numpy(),
            planned_only.to_numpy(),
            merge_gap_s=CFG.exposure_merge_gap_s,
        )

        for ep in episodes:
            episode_id += 1
            start = int(ep["start_sec"])
            end = int(ep["end_sec"])

            future_current = g[
                (g["sec"] > start)
                & (g["sec"] <= start + CFG.prospective_link_horizon_s)
                & (g["same_physical_edge"].fillna(0).astype(int) == 1)
            ]
            if future_current.empty:
                later_same_sec = np.nan
                lead_time = np.nan
                linked = 0
            else:
                later_same_sec = int(future_current.iloc[0]["sec"])
                lead_time = int(later_same_sec - start)
                linked = 1

            state, state_offset = first_followup_supervisory_state(
                g, start, CFG.supervisory_followup_horizon_s
            )
            segment = g[(g["sec"] >= start) & (g["sec"] <= end)]

            rows.append(
                {
                    "exposure_episode_id": episode_id,
                    "run_id": run_id,
                    "planned_exposure_start_sec": start,
                    "planned_exposure_end_sec": end,
                    "planned_exposure_duration_s": int(ep["duration_s"]),
                    "linked_to_later_current_same_corridor": linked,
                    "later_current_same_corridor_sec": later_same_sec,
                    "planned_reference_lead_time_s": lead_time,
                    "first_followup_supervisory_state": state,
                    "supervisory_state_offset_s": state_offset,
                    "min_planned_to_sim_distance_m": segment[
                        "planned_to_sim_distance_m"
                    ].min(),
                    "min_current_real_sim_distance_m": segment[
                        "real_sim_distance_m"
                    ].min(),
                    "mean_plan_real_deviation_m": segment[
                        "plan_real_deviation_m"
                    ].mean(),
                }
            )

    return pd.DataFrame(rows)


def summarize_exposure_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "candidate_episodes",
                "linked_to_later_current_same_corridor",
                "median_lead_time_s",
                "min_lead_time_s",
                "max_lead_time_s",
                "followed_by_supervisory_state",
            ]
        )

    for run_id, g in episodes.groupby("run_id", sort=True):
        linked = g[g["linked_to_later_current_same_corridor"] == 1]
        rows.append(
            {
                "run_id": run_id,
                "candidate_episodes": len(g),
                "linked_to_later_current_same_corridor": int(len(linked)),
                "median_lead_time_s": linked[
                    "planned_reference_lead_time_s"
                ].median(),
                "min_lead_time_s": linked["planned_reference_lead_time_s"].min(),
                "max_lead_time_s": linked["planned_reference_lead_time_s"].max(),
                "followed_by_supervisory_state": int(
                    (g["first_followup_supervisory_state"] != "no_supervisory_state").sum()
                ),
            }
        )

    linked = episodes[episodes["linked_to_later_current_same_corridor"] == 1]
    rows.append(
        {
            "run_id": "combined",
            "candidate_episodes": len(episodes),
            "linked_to_later_current_same_corridor": int(len(linked)),
            "median_lead_time_s": linked["planned_reference_lead_time_s"].median(),
            "min_lead_time_s": linked["planned_reference_lead_time_s"].min(),
            "max_lead_time_s": linked["planned_reference_lead_time_s"].max(),
            "followed_by_supervisory_state": int(
                (
                    episodes["first_followup_supervisory_state"]
                    != "no_supervisory_state"
                ).sum()
            ),
        }
    )
    return pd.DataFrame(rows)


def run_plan_availability_stratified_models(
    model_df: pd.DataFrame,
    corrected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_features = corrected_features
    minus_planned_features = [f for f in corrected_features if f not in PLANNED_FEATURES]

    rows = []
    prediction_parts = []

    for test_run in sorted(model_df["run_id"].unique()):
        train = model_df[model_df["run_id"] != test_run].copy()
        test = model_df[model_df["run_id"] == test_run].copy()

        for representation, features in [
            ("full_corrected_21", full_features),
            ("full_minus_planned", minus_planned_features),
        ]:
            model = make_model(features)
            model.fit(train[features], train["risk_binary"].astype(int))
            prob = model.predict_proba(test[features])[:, 1]
            pred = (prob >= CFG.probability_threshold).astype(int)

            pred_df = test[
                ["run_id", "sec", "state_label", "risk_binary", "planned_real_available"]
            ].copy()
            pred_df["representation"] = representation
            pred_df["risk_probability"] = prob
            pred_df["predicted_risk"] = pred
            prediction_parts.append(pred_df)

            strata = {
                "all_samples": np.ones(len(test), dtype=bool),
                "planned_available": test["planned_real_available"]
                .fillna(0)
                .astype(int)
                .eq(1)
                .to_numpy(),
                "planned_unavailable": test["planned_real_available"]
                .fillna(0)
                .astype(int)
                .eq(0)
                .to_numpy(),
            }

            for stratum_name, mask in strata.items():
                metrics = safe_metric(
                    test.loc[mask, "risk_binary"].to_numpy(),
                    pred[mask],
                    prob[mask],
                )
                rows.append(
                    {
                        **metrics,
                        "representation": representation,
                        "n_features": len(features),
                        "stratum": stratum_name,
                        "train_runs": "+".join(
                            sorted(train["run_id"].unique().astype(str))
                        ),
                        "test_run": test_run,
                    }
                )

    return pd.DataFrame(rows), pd.concat(prediction_parts, ignore_index=True)


                                                                               
         
                                                                               

def plot_coverage(coverage: pd.DataFrame, path: Path) -> None:
    d = coverage[coverage["run_id"] != "combined"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(d["run_id"], d["planned_coverage_percent"])
    ax.set_ylabel("Planned-state coverage (%)")
    ax.set_xlabel("Runtime run")
    ax.set_title("Availability of the time-indexed planning reference")
    ax.set_ylim(0, max(100, d["planned_coverage_percent"].max() * 1.15))
    for i, value in enumerate(d["planned_coverage_percent"]):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
    save_figure(fig, path)


def plot_deviation_by_state(deviation_by_state: pd.DataFrame, path: Path) -> None:
    d = deviation_by_state[
        deviation_by_state["run_id"].eq("combined")
    ].copy()
    d = d.sort_values("median_deviation_m", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(d["state_label"], d["median_deviation_m"])
    ax.set_ylabel("Median planned--REAL deviation (m)")
    ax.set_xlabel("Reconstructed supervisory state")
    ax.set_title("Execution deviation during valid planning-reference intervals")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, path)


def plot_metrics_by_availability(metrics: pd.DataFrame, path: Path) -> None:
    d = metrics[
        metrics["stratum"].isin(["planned_available", "planned_unavailable"])
    ].copy()
    d["label"] = (
        d["test_run"]
        + " | "
        + d["representation"].str.replace("_", " ", regex=False)
        + " | "
        + d["stratum"].str.replace("_", " ", regex=False)
    )
    d = d.sort_values(["test_run", "representation", "stratum"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(d)), d["macro_f1"])
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["label"], rotation=60, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Corrected-model performance stratified by planned-state availability")
    ax.set_ylim(0, 1)
    save_figure(fig, path)


def select_representative_episode(
    episodes: pd.DataFrame,
) -> Optional[pd.Series]:
    if episodes.empty:
        return None
    linked = episodes[
        episodes["linked_to_later_current_same_corridor"].eq(1)
        & episodes["planned_reference_lead_time_s"].notna()
    ].copy()
    if not linked.empty:
                                                                                     
        linked["has_supervisory_followup"] = (
            linked["first_followup_supervisory_state"] != "no_supervisory_state"
        ).astype(int)
        return linked.sort_values(
            ["has_supervisory_followup", "planned_reference_lead_time_s"],
            ascending=[False, False],
        ).iloc[0]
    return episodes.sort_values("planned_exposure_duration_s", ascending=False).iloc[0]


def plot_representative_plan_execution_episode(
    runtime_df: pd.DataFrame,
    selected: Optional[pd.Series],
    path: Path,
) -> None:
    if selected is None:
        return
    run_id = str(selected["run_id"])
    start = int(selected["planned_exposure_start_sec"])
    later = selected.get("later_current_same_corridor_sec", np.nan)
    end_anchor = int(later) if pd.notna(later) else int(selected["planned_exposure_end_sec"])
    lo = max(0, start - 20)
    hi = end_anchor + 30

    g = runtime_df[
        (runtime_df["run_id"] == run_id)
        & (runtime_df["sec"] >= lo)
        & (runtime_df["sec"] <= hi)
    ].copy()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(g["sec"], g["plan_real_deviation_m"], label="Planned--REAL deviation")
    axes[0].set_ylabel("Deviation (m)")
    axes[0].legend(loc="upper right")

    axes[1].plot(g["sec"], g["planned_to_sim_distance_m"], label="Planned--SIM distance")
    axes[1].plot(g["sec"], g["real_sim_distance_m"], label="REAL--SIM distance")
    axes[1].set_ylabel("Distance (m)")
    axes[1].legend(loc="upper right")

    axes[2].step(
        g["sec"],
        g["planned_same_sim_edge"].fillna(0),
        where="post",
        label="Planned same corridor as SIM",
    )
    axes[2].step(
        g["sec"],
        g["same_physical_edge"].fillna(0),
        where="post",
        label="Current REAL same corridor as SIM",
    )
    axes[2].step(
        g["sec"],
        g["risk_binary"].fillna(0),
        where="post",
        label="Supervisory attention state",
    )
    axes[2].set_ylabel("Binary relation")
    axes[2].set_xlabel("Runtime second")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.axvline(start, linestyle="--", linewidth=1.2, label=None)
        if pd.notna(later):
            ax.axvline(int(later), linestyle=":", linewidth=1.2, label=None)

    fig.suptitle(
        f"Representative planning-reference exposure: {run_id}, "
        f"start={start}s"
        + (
            f", later current corridor relation={int(later)}s"
            if pd.notna(later)
            else ""
        )
    )
    save_figure(fig, path)


def plot_planned_vs_current_exposure(
    runtime_df: pd.DataFrame,
    path: Path,
) -> None:
    valid = runtime_df[
        runtime_df["planned_real_available"].fillna(0).astype(int).eq(1)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        valid["real_sim_distance_m"],
        valid["planned_to_sim_distance_m"],
        s=12,
        alpha=0.35,
    )
    upper = np.nanpercentile(
        np.concatenate(
            [
                valid["real_sim_distance_m"].dropna().to_numpy(),
                valid["planned_to_sim_distance_m"].dropna().to_numpy(),
            ]
        ),
        95,
    )
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Current REAL--SIM distance (m)")
    ax.set_ylabel("Planned--SIM distance (m)")
    ax.set_title("Planned and current traffic exposure during valid plan intervals")
    save_figure(fig, path)


                                                                               
        
                                                                               



                                                                               
             
                                                                               

def main() -> None:
    banner("PLANNING-REFERENCE OBSERVABILITY AUDIT")

    root = find_project_root()
    data_dir = root / "dynamic_outputs" / "data"
    output_dir = root / "dynamic_outputs" / "planning_reference_observability"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = data_dir / "combined_dynamic_tms_dataset_1hz_v4.csv"
    model_path = data_dir / "combined_model_table_v4.csv"
    if not runtime_path.exists():
        raise FileNotFoundError(runtime_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"[PROJECT] {root}")
    print(f"[RUNTIME INPUT] {runtime_path}")
    print(f"[MODEL INPUT]   {model_path}")
    print(f"[OUTPUT]        {output_dir}")

    runtime_df = pd.read_csv(runtime_path)
    model_df = pd.read_csv(model_path)

    runtime_required = [
        "run_id",
        "sec",
        "state_label",
        "risk_binary",
        "real_x",
        "real_y",
        "real_sim_distance_m",
        "same_physical_edge",
        "planned_real_x",
        "planned_real_y",
        "planned_real_u",
        "planned_real_v",
        "planned_real_available",
        "planned_to_sim_distance_m",
        "planned_same_sim_edge",
    ]
    require_columns(runtime_df, runtime_required, "Runtime dataset")
    require_columns(
        model_df,
        ["run_id", "sec", "state_label", "risk_binary", "planned_real_available"],
        "Model table",
    )

    runtime_df = runtime_df.sort_values(["run_id", "sec"]).reset_index(drop=True)
    model_df = model_df.sort_values(["run_id", "sec"]).reset_index(drop=True)
    runtime_df = add_plan_real_deviation(runtime_df)

    corrected_features = infer_corrected_features(model_df, root)

    banner("1. Planned-reference coverage")
    coverage = build_plan_coverage_table(runtime_df)
    save_csv(coverage, tables_dir / "table_plan_coverage_by_run.csv")
    plot_coverage(coverage, figures_dir / "fig_plan_coverage_by_run.png")

    banner("2. Planned-versus-observed execution deviation")
    deviation_run, deviation_state = build_deviation_tables(runtime_df)
    save_csv(deviation_run, tables_dir / "table_plan_real_deviation.csv")
    save_csv(
        deviation_state,
        tables_dir / "table_plan_real_deviation_by_state.csv",
    )
    plot_deviation_by_state(
        deviation_state,
        figures_dir / "fig_plan_real_deviation_by_state.png",
    )

    banner("3. Planned corridor-exposure candidates")
    exposure_episodes = build_prospective_exposure_table(runtime_df)
    exposure_summary = summarize_exposure_episodes(exposure_episodes)
    save_csv(
        exposure_episodes,
        tables_dir / "table_planned_corridor_exposure_episodes.csv",
    )
    save_csv(
        exposure_summary,
        tables_dir / "table_planned_exposure_lead_time.csv",
    )
    plot_planned_vs_current_exposure(
        runtime_df,
        figures_dir / "fig_planned_vs_current_traffic_exposure.png",
    )
    selected = select_representative_episode(exposure_episodes)
    plot_representative_plan_execution_episode(
        runtime_df,
        selected,
        figures_dir / "fig_representative_plan_execution_episode.png",
    )
    if selected is not None:
        save_csv(
            pd.DataFrame([selected]),
            tables_dir / "table_representative_plan_execution_episode.csv",
        )

    banner("4. Performance stratified by planning-reference availability")
    stratified_metrics, stratified_predictions = (
        run_plan_availability_stratified_models(model_df, corrected_features)
    )
    save_csv(
        stratified_metrics,
        tables_dir / "table_metrics_by_plan_availability.csv",
    )
    save_csv(
        stratified_predictions,
        tables_dir / "predictions_by_plan_availability.csv",
    )
    plot_metrics_by_availability(
        stratified_metrics,
        figures_dir / "fig_metrics_by_plan_availability.png",
    )

    banner("5. Configuration and interpretation notes")
    config_payload = {
        **asdict(CFG),
        "corrected_features": corrected_features,
        "planned_features_removed_in_comparator": sorted(
            PLANNED_FEATURES.intersection(corrected_features)
        ),
        "direct_event_features_excluded": sorted(DIRECT_EVENT_FEATURES),
        "direct_rule_features_excluded": sorted(DIRECT_RULE_FEATURES),
        "interpretation": (
            "This is an observability audit. Planned-only exposure episodes are "
            "retrospective corridor-context candidates, not certified warnings."
        ),
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )
    print(f"[SAVE] {output_dir / 'run_configuration.json'}")


    banner("COMPLETED SUCCESSFULLY")
    print(f"All outputs are in:\n{output_dir}")


if __name__ == "__main__":
    main()
