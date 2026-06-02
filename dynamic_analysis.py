# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "dynamic_outputs"

RESULTS_DIR = OUT_DIR / "results"

ANALYSIS_DIR = OUT_DIR / "journal_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[MISSING] {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        print(f"[OK] Loaded {path.name}: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}")
        return pd.DataFrame()


def save_table(df: pd.DataFrame, name: str):
    out = ANALYSIS_DIR / name
    df.to_csv(out, index=False)
    print(f"[SAVE] {out}")


def clean_main_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "setting",
        "task",
        "model",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy()

    numeric_cols = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    sort_cols = [c for c in ["setting", "task", "macro_f1"] if c in out.columns]

    if sort_cols:
        ascending = [True, True, False][: len(sort_cols)]
        out = out.sort_values(by=sort_cols, ascending=ascending)

    return out


def select_best_event_free_binary(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()

    required_cols = {"setting", "task", "macro_f1"}

    if not required_cols.issubset(main_df.columns):
        return pd.DataFrame()

    out = main_df[
        (main_df["setting"].astype(str) == "event_free_early")
        & (main_df["task"].astype(str) == "binary_risk")
    ].copy()

    if out.empty:
        return pd.DataFrame()

    out["macro_f1"] = pd.to_numeric(out["macro_f1"], errors="coerce")
    out = out.sort_values("macro_f1", ascending=False)

    return out.head(1)


def clean_future_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "prediction_horizon_s",
        "model",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
        "n_train",
        "n_test",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy()

    numeric_cols = [
        "prediction_horizon_s",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
        "n_train",
        "n_test",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"prediction_horizon_s", "macro_f1"}.issubset(out.columns):
        out = out.sort_values(
            ["prediction_horizon_s", "macro_f1"],
            ascending=[True, False],
        )

    return out


def best_future_by_horizon(future_df: pd.DataFrame) -> pd.DataFrame:
    if future_df.empty:
        return pd.DataFrame()

    if "prediction_horizon_s" not in future_df.columns:
        return pd.DataFrame()

    out = future_df.copy()
    out["macro_f1"] = pd.to_numeric(out["macro_f1"], errors="coerce")

    out = out.sort_values(
        ["prediction_horizon_s", "macro_f1"],
        ascending=[True, False],
    )

    return out.groupby("prediction_horizon_s").head(1).reset_index(drop=True)


def clean_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "setting",
        "feature_set",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
    ]

    existing = [c for c in keep_cols if c in df.columns]

    if not existing:
        return df.copy()

    out = df[existing].copy()

    numeric_cols = [
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "risk_precision",
        "risk_recall",
        "roc_auc",
        "pr_auc",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def clean_distance_regression(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "prediction_horizon_s",
        "model",
        "mae_m",
        "rmse_m",
        "r2",
        "n_train",
        "n_test",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy() if existing else df.copy()

    numeric_cols = [
        "prediction_horizon_s",
        "mae_m",
        "rmse_m",
        "r2",
        "n_train",
        "n_test",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    sort_cols = [c for c in ["prediction_horizon_s", "mae_m"] if c in out.columns]

    if sort_cols:
        out = out.sort_values(sort_cols)

    return out


def clean_policy_threshold(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "threshold",
        "macro_f1",
        "risk_recall",
        "risk_precision",
        "accuracy",
        "balanced_accuracy",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy() if existing else df.copy()

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "threshold" in out.columns:
        out = out.sort_values("threshold")

    return out


def clean_loro(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "task",
        "model",
        "train_runs",
        "test_run",
        "accuracy",
        "macro_f1",
        "n_train",
        "n_test",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy() if existing else df.copy()

    numeric_cols = [
        "accuracy",
        "macro_f1",
        "n_train",
        "n_test",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def clean_ablation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "task",
        "model",
        "feature_set",
        "n_features",
        "accuracy",
        "macro_f1",
        "n_train",
        "n_test",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy() if existing else df.copy()

    numeric_cols = [
        "n_features",
        "accuracy",
        "macro_f1",
        "n_train",
        "n_test",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def plot_future_prediction(future_df: pd.DataFrame):
    required_cols = {"prediction_horizon_s", "model", "macro_f1"}

    if future_df.empty or not required_cols.issubset(future_df.columns):
        return

    plt.figure(figsize=(8, 5))

    for model, group in future_df.groupby("model"):
        group = group.sort_values("prediction_horizon_s")
        plt.plot(
            group["prediction_horizon_s"],
            group["macro_f1"],
            marker="o",
            label=str(model),
        )

    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.05)
    plt.title("Future Risk Prediction")
    plt.legend()
    plt.tight_layout()

    out = ANALYSIS_DIR / "fig_future_prediction.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {out}")


def plot_leakage_audit(leak_df: pd.DataFrame):
    if leak_df.empty or "macro_f1" not in leak_df.columns:
        return

    if "feature_set" in leak_df.columns:
        label_col = "feature_set"
    elif "setting" in leak_df.columns:
        label_col = "setting"
    else:
        return

    out_df = leak_df.copy()
    out_df["macro_f1"] = pd.to_numeric(out_df["macro_f1"], errors="coerce")
    out_df = out_df.dropna(subset=["macro_f1"])

    if out_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(out_df[label_col].astype(str), out_df["macro_f1"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.05)
    plt.title("Feature Leakage Audit")
    plt.tight_layout()

    out = ANALYSIS_DIR / "fig_leakage_audit.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {out}")


def plot_policy_threshold(policy_df: pd.DataFrame):
    if policy_df.empty or "threshold" not in policy_df.columns:
        return

    plt.figure(figsize=(8, 5))

    for metric in ["macro_f1", "risk_recall", "risk_precision"]:
        if metric in policy_df.columns:
            plt.plot(
                policy_df["threshold"],
                policy_df[metric],
                marker="o",
                label=metric,
            )

    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Decision Threshold Analysis")
    plt.legend()
    plt.tight_layout()

    out = ANALYSIS_DIR / "fig_policy_threshold.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {out}")


def plot_distance_regression(distance_df: pd.DataFrame):
    required_cols = {"prediction_horizon_s", "model", "rmse_m"}

    if distance_df.empty or not required_cols.issubset(distance_df.columns):
        return

    plt.figure(figsize=(8, 5))

    for model, group in distance_df.groupby("model"):
        group = group.sort_values("prediction_horizon_s")
        plt.plot(
            group["prediction_horizon_s"],
            group["rmse_m"],
            marker="o",
            label=str(model),
        )

    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("RMSE (m)")
    plt.title("Future Distance Regression")
    plt.legend()
    plt.tight_layout()

    out = ANALYSIS_DIR / "fig_distance_regression.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {out}")


def main():
    print("=" * 80)
    print("DYNAMIC TMS ANALYSIS")
    print("=" * 80)

    main_results = read_csv_if_exists(
        RESULTS_DIR / "model_results_v5_event_aware_and_event_free.csv"
    )

    if main_results.empty:
        main_results = read_csv_if_exists(
            RESULTS_DIR / "model_results_v4_event_aware_and_event_free.csv"
        )

    future_results = read_csv_if_exists(
        RESULTS_DIR / "future_horizon_binary_risk_results_v1.csv"
    )

    leakage_audit = read_csv_if_exists(
        RESULTS_DIR / "event_feature_leakage_audit_v1.csv"
    )

    distance_regression = read_csv_if_exists(
        RESULTS_DIR / "future_distance_regression_results_v1.csv"
    )

    policy_threshold = read_csv_if_exists(
        RESULTS_DIR / "decision_policy_threshold_results_v1.csv"
    )

    loro_results = read_csv_if_exists(
        RESULTS_DIR / "leave_one_run_out_results_v4.csv"
    )

    ablation_results = read_csv_if_exists(
        RESULTS_DIR / "feature_group_ablation_v4.csv"
    )

    main_table = clean_main_results(main_results)
    best_event_free = select_best_event_free_binary(main_table)
    future_table = clean_future_results(future_results)
    future_best = best_future_by_horizon(future_table)
    leak_table = clean_leakage_audit(leakage_audit)
    distance_table = clean_distance_regression(distance_regression)
    policy_table = clean_policy_threshold(policy_threshold)
    loro_table = clean_loro(loro_results)
    ablation_table = clean_ablation(ablation_results)

    save_table(main_table, "table_main_results.csv")
    save_table(best_event_free, "table_best_event_free_binary_model.csv")
    save_table(future_table, "table_future_prediction.csv")
    save_table(future_best, "table_future_prediction_best_by_horizon.csv")
    save_table(leak_table, "table_leakage_audit.csv")
    save_table(distance_table, "table_distance_regression.csv")
    save_table(policy_table, "table_policy_threshold.csv")
    save_table(loro_table, "table_leave_one_run_out.csv")
    save_table(ablation_table, "table_feature_ablation.csv")

    plot_future_prediction(future_table)
    plot_leakage_audit(leak_table)
    plot_policy_threshold(policy_table)
    plot_distance_regression(distance_table)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output folder: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()