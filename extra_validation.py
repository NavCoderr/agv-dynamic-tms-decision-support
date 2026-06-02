# -*- coding: utf-8 -*-

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000


def set_global_seed(seed=GLOBAL_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(GLOBAL_SEED)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "dynamic_outputs"
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
EXTRA_DIR = OUT_DIR / "extra_outputs"
JOURNAL_EXTRA_DIR = OUT_DIR / "journal_extra_validation"

for d in [JOURNAL_EXTRA_DIR, JOURNAL_EXTRA_DIR / "figures", JOURNAL_EXTRA_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)


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

CORE_COLUMNS = {"run_id", "sec", "state_label", "risk_binary"}


def print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def load_model_table() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "combined_model_table_v4.csv",
        DATA_DIR / "combined_model_table_v2.csv",
        DATA_DIR / "combined_model_table.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"[OK] Loaded model table: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find combined_model_table_v4.csv or combined_model_table_v2.csv inside dynamic_outputs/data."
    )


def get_event_free_features(df: pd.DataFrame):
    numeric_cols = []

    for c in df.columns:
        if c in CORE_COLUMNS:
            continue

        if c in EVENT_AWARE_FEATURES:
            continue

        vals = pd.to_numeric(df[c], errors="coerce")

        if vals.notna().mean() >= 0.15:
            df[c] = vals
            numeric_cols.append(c)

    return numeric_cols


def split_train_test(df: pd.DataFrame):
    train = df[df["run_id"].isin(["dataset1", "dataset2"])].copy()
    test = df[df["run_id"].isin(["dataset3"])].copy()

    if train.empty or test.empty:
        raise RuntimeError(
            "Train/test split failed. Expected run_id values dataset1, dataset2, dataset3."
        )

    return train, test


def fit_event_free_extratrees(train, test, features):
    Xtr = train[features].values
    Xte = test[features].values

    ytr = train["risk_binary"].astype(int).values
    yte = test["risk_binary"].astype(int).values

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=GLOBAL_SEED,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xte)[:, 1]
    else:
        prob = pred.astype(float)

    return model, yte, pred, prob


def calculate_binary_metrics(y_true, y_pred, y_prob=None):
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "risk_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "risk_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "risk_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "support_total": int(len(y_true)),
        "risk_support": int(np.sum(y_true == 1)),
        "non_risk_support": int(np.sum(y_true == 0)),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, y_prob)
        row["pr_auc"] = average_precision_score(y_true, y_prob)
        row["brier_score"] = brier_score_loss(y_true, y_prob)

    return row


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(GLOBAL_SEED)

    rows = []
    n = len(y_true)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        yt = y_true[idx]
        yp = y_pred[idx]
        pr = y_prob[idx]

        if len(np.unique(yt)) < 2:
            continue

        rows.append(calculate_binary_metrics(yt, yp, pr))

    boot = pd.DataFrame(rows)

    summary_rows = []

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "risk_precision",
        "risk_recall",
        "risk_f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        if metric not in boot.columns:
            continue

        summary_rows.append(
            {
                "metric": metric,
                "mean": boot[metric].mean(),
                "ci95_low": boot[metric].quantile(0.025),
                "ci95_high": boot[metric].quantile(0.975),
                "std": boot[metric].std(),
                "n_bootstrap_valid": len(boot),
            }
        )

    return pd.DataFrame(summary_rows), boot


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig_path = JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_extratrees_confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Event-free ExtraTrees risk prediction")
    plt.colorbar()

    plt.xticks([0, 1], ["non-risk", "risk/action"], rotation=25, ha="right")
    plt.yticks([0, 1], ["non-risk", "risk/action"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {fig_path}")


def save_feature_importance(model, features):
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)

    out_csv = JOURNAL_EXTRA_DIR / "tables" / "journal_extra_feature_importance.csv"
    imp.to_csv(out_csv, index=False)

    print(f"[SAVE] {out_csv}")

    top = imp.head(15).iloc[::-1]

    fig_path = JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_feature_importance_top15.png"

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Top event-free features for risk/action prediction")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {fig_path}")

    return imp


def save_calibration(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=8,
        strategy="uniform",
    )

    cal_df = pd.DataFrame(
        {
            "mean_predicted_probability": prob_pred,
            "fraction_of_positives": prob_true,
        }
    )

    out_csv = JOURNAL_EXTRA_DIR / "tables" / "journal_extra_calibration_curve.csv"
    cal_df.to_csv(out_csv, index=False)

    print(f"[SAVE] {out_csv}")

    fig_path = JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_probability_calibration.png"

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="ExtraTrees")
    plt.xlabel("Mean predicted risk probability")
    plt.ylabel("Observed risk/action fraction")
    plt.title("Probability calibration for event-free risk prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {fig_path}")

    return cal_df


def save_tables(metrics_row, ci_df):
    metrics_df = pd.DataFrame([metrics_row])

    metrics_out = JOURNAL_EXTRA_DIR / "tables" / "journal_extra_event_free_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    print(f"[SAVE] {metrics_out}")

    ci_out = JOURNAL_EXTRA_DIR / "tables" / "journal_extra_bootstrap_ci.csv"
    ci_df.to_csv(ci_out, index=False)

    print(f"[SAVE] {ci_out}")


def main():
    print_header("JOURNAL EXTRA VALIDATION: BOOTSTRAP + FEATURE IMPORTANCE + CALIBRATION")
    print("Base folder:", BASE_DIR)
    print("Output folder:", JOURNAL_EXTRA_DIR)

    df = load_model_table()

    features = get_event_free_features(df)

    print(f"[INFO] Event-free features used: {len(features)}")

    for f in features:
        print(" -", f)

    train, test = split_train_test(df)

    model, y_true, y_pred, y_prob = fit_event_free_extratrees(
        train,
        test,
        features,
    )

    print_header("MAIN EVENT-FREE EXTRATREES METRICS")

    metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:22s}: {v:.4f}")
        else:
            print(f"{k:22s}: {v}")

    print_header("BOOTSTRAP 95% CONFIDENCE INTERVALS")

    ci_df, boot_df = bootstrap_ci(y_true, y_pred, y_prob, N_BOOTSTRAP)

    print(ci_df)

    boot_out = JOURNAL_EXTRA_DIR / "tables" / "journal_extra_bootstrap_raw.csv"
    boot_df.to_csv(boot_out, index=False)

    print(f"[SAVE] {boot_out}")

    save_confusion_matrix(y_true, y_pred)
    imp_df = save_feature_importance(model, features)
    save_calibration(y_true, y_prob)
    save_tables(metrics, ci_df)

    print_header("DONE")
    print("Created folder:", JOURNAL_EXTRA_DIR)
    print("Main files:")
    print("1.", JOURNAL_EXTRA_DIR / "tables" / "journal_extra_bootstrap_ci.csv")
    print("2.", JOURNAL_EXTRA_DIR / "tables" / "journal_extra_feature_importance.csv")
    print("3.", JOURNAL_EXTRA_DIR / "tables" / "journal_extra_event_free_metrics.csv")
    print("4.", JOURNAL_EXTRA_DIR / "tables" / "journal_extra_calibration_curve.csv")
    print("5.", JOURNAL_EXTRA_DIR / "tables" / "journal_extra_bootstrap_raw.csv")
    print("6.", JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_feature_importance_top15.png")
    print("7.", JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_probability_calibration.png")
    print("8.", JOURNAL_EXTRA_DIR / "figures" / "fig_event_free_extratrees_confusion_matrix.png")


if __name__ == "__main__":
    main()