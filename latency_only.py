from __future__ import annotations

import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


SEED = 42
WARMUP_CALLS = 100
TIMED_CALLS = 2000
TREE_N_JOBS = 1

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dynamic_outputs" / "data"
OUTPUT_DIR = BASE_DIR / "dynamic_outputs" / "latency_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "runtime_latency_21_features.csv"

PRIMARY_FEATURES = [
    "real_x",
    "real_y",
    "real_speed",
    "real_node",
    "sim1_x",
    "sim1_y",
    "sim1_speed",
    "real_sim_distance_m",
    "same_physical_edge",
    "planned_real_x",
    "planned_real_y",
    "planned_real_u",
    "planned_real_v",
    "planned_real_tau",
    "planned_real_available",
    "planned_to_sim_distance_m",
    "planned_same_sim_edge",
    "distance_delta_1s",
    "distance_delta_3s",
    "closing_speed_est",
    "planned_distance_delta_1s",
]

FORBIDDEN_FEATURES = {
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
    "close_distance_flag",
    "very_close_flag",
    "opposite_edge_flag",
}


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def machine_information() -> dict[str, object]:
    processor = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    memory_bytes = run_command(["sysctl", "-n", "hw.memsize"])
    machine_model = run_command(["sysctl", "-n", "hw.model"])
    macos_version = run_command(["sw_vers", "-productVersion"])
    macos_build = run_command(["sw_vers", "-buildVersion"])

    if not processor:
        processor = platform.processor() or platform.machine()

    try:
        memory_gb = round(int(memory_bytes) / (1024 ** 3), 2)
    except Exception:
        memory_gb = np.nan

    return {
        "machine_model": machine_model or platform.machine(),
        "processor": processor,
        "memory_gb": memory_gb,
        "operating_system": (
            f"macOS {macos_version} (build {macos_build})"
            if macos_version
            else f"{platform.system()} {platform.release()}"
        ),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }


def load_model_table() -> pd.DataFrame:
    candidates = sorted(
        DATA_DIR.glob("combined_model_table_v4*.csv"),
        key=lambda path: (path.stat().st_mtime, path.stat().st_size),
        reverse=True,
    )

    if not candidates:
        candidates = sorted(
            DATA_DIR.glob("combined_model_table*.csv"),
            key=lambda path: (path.stat().st_mtime, path.stat().st_size),
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            f"No combined_model_table_v4*.csv or combined_model_table*.csv found in {DATA_DIR}"
        )

    input_file = candidates[0]
    print(f"[LOAD] {input_file}")
    return pd.read_csv(input_file)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"run_id", "risk_binary", *PRIMARY_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(PRIMARY_FEATURES) != 21:
        raise AssertionError("The primary representation must contain exactly 21 features.")

    forbidden_overlap = FORBIDDEN_FEATURES.intersection(PRIMARY_FEATURES)
    if forbidden_overlap:
        raise AssertionError(f"Forbidden features found: {sorted(forbidden_overlap)}")

    work = df.copy()
    work["run_id"] = work["run_id"].astype(str)
    work["risk_binary"] = pd.to_numeric(work["risk_binary"], errors="raise").astype(int)

    for feature in PRIMARY_FEATURES:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    train = work[work["run_id"].isin(["dataset1", "dataset2"])].copy()
    test = work[work["run_id"].eq("dataset3")].copy()

    if train.empty or test.empty:
        raise RuntimeError("Expected dataset1 and dataset2 for training and dataset3 for testing.")

    for feature in PRIMARY_FEATURES:
        median = train[feature].median()
        fill_value = 0.0 if pd.isna(median) else float(median)
        train[feature] = train[feature].fillna(fill_value)
        test[feature] = test[feature].fillna(fill_value)

    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_models(train: pd.DataFrame):
    x_train = train[PRIMARY_FEATURES].to_numpy(dtype=float)
    y_train = train["risk_binary"].to_numpy(dtype=int)

    models = {}

    random_forest = RandomForestClassifier(
        n_estimators=500,
        random_state=SEED,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=TREE_N_JOBS,
    )
    random_forest.fit(x_train, y_train)
    models["Random Forest"] = (random_forest, None)

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        random_state=SEED,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=TREE_N_JOBS,
    )
    extra_trees.fit(x_train, y_train)
    models["ExtraTrees"] = (extra_trees, None)

    hgb_scaler = StandardScaler()
    x_train_hgb = hgb_scaler.fit_transform(x_train)
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        random_state=SEED,
    )
    hgb.fit(x_train_hgb, y_train)
    models["HistGradientBoosting"] = (hgb, hgb_scaler)

    mlp_scaler = StandardScaler()
    x_train_mlp = mlp_scaler.fit_transform(x_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        learning_rate_init=0.001,
        max_iter=800,
        random_state=SEED,
    )
    mlp.fit(x_train_mlp, y_train)
    models["MLP"] = (mlp, mlp_scaler)

    return models


def measure_latency(model, prepared_test: np.ndarray) -> np.ndarray:
    test_rows = len(prepared_test)

    for index in range(WARMUP_CALLS):
        row = prepared_test[index % test_rows : index % test_rows + 1]
        model.predict_proba(row)

    timings_ms = np.empty(TIMED_CALLS, dtype=float)

    for index in range(TIMED_CALLS):
        row_index = index % test_rows
        row = prepared_test[row_index : row_index + 1]

        start = time.perf_counter_ns()
        model.predict_proba(row)
        end = time.perf_counter_ns()

        timings_ms[index] = (end - start) / 1_000_000.0

    return timings_ms


def main() -> None:
    set_seed()

    df = load_model_table()
    train, test = prepare_data(df)
    models = fit_models(train)

    x_test = test[PRIMARY_FEATURES].to_numpy(dtype=float)
    machine = machine_information()
    rows = []

    for model_name, (model, scaler) in models.items():
        prepared_test = scaler.transform(x_test) if scaler is not None else x_test
        timings = measure_latency(model, prepared_test)
        mean_ms = float(np.mean(timings))

        rows.append(
            {
                "model": model_name,
                "n_features": len(PRIMARY_FEATURES),
                "train_rows": len(train),
                "test_rows": len(test),
                "random_seed": SEED,
                "warmup_calls": WARMUP_CALLS,
                "timed_calls": TIMED_CALLS,
                "tree_n_jobs": TREE_N_JOBS if model_name in {"Random Forest", "ExtraTrees"} else np.nan,
                "mean_latency_ms": mean_ms,
                "std_latency_ms": float(np.std(timings, ddof=1)),
                "median_latency_ms": float(np.median(timings)),
                "p95_latency_ms": float(np.quantile(timings, 0.95)),
                "p99_latency_ms": float(np.quantile(timings, 0.99)),
                "samples_per_second": 1000.0 / mean_ms,
                "below_1_second": "Yes" if float(np.quantile(timings, 0.95)) < 1000.0 else "No",
                "measurement_scope": "single-sample model inference from a prepared 21-feature vector",
                **machine,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False)

    print(result.to_string(index=False))
    print(f"\n[SAVE] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
