# -*- coding: utf-8 -*-

import ast
import json
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    except Exception:
        pass


set_global_seed(GLOBAL_SEED)

import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parent

RUN_FOLDERS = {
    "dataset1": BASE_DIR / "dataset1",
    "dataset2": BASE_DIR / "dataset2",
    "dataset3": BASE_DIR / "dataset3",
}

OUT_DIR = BASE_DIR / "dynamic_outputs"
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
MODELS_DIR = OUT_DIR / "models"
EXTRA_DIR = OUT_DIR / "extra_outputs"

for _dir in [OUT_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, EXTRA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


WINDOW_SIZE = 8
MAX_INTERPOLATE_GAP_SECONDS = 4
CLOSE_DISTANCE_M = 0.30
VERY_CLOSE_DISTANCE_M = 0.10
MIN_LABEL_COUNT_TRAIN = 3
FUTURE_HORIZONS_S = [0, 1, 3, 5]

LOG_PATTERNS = {
    "agent": "agent_live_trajectory_log_v145*.csv",
    "command_accept": "command_accept_log_v101*.csv",
    "command": "command_log_v99*.csv",
    "dynamic_decision": "dynamic_decision_log_v99*.csv",
    "edge_traversals": "dynamic_edge_traversals_v111*.csv",
    "dynamic_training": "dynamic_training_log_v99*.csv",
    "edge_costs": "edge_costs_runtime_v98*.csv",
    "emergency": "emergency_event_log_v99*.csv",
    "hard_stop": "hard_stop_log_v141*.csv",
    "opc_write": "opc_write_log_v99*.csv",
    "safe_node": "safe_node_tms_log_v146*.csv",
    "series_plan": "series_leg_plan_log_v99*.csv",
    "series_schedule": "series_leg_schedule_log_v98*.csv",
    "trajectory_sample": "series_leg_trajectory_sample_log_v98*.csv",
    "series_mission": "series_mission_log_v99*.csv",
    "settings": "settings_log_v108*.csv",
    "sim_series": "sim_series_log_v98*.csv",
    "tms_action": "tms_runtime_action_log_v143*.csv",
    "tms_enable": "tms_runtime_enable_log_v145*.csv",
    "urgent_control": "urgent_control_log_v107*.csv",
    "virtual_scanner": "virtual_scanner_deadlock_log_v145*.csv",
}

MIXED_EVENT_LOG_KEYS = {
    "command",
    "emergency",
    "hard_stop",
    "opc_write",
    "safe_node",
    "series_mission",
    "sim_series",
    "virtual_scanner",
}

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

RISK_LABELS = [
    "deadlock",
    "operator_stop",
    "fallback_hold",
    "safe_node_hold",
    "preentry_block",
    "conflict_risk",
]


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def pick_file(folder: Path, pattern: str) -> Optional[Path]:
    files = list(folder.glob(pattern))
    if not files:
        return None
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)[0]


def safe_get_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    lower = {c.lower().strip(): c for c in df.columns}

    for n in names:
        key = n.lower().strip()
        if key in lower:
            return lower[key]

    return None


def parse_time_column(df: pd.DataFrame) -> pd.Series:
    time_col = safe_get_col(df, ["local_time", "timestamp", "time", "datetime", "date_time"])

    if time_col is None:
        return pd.Series([pd.NaT] * len(df), index=df.index)

    return pd.to_datetime(df[time_col], errors="coerce")


def add_global_sec(df: pd.DataFrame, run_start: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["parsed_time"] = parse_time_column(df)

    if pd.isna(run_start) or df["parsed_time"].isna().all():
        df["sec"] = np.arange(len(df), dtype=int)
    else:
        rel = (df["parsed_time"] - run_start).dt.total_seconds()
        df["sec"] = rel.round().astype("Int64")
        df = df[df["sec"].notna()].copy()
        df["sec"] = df["sec"].astype(int)

    return df


def read_csv_basic(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Could not read {path.name}: {e}")
            return pd.DataFrame()

    df["__file"] = path.name
    return df


def read_mixed_event_raw(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["local_time", "action", "raw_detail", "raw_text", "__file"])

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        print(f"[WARN] Could not raw read {path.name}: {e}")
        return pd.DataFrame(columns=["local_time", "action", "raw_detail", "raw_text", "__file"])

    if not lines:
        return pd.DataFrame(columns=["local_time", "action", "raw_detail", "raw_text", "__file"])

    start = 1 if ("local_time" in lines[0].lower() or "timestamp" in lines[0].lower()) else 0
    rows = []

    for line in lines[start:]:
        if not line.strip():
            continue

        parts = line.split(",", 2)

        if len(parts) < 2:
            continue

        local_time = parts[0].strip().strip('"')
        action = parts[1].strip().strip('"')
        raw_detail = parts[2].strip() if len(parts) > 2 else ""

        if not action or action.lower() in {"action", "event"}:
            continue

        rows.append(
            {
                "local_time": local_time,
                "action": action,
                "raw_detail": raw_detail,
                "raw_text": (action + " " + raw_detail).lower(),
                "__file": path.name,
            }
        )

    return pd.DataFrame(rows)


def load_logs(run_name: str, folder: Path) -> Dict[str, pd.DataFrame]:
    print_header(f"LOAD LOGS: {run_name} ({folder})")

    logs = {}
    paths = {k: pick_file(folder, pattern) for k, pattern in LOG_PATTERNS.items()}

    agent = read_csv_basic(paths["agent"])

    if agent.empty:
        raise RuntimeError(f"{run_name}: missing agent_live_trajectory_log_v145*.csv")

    run_start = parse_time_column(agent).dropna().min()
    print(f"[{run_name}] run_start from agent log = {run_start}")

    for k, path in paths.items():
        df = read_mixed_event_raw(path) if k in MIXED_EVENT_LOG_KEYS else read_csv_basic(path)

        if not df.empty:
            df = add_global_sec(df, run_start)

        logs[k] = df
        print(f"{k:20s} rows={len(df):6d} file={path.name if path else '-'}")

    logs["_meta"] = {"run_start": run_start}

    return logs


def normalize_agent_name(x) -> str:
    s = str(x).upper().strip()

    if "REAL" in s:
        return "REAL"

    if "SIM1" in s:
        return "SIM1"

    if "SIM2" in s:
        return "SIM2"

    if s in ["1", "AGV1", "PHYSICAL"]:
        return "REAL"

    return s


def detect_agent_columns(agent: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "agent": safe_get_col(agent, ["agent", "agent_id", "agv", "agv_id", "vehicle", "vehicle_id", "name", "id", "label"]),
        "x": safe_get_col(agent, ["x", "pos_x", "real_x", "X"]),
        "y": safe_get_col(agent, ["y", "pos_y", "real_y", "Y"]),
        "speed": safe_get_col(agent, ["speed_mps", "speed", "abs_speed", "signed_speed"]),
        "edge": safe_get_col(agent, ["edge", "current_edge", "active_edge"]),
        "edge_u": safe_get_col(agent, ["edge_u", "u", "from", "from_node"]),
        "edge_v": safe_get_col(agent, ["edge_v", "v", "to", "to_node"]),
        "node": safe_get_col(agent, ["snap_node", "node", "current_node"]),
        "status": safe_get_col(agent, ["status", "state", "hold_reason", "reason"]),
    }


def build_edge_text(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    if cols.get("edge") is not None and cols["edge"] in df.columns:
        return df[cols["edge"]].astype(str)

    if cols.get("edge_u") is not None and cols.get("edge_v") is not None:
        u = pd.to_numeric(df[cols["edge_u"]], errors="coerce")
        v = pd.to_numeric(df[cols["edge_v"]], errors="coerce")

        return pd.Series(
            [f"{int(a)}->{int(b)}" if pd.notna(a) and pd.notna(b) else "" for a, b in zip(u, v)],
            index=df.index,
        )

    return pd.Series([""] * len(df), index=df.index)


def compact_agent(agent: pd.DataFrame, target: str, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    if cols["agent"] is None:
        raise ValueError("Cannot find agent_id/agent column.")

    df = agent.copy()
    df["agent_norm"] = df[cols["agent"]].apply(normalize_agent_name)
    df = df[df["agent_norm"] == target].copy()

    if df.empty:
        return pd.DataFrame()

    prefix = target.lower()

    out = pd.DataFrame()
    out["sec"] = df["sec"].astype(int)
    out[f"{prefix}_x"] = pd.to_numeric(df[cols["x"]], errors="coerce") if cols["x"] else np.nan
    out[f"{prefix}_y"] = pd.to_numeric(df[cols["y"]], errors="coerce") if cols["y"] else np.nan
    out[f"{prefix}_speed"] = pd.to_numeric(df[cols["speed"]], errors="coerce") if cols["speed"] else np.nan
    out[f"{prefix}_node"] = pd.to_numeric(df[cols["node"]], errors="coerce") if cols["node"] else np.nan
    out[f"{prefix}_edge"] = build_edge_text(df, cols).values
    out[f"{prefix}_status"] = df[cols["status"]].astype(str).values if cols["status"] else ""

    return out.sort_values("sec").drop_duplicates("sec", keep="last")


def fill_continuous(grid, df, col):
    tmp = grid[["sec"]].merge(df[["sec", col]], on="sec", how="left")
    s = pd.to_numeric(tmp[col], errors="coerce")
    s = s.interpolate("linear", limit=MAX_INTERPOLATE_GAP_SECONDS, limit_direction="both")
    return s.ffill().bfill()


def fill_discrete(grid, df, col, default=""):
    tmp = grid[["sec"]].merge(df[["sec", col]], on="sec", how="left")
    return tmp[col].ffill().bfill().fillna(default)


def build_agent_1hz(grid, compact, prefix):
    out = grid[["sec"]].copy()

    if compact.empty:
        for c in ["x", "y", "speed", "node"]:
            out[f"{prefix}_{c}"] = np.nan

        out[f"{prefix}_edge"] = ""
        out[f"{prefix}_status"] = ""
        return out

    for c in [f"{prefix}_x", f"{prefix}_y", f"{prefix}_speed", f"{prefix}_node"]:
        out[c] = fill_continuous(grid, compact, c) if c in compact.columns else np.nan

    for c in [f"{prefix}_edge", f"{prefix}_status"]:
        out[c] = fill_discrete(grid, compact, c, "")

    return out


def edge_key(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if not s or s.lower() in ["nan", "none", "-"]:
        return None

    for sep in ["->", "→", "-", "—", " to "]:
        if sep in s:
            p = s.split(sep)

            if len(p) >= 2:
                try:
                    return int(float(p[0].strip())), int(float(p[1].strip()))
                except Exception:
                    return None

    return None


def undirected(e):
    if e is None:
        return None

    return tuple(sorted(e))


def make_action_table(log: pd.DataFrame, candidates: List[str], out_col: str) -> pd.DataFrame:
    if log is None or log.empty:
        return pd.DataFrame(columns=["sec", out_col])

    c = safe_get_col(log, candidates)

    if c is None:
        return pd.DataFrame(columns=["sec", out_col])

    out = log[["sec", c]].copy().rename(columns={c: out_col})
    out[out_col] = out[out_col].astype(str)

    return out.sort_values("sec").drop_duplicates("sec", keep="last")


def make_flag_table(log: pd.DataFrame, flag: str, candidates: List[str], keywords: List[str]) -> pd.DataFrame:
    if log is None or log.empty:
        return pd.DataFrame(columns=["sec", flag])

    c = safe_get_col(log, candidates)

    if c is None:
        out = log[["sec"]].copy()
        out[flag] = 1
        return out.groupby("sec")[flag].max().reset_index()

    text = log[c].astype(str).str.lower()
    mask = np.zeros(len(log), dtype=bool)

    for kw in keywords:
        mask |= text.str.contains(kw.lower(), na=False).values

    out = log.loc[mask, ["sec"]].copy()
    out[flag] = 1

    if out.empty:
        return pd.DataFrame(columns=["sec", flag])

    return out.groupby("sec")[flag].max().reset_index()


def map_action(grid, table, col, default="none"):
    tmp = grid[["sec"]].merge(table, on="sec", how="left")
    return tmp[col].fillna(default).astype(str)


def map_flag(grid, table, col):
    tmp = grid[["sec"]].merge(table, on="sec", how="left")
    return tmp[col].fillna(0).astype(int)


def build_duration_flag(
    grid: pd.DataFrame,
    log: pd.DataFrame,
    flag: str,
    start_keywords: List[str],
    end_keywords: List[str],
    text_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    if log is None or log.empty or "sec" not in log.columns:
        return pd.DataFrame({"sec": grid["sec"].values, flag: 0})

    if text_candidates is None:
        text_candidates = ["action", "event", "reason", "raw_detail", "raw_text"]

    cols = [c for c in text_candidates if c in log.columns]

    if not cols:
        return pd.DataFrame({"sec": grid["sec"].values, flag: 0})

    ev = log[["sec"] + cols].copy()
    ev["_text"] = ev[cols].astype(str).agg(" ".join, axis=1).str.lower()
    ev = ev.sort_values("sec")

    start_keywords = [k.lower() for k in start_keywords]
    end_keywords = [k.lower() for k in end_keywords]

    starts = []
    ends = []

    for _, r in ev.iterrows():
        txt = r["_text"]
        sec = int(r["sec"])

        is_end = any(k in txt for k in end_keywords)
        is_start = any(k in txt for k in start_keywords) and not is_end

        if is_start:
            starts.append(sec)

        if is_end:
            ends.append(sec)

    values = []
    active = False
    start_set = set(starts)
    end_set = set(ends)

    for sec in list(map(int, grid["sec"].values)):
        if sec in start_set:
            active = True

        values.append(1 if active else 0)

        if sec in end_set:
            active = False

    return pd.DataFrame({"sec": grid["sec"].values, flag: values})


def _action_text(*values) -> str:
    return " ".join(str(v).lower().strip() for v in values if pd.notna(v))


def _action_tokens(text: str) -> set:
    s = str(text).lower()

    for sep in [",", ";", "|", "/", "\\", "\n", "\t"]:
        s = s.replace(sep, " ")

    return {t.strip() for t in s.split() if t.strip()}


def _has_any_explicit(text: str, patterns) -> bool:
    import re

    s = str(text).lower()
    tokens = _action_tokens(s)

    for p in patterns:
        p = str(p).lower().strip()

        if not p:
            continue

        if p in tokens:
            return True

        pat = r"(?<![a-z0-9_])" + re.escape(p) + r"(?![a-z0-9_])"

        if re.search(pat, s):
            return True

    return False


def build_planned_trajectory_features(logs: Dict[str, pd.DataFrame], grid: pd.DataFrame) -> pd.DataFrame:
    traj = logs.get("trajectory_sample", pd.DataFrame())
    run_start = logs["_meta"]["run_start"]

    out = grid[["sec"]].copy()

    for c in [
        "planned_real_x",
        "planned_real_y",
        "planned_real_u",
        "planned_real_v",
        "planned_real_tau",
        "planned_real_available",
    ]:
        out[c] = np.nan if c != "planned_real_available" else 0

    if traj is None or traj.empty or pd.isna(run_start):
        return out

    tcol = safe_get_col(traj, ["t_s", "ts", "trajectory_t"])
    xcol = safe_get_col(traj, ["x"])
    ycol = safe_get_col(traj, ["y"])
    ucol = safe_get_col(traj, ["u"])
    vcol = safe_get_col(traj, ["v"])
    taucol = safe_get_col(traj, ["tau"])
    agvcol = safe_get_col(traj, ["agv_id", "agent_id", "agv"])

    if tcol is None or xcol is None or ycol is None:
        return out

    tr = traj.copy()

    if agvcol is not None:
        tr = tr[tr[agvcol].astype(str).str.upper().str.contains("REAL", na=False)].copy()

    if tr.empty:
        return out

    tr["base_time"] = parse_time_column(tr)
    tr["t_s_num"] = pd.to_numeric(tr[tcol], errors="coerce").fillna(0)
    tr["abs_time"] = tr["base_time"] + pd.to_timedelta(tr["t_s_num"], unit="s")
    tr = tr[tr["abs_time"].notna()].copy()
    tr["sec"] = ((tr["abs_time"] - run_start).dt.total_seconds()).round().astype(int)

    plan = pd.DataFrame()
    plan["sec"] = tr["sec"]
    plan["planned_real_x"] = pd.to_numeric(tr[xcol], errors="coerce")
    plan["planned_real_y"] = pd.to_numeric(tr[ycol], errors="coerce")
    plan["planned_real_u"] = pd.to_numeric(tr[ucol], errors="coerce") if ucol else np.nan
    plan["planned_real_v"] = pd.to_numeric(tr[vcol], errors="coerce") if vcol else np.nan
    plan["planned_real_tau"] = pd.to_numeric(tr[taucol], errors="coerce") if taucol else np.nan
    plan["planned_real_available"] = 1

    plan = plan.sort_values("sec").drop_duplicates("sec", keep="last")

    out = grid[["sec"]].merge(plan, on="sec", how="left")

    for c in ["planned_real_x", "planned_real_y", "planned_real_tau"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").interpolate("linear", limit=2, limit_direction="both")

    for c in ["planned_real_u", "planned_real_v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill()

    out["planned_real_available"] = out["planned_real_available"].fillna(0).astype(int)

    return out


def make_state_label(row) -> str:
    safe = str(row.get("safe_node_action", "")).lower()
    tms = str(row.get("tms_action", "")).lower()
    dyn = str(row.get("dynamic_decision_action", "")).lower()
    sim_status = str(row.get("sim1_status", "")).lower()
    real_status = str(row.get("real_status", "")).lower()

    joined = " ".join([safe, tms, dyn, sim_status, real_status])
    action_sources = _action_text(safe, tms, dyn)

    if row.get("operator_event", 0) == 1 or row.get("hard_stop_operator", 0) == 1 or _has_any_explicit(joined, ["operator", "op1", "op2"]):
        return "operator_stop"

    if row.get("urgent_control_event", 0) == 1 or row.get("virtual_scanner_event", 0) == 1 or row.get("hard_stop_scanner", 0) == 1 or _has_any_explicit(joined, ["deadlock", "virtual_scanner_deadlock", "scanner_deadlock"]):
        return "deadlock"

    if _has_any_explicit(action_sources, ["preentry_block", "pre_entry_block", "preentry"]):
        return "preentry_block"

    if _has_any_explicit(action_sources, ["no_safe_node_fallback_hold", "fallback_hold", "fallback"]):
        return "fallback_hold"

    if _has_any_explicit(action_sources, ["hold_at_safe_node", "safe_node_hold", "safe_node_wait", "hold_safe_node"]):
        return "safe_node_hold"

    if _has_any_explicit(action_sources, ["hold_sim", "sim_hold", "sim_hold_live_close", "sim_hold_safe_node", "yield_sim", "sim_yield", "yield_to_real", "yield"]):
        return "fallback_hold"

    if _has_any_explicit(action_sources, ["resume_original_goal", "resume", "release", "manual_release"]):
        return "resume"

    if row.get("opposite_edge_flag", 0) == 1 or row.get("close_distance_flag", 0) == 1:
        return "conflict_risk"

    return "normal"


def extract_tms_conflict_features(tms: pd.DataFrame) -> pd.DataFrame:
    if tms is None or tms.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["sec"] = tms["sec"].astype(int)

    cc = safe_get_col(tms, ["conflict_count", "tms_conflict_count"])
    out["tms_conflict_count"] = pd.to_numeric(tms[cc], errors="coerce") if cc else 0.0

    h = safe_get_col(tms, ["horizon_s"])
    out["tms_horizon_s"] = pd.to_numeric(tms[h], errors="coerce") if h else 0.0

    hs = safe_get_col(tms, ["hold_s"])
    out["tms_hold_s"] = pd.to_numeric(tms[hs], errors="coerce") if hs else 0.0

    fc = safe_get_col(tms, ["first_conflict"])

    severities = []

    if fc:
        for val in tms[fc].astype(str):
            sev = 0.0

            try:
                d = json.loads(val)
                sev = float(d.get("severity", 0.0))
            except Exception:
                try:
                    d = ast.literal_eval(val)
                    sev = float(d.get("severity", 0.0))
                except Exception:
                    sev = 0.0

            severities.append(sev)
    else:
        severities = [0.0] * len(tms)

    out["tms_first_severity"] = severities

    return out.sort_values("sec").drop_duplicates("sec", keep="last")


def build_run_dataset(run_name: str, folder: Path) -> pd.DataFrame:
    logs = load_logs(run_name, folder)
    agent = logs["agent"]

    cols = detect_agent_columns(agent)
    print(f"\n[{run_name}] AGENT COLUMNS: {cols}")

    real_c = compact_agent(agent, "REAL", cols)
    sim_c = compact_agent(agent, "SIM1", cols)

    sec_min = int(agent["sec"].min())
    sec_max = int(agent["sec"].max())
    grid = pd.DataFrame({"sec": np.arange(sec_min, sec_max + 1)})

    print(f"[{run_name}] 1HZ GRID seconds={len(grid)} from {sec_min} to {sec_max}")

    df = grid.copy()

    df = df.merge(build_agent_1hz(grid, real_c, "real"), on="sec", how="left")
    df = df.merge(build_agent_1hz(grid, sim_c, "sim1"), on="sec", how="left")
    df = df.merge(build_planned_trajectory_features(logs, grid), on="sec", how="left")

    df["real_sim_distance_m"] = np.sqrt(
        (df["real_x"] - df["sim1_x"]) ** 2
        + (df["real_y"] - df["sim1_y"]) ** 2
    )

    df["real_edge_tuple"] = df["real_edge"].apply(edge_key)
    df["sim1_edge_tuple"] = df["sim1_edge"].apply(edge_key)

    df["real_edge_undir"] = df["real_edge_tuple"].apply(undirected)
    df["sim1_edge_undir"] = df["sim1_edge_tuple"].apply(undirected)

    df["same_physical_edge"] = (
        df["real_edge_undir"].notna()
        & (df["real_edge_undir"] == df["sim1_edge_undir"])
    ).astype(int)

    def opp(r):
        a, b = r["real_edge_tuple"], r["sim1_edge_tuple"]

        if a is None or b is None:
            return 0

        return int(a[0] == b[1] and a[1] == b[0])

    df["opposite_edge_flag"] = df.apply(opp, axis=1)

    df["close_distance_flag"] = (df["real_sim_distance_m"] < CLOSE_DISTANCE_M).astype(int)
    df["very_close_flag"] = (df["real_sim_distance_m"] < VERY_CLOSE_DISTANCE_M).astype(int)

    df["planned_to_sim_distance_m"] = np.sqrt(
        (df["planned_real_x"] - df["sim1_x"]) ** 2
        + (df["planned_real_y"] - df["sim1_y"]) ** 2
    )

    planned_undir = df.apply(
        lambda r: undirected((int(r["planned_real_u"]), int(r["planned_real_v"])))
        if pd.notna(r["planned_real_u"]) and pd.notna(r["planned_real_v"])
        else None,
        axis=1,
    )

    df["planned_same_sim_edge"] = (
        df["planned_real_u"].notna()
        & df["planned_real_v"].notna()
        & df["sim1_edge_tuple"].notna()
        & (planned_undir == df["sim1_edge_undir"])
    ).astype(int)

    df["distance_delta_1s"] = df["real_sim_distance_m"].diff().fillna(0)
    df["distance_delta_3s"] = df["real_sim_distance_m"].diff(3).fillna(0)
    df["closing_speed_est"] = -df["distance_delta_1s"]
    df["planned_distance_delta_1s"] = df["planned_to_sim_distance_m"].diff().fillna(0)

    tms_table = make_action_table(
        logs["tms_action"],
        ["action", "tms_action", "runtime_tms_action", "event"],
        "tms_action",
    )
    df["tms_action"] = map_action(grid, tms_table, "tms_action", "clear")

    tms_extra = extract_tms_conflict_features(logs["tms_action"])

    if not tms_extra.empty:
        df = df.merge(tms_extra, on="sec", how="left")

    for c in ["tms_conflict_count", "tms_first_severity", "tms_horizon_s", "tms_hold_s"]:
        if c not in df.columns:
            df[c] = 0.0

        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    dyn_table = make_action_table(
        logs["dynamic_decision"],
        ["tms_action", "runtime_tms_action", "action", "event", "decision"],
        "dynamic_decision_action",
    )
    df["dynamic_decision_action"] = map_action(grid, dyn_table, "dynamic_decision_action", "none")

    safe_table = make_action_table(
        logs["safe_node"],
        ["action", "safe_node_action", "event"],
        "safe_node_action",
    )
    df["safe_node_action"] = map_action(grid, safe_table, "safe_node_action", "none")

    emergency_log = logs["emergency"].copy()

    if emergency_log.empty:
        emergency_operator_log = emergency_log
    else:
        op_risk_col = safe_get_col(emergency_log, ["operator_risk", "op_risk", "operator", "human_risk"])
        sim_risk_col = safe_get_col(emergency_log, ["sim_risk", "simulator_risk", "sim1_risk"])

        if op_risk_col is not None:
            op_mask = emergency_log[op_risk_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])

            if sim_risk_col is not None:
                sim_mask = emergency_log[sim_risk_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
                op_mask = op_mask & ~sim_mask

            emergency_operator_log = emergency_log[op_mask].copy()

        else:
            text_cols = [c for c in ["action", "event", "reason", "raw_detail", "raw_text"] if c in emergency_log.columns]

            if text_cols:
                event_text = emergency_log[text_cols].astype(str).agg(" ".join, axis=1).str.lower()

                emergency_operator_log = emergency_log[
                    event_text.str.contains("operator|op1|op2|human|person", na=False)
                    & ~event_text.str.contains("sim_hold_live_close|sim close to real|real-sim|real sim", na=False)
                ].copy()
            else:
                emergency_operator_log = emergency_log.iloc[0:0].copy()

    emergency_flag = build_duration_flag(
        grid,
        emergency_operator_log,
        "operator_event",
        start_keywords=["emergency_brake", "operator", "op1", "op2", "human", "person"],
        end_keywords=["clear", "release", "resume"],
    )

    hard_generic_flag = make_flag_table(
        logs["hard_stop"],
        "hard_stop_event",
        ["action", "event", "reason", "raw_detail", "raw_text"],
        ["stop", "brake", "release", "safety"],
    )

    hard_operator_flag = build_duration_flag(
        grid,
        logs["hard_stop"],
        "hard_stop_operator",
        start_keywords=["op1", "op2", "operator", "human", "person"],
        end_keywords=["release", "clear", "resume"],
    )

    hard_scanner_flag = build_duration_flag(
        grid,
        logs["hard_stop"],
        "hard_stop_scanner",
        start_keywords=[
            "virtual scanner",
            "virtual_scanner",
            "real-sim",
            "real sim",
            "close distance",
            "close_distance",
            "deadlock",
            "scanner",
        ],
        end_keywords=["release", "clear", "resume"],
    )

    urgent_flag = make_flag_table(
        logs["urgent_control"],
        "urgent_control_event",
        ["action", "event", "reason", "raw_detail", "raw_text"],
        ["deadlock", "brake", "urgent", "stop"],
    )

    virt_flag = build_duration_flag(
        grid,
        logs["virtual_scanner"],
        "virtual_scanner_event",
        start_keywords=["deadlock_triggered", "virtual_scanner_deadlock", "brake", "stop_real"],
        end_keywords=["manual_release", "release", "clear", "disable"],
    )

    df["operator_event"] = map_flag(grid, emergency_flag, "operator_event")
    df["hard_stop_event"] = map_flag(grid, hard_generic_flag, "hard_stop_event")
    df["hard_stop_operator"] = map_flag(grid, hard_operator_flag, "hard_stop_operator")
    df["hard_stop_scanner"] = map_flag(grid, hard_scanner_flag, "hard_stop_scanner")
    df["urgent_control_event"] = map_flag(grid, urgent_flag, "urgent_control_event")
    df["virtual_scanner_event"] = map_flag(grid, virt_flag, "virtual_scanner_event")

    df["state_label"] = df.apply(make_state_label, axis=1)
    df["risk_binary"] = df["state_label"].isin(RISK_LABELS).astype(int)
    df["run_id"] = run_name

    for c in ["real_edge_tuple", "sim1_edge_tuple", "real_edge_undir", "sim1_edge_undir"]:
        df[c] = df[c].astype(str)

    out_path = DATA_DIR / f"{run_name}_dynamic_tms_dataset_1hz_v4.csv"
    df.to_csv(out_path, index=False)

    print(f"[{run_name}] SAVE: {out_path}")
    print(f"[{run_name}] labels:\n{df['state_label'].value_counts()}")

    return df


def prepare_model_table(df: pd.DataFrame):
    feature_cols = [
        "real_x",
        "real_y",
        "real_speed",
        "real_node",
        "sim1_x",
        "sim1_y",
        "sim1_speed",
        "sim1_node",
        "real_sim_distance_m",
        "same_physical_edge",
        "opposite_edge_flag",
        "close_distance_flag",
        "very_close_flag",
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

    df = df.copy()
    usable = []

    for c in feature_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")

            if vals.notna().mean() >= 0.15:
                df[c] = vals
                usable.append(c)

    model_df = df[["run_id", "sec", "state_label", "risk_binary"] + usable].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    for c in usable:
        med = model_df[c].median()
        model_df[c] = model_df[c].fillna(0.0 if pd.isna(med) else med)

    return model_df, usable


def split_train_test(model_df):
    train = model_df[model_df["run_id"].isin(["dataset1", "dataset2"])].copy()
    test = model_df[model_df["run_id"].isin(["dataset3"])].copy()
    return train, test


def filter_common_labels(train, test):
    train_counts = train["state_label"].value_counts()

    valid = set(train_counts[train_counts >= MIN_LABEL_COUNT_TRAIN].index).intersection(
        set(test["state_label"].unique())
    )
    valid = sorted(list(valid))

    train = train[train["state_label"].isin(valid)].copy()
    test = test[test["state_label"].isin(valid)].copy()

    if not valid or train.empty or test.empty:
        raise RuntimeError("No valid common labels after MIN_LABEL_COUNT_TRAIN filtering.")

    return train, test, valid


def save_cm(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    if name == "ExtraTrees_binary_risk":
        out_name = "fig_confusion_extratrees_binary.png"
    elif name == "ExtraTrees_v2":
        out_name = "fig_confusion_extratrees_multiclass.png"
    else:
        out_name = f"confusion_matrix_{name}.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(name)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_binary_with_extra_metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "risk_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    out["non_risk_precision"] = precision[0]
    out["non_risk_recall"] = recall[0]
    out["risk_precision"] = precision[1]
    out["risk_recall"] = recall[1]
    out["non_risk_support"] = support[0]
    out["risk_support"] = support[1]

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["roc_auc"] = np.nan

        try:
            out["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            out["pr_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def train_multiclass_models(model_df, features):
    train, test = split_train_test(model_df)
    train, test, valid_labels = filter_common_labels(train, test)

    print_header("MULTICLASS RUN-WISE SPLIT")
    print("Train rows:", len(train), " Test rows:", len(test))
    print("Labels:", valid_labels)
    print("\nTrain labels:\n", train["state_label"].value_counts())
    print("\nTest labels:\n", test["state_label"].value_counts())

    le = LabelEncoder()
    le.fit(valid_labels)

    Xtr = train[features].values
    Xte = test[features].values
    ytr = le.transform(train["state_label"].values)
    yte = le.transform(test["state_label"].values)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "RandomForest_v2": RandomForestClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
        "ExtraTrees_v2": ExtraTreesClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "HistGradientBoosting_v2": HistGradientBoostingClassifier(
            random_state=GLOBAL_SEED,
            max_iter=300,
            learning_rate=0.05,
        ),
        "MLP_v2": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=800,
            learning_rate_init=1e-3,
            random_state=GLOBAL_SEED,
        ),
    }

    results = []

    for name, model in models.items():
        if name in ["MLP_v2", "HistGradientBoosting_v2"]:
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xte_s)
        else:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        rep = classification_report(yte, pred, target_names=le.classes_, zero_division=0)

        (EXTRA_DIR / f"classification_report_{name}.txt").write_text(rep, encoding="utf-8")
        save_cm(yte, pred, le.classes_, name)

        results.append(
            {
                "task": "multiclass",
                "model": name,
                "accuracy": acc,
                "macro_f1": f1,
            }
        )

        print_header(name)
        print(rep)

    try:
        gru_res = train_gru(train, test, features, le, scaler)

        if gru_res:
            results.append(gru_res)

    except Exception as e:
        print(f"[WARN] GRU failed: {e}")

    return pd.DataFrame(results)


def make_sequences(df, features, label_col, le, scaler):
    df = df.sort_values(["run_id", "sec"]).reset_index(drop=True)

    X = scaler.transform(df[features].values).astype(np.float32)
    y = le.transform(df[label_col].values).astype(np.int64)
    runs = df["run_id"].values

    Xs, ys = [], []

    for i in range(len(df) - WINDOW_SIZE):
        if len(set(runs[i:i + WINDOW_SIZE + 1])) != 1:
            continue

        Xs.append(X[i:i + WINDOW_SIZE])
        ys.append(y[i + WINDOW_SIZE])

    if not Xs:
        return (
            np.empty((0, WINDOW_SIZE, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return np.stack(Xs), np.array(ys)


def train_gru(train, test, features, le, scaler):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(GLOBAL_SEED)

    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)

    Xtr, ytr = make_sequences(train, features, "state_label", le, scaler)
    Xte, yte = make_sequences(test, features, "state_label", le, scaler)

    print_header("GRU DATA V4")
    print("Train:", Xtr.shape, "Test:", Xte.shape)

    if len(Xtr) < 50 or len(Xte) < 20:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.arange(len(le.classes_))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    class Net(nn.Module):
        def __init__(self, in_dim, ncls):
            super().__init__()
            self.gru = nn.GRU(in_dim, 96, num_layers=2, batch_first=True, dropout=0.25)
            self.head = nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, ncls),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    model = Net(len(features), len(le.classes_)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights_t)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
        batch_size=64,
        shuffle=False,
    )

    hist = []

    for epoch in range(1, 81):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        hist.append(float(np.mean(losses)))

        if epoch % 10 == 0:
            print(f"[GRU V4] epoch={epoch:03d} loss={hist[-1]:.4f}")

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)

    rep = classification_report(trues, preds, target_names=le.classes_, zero_division=0)

    (EXTRA_DIR / "classification_report_GRU_v4.txt").write_text(rep, encoding="utf-8")
    save_cm(np.array(trues), np.array(preds), le.classes_, "GRU_v4")

    plt.figure(figsize=(8, 5))
    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("GRU Dynamic Neural TMS V4 Training Loss")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_history_gru_v4.png", dpi=300, bbox_inches="tight")
    plt.close()

    torch.save(model.state_dict(), MODELS_DIR / "gru_dynamic_tms_v4.pt")

    print_header("GRU V4")
    print(rep)

    return {
        "task": "multiclass",
        "model": "GRU_dynamic_sequence_v4",
        "accuracy": acc,
        "macro_f1": f1,
    }


def train_binary_risk_model(model_df, features):
    train, test = split_train_test(model_df)

    Xtr = train[features].values
    Xte = test[features].values

    ytr = train["risk_binary"].values.astype(int)
    yte = test["risk_binary"].values.astype(int)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "RF_binary_risk": RandomForestClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
        "ExtraTrees_binary_risk": ExtraTreesClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "MLP_binary_risk": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=800,
            random_state=GLOBAL_SEED,
        ),
    }

    rows = []
    labels = ["non_risk", "risk_action"]

    for name, model in models.items():
        if "MLP" in name:
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xte_s)
            score = model.predict_proba(Xte_s)[:, 1] if hasattr(model, "predict_proba") else None
        else:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            score = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_binary_with_extra_metrics(yte, pred, score)

        rep = classification_report(yte, pred, target_names=labels, zero_division=0)

        (EXTRA_DIR / f"classification_report_{name}.txt").write_text(rep, encoding="utf-8")
        save_cm(yte, pred, labels, name)

        row = {
            "task": "binary_risk",
            "model": name,
        }
        row.update(metrics)
        rows.append(row)

        print_header(name)
        print(rep)

    return pd.DataFrame(rows)


def save_summaries(combined):
    rows = []

    for run_id, g in combined.groupby("run_id"):
        rows.append(
            {
                "run_id": run_id,
                "rows_1hz": len(g),
                "duration_s": int(g["sec"].max() - g["sec"].min()),
                "min_distance_m": float(g["real_sim_distance_m"].min()),
                "mean_distance_m": float(g["real_sim_distance_m"].mean()),
                "close_events_lt_0_30m": int((g["real_sim_distance_m"] < CLOSE_DISTANCE_M).sum()),
                "very_close_events_lt_0_10m": int((g["real_sim_distance_m"] < VERY_CLOSE_DISTANCE_M).sum()),
                "same_physical_edge_events": int(g["same_physical_edge"].sum()),
                "opposite_edge_events": int(g["opposite_edge_flag"].sum()),
                "planned_available_rows": int(g["planned_real_available"].sum()),
                "operator_events": int(g["operator_event"].sum()),
                "urgent_control_events": int(g["urgent_control_event"].sum()),
                "virtual_scanner_events": int(g["virtual_scanner_event"].sum()),
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "runtime_summary_by_run_v4.csv", index=False)

    combined.groupby(["run_id", "state_label"]).size().reset_index(name="count").to_csv(
        RESULTS_DIR / "label_distribution_by_run_v4.csv",
        index=False,
    )

    combined["state_label"].value_counts().rename_axis("state_label").reset_index(name="count").to_csv(
        RESULTS_DIR / "label_distribution_combined_v4.csv",
        index=False,
    )


def _fit_predict_for_task(train, test, features, label_col, model_name="ExtraTrees"):
    Xtr = train[features].values
    Xte = test[features].values

    ytr_raw = train[label_col].values
    yte_raw = test[label_col].values

    if label_col == "state_label":
        train_counts = Counter(ytr_raw)

        valid = sorted(
            list(
                {
                    k
                    for k, v in train_counts.items()
                    if v >= MIN_LABEL_COUNT_TRAIN
                }.intersection(set(yte_raw))
            )
        )

        train = train[train[label_col].isin(valid)].copy()
        test = test[test[label_col].isin(valid)].copy()

        if not valid or train.empty or test.empty:
            raise RuntimeError("No valid common labels after MIN_LABEL_COUNT_TRAIN filtering.")

        Xtr = train[features].values
        Xte = test[features].values

        ytr_raw = train[label_col].values
        yte_raw = test[label_col].values

        le = LabelEncoder()
        le.fit(valid)

        ytr = le.transform(ytr_raw)
        yte = le.transform(yte_raw)

    else:
        ytr = ytr_raw.astype(int)
        yte = yte_raw.astype(int)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        score = model.predict_proba(Xte)[:, 1] if label_col != "state_label" and hasattr(model, "predict_proba") else None

    elif model_name == "ExtraTrees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        score = model.predict_proba(Xte)[:, 1] if label_col != "state_label" and hasattr(model, "predict_proba") else None

    elif model_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(
            random_state=GLOBAL_SEED,
            max_iter=300,
            learning_rate=0.05,
        )
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        score = model.predict_proba(Xte_s)[:, 1] if label_col != "state_label" and hasattr(model, "predict_proba") else None

    elif model_name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=800,
            learning_rate_init=1e-3,
            random_state=GLOBAL_SEED,
        )
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        score = model.predict_proba(Xte_s)[:, 1] if label_col != "state_label" and hasattr(model, "predict_proba") else None

    else:
        raise ValueError(f"Unknown model_name={model_name}")

    if label_col == "state_label":
        metrics = {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "n_train": len(train),
            "n_test": len(test),
        }
    else:
        metrics = evaluate_binary_with_extra_metrics(yte, pred, score)
        metrics["n_train"] = len(train)
        metrics["n_test"] = len(test)

    return metrics, model, scaler


def run_leave_one_run_out(model_df, features):
    print_header("EXTRA VALIDATION: LEAVE-ONE-RUN-OUT")

    runs = sorted(model_df["run_id"].unique())
    all_rows = []

    for test_run in runs:
        train = model_df[model_df["run_id"] != test_run].copy()
        test = model_df[model_df["run_id"] == test_run].copy()

        for task, label_col in [("binary_risk", "risk_binary"), ("multiclass", "state_label")]:
            scores, _, _ = _fit_predict_for_task(train, test, features, label_col, model_name="ExtraTrees")

            row = {
                "task": task,
                "model": "ExtraTrees",
                "train_runs": "+".join([r for r in runs if r != test_run]),
                "test_run": test_run,
            }
            row.update(scores)
            all_rows.append(row)

            print(
                f"[{task}] train={row['train_runs']} test={test_run} "
                f"acc={scores['accuracy']:.4f} macro_f1={scores['macro_f1']:.4f}"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "leave_one_run_out_results_v4.csv", index=False)

    summary = df.groupby(["task", "model"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).reset_index()
    summary.to_csv(RESULTS_DIR / "leave_one_run_out_summary_v4.csv", index=False)

    return df, summary


def get_feature_groups(features):
    groups = {
        "Live spatial only": [
            "real_x",
            "real_y",
            "real_speed",
            "real_node",
            "sim1_x",
            "sim1_y",
            "sim1_speed",
            "sim1_node",
            "real_sim_distance_m",
            "close_distance_flag",
            "very_close_flag",
        ],
        "Live + graph relations": [
            "real_x",
            "real_y",
            "real_speed",
            "real_node",
            "sim1_x",
            "sim1_y",
            "sim1_speed",
            "sim1_node",
            "real_sim_distance_m",
            "close_distance_flag",
            "very_close_flag",
            "same_physical_edge",
            "opposite_edge_flag",
        ],
        "Live + trajectory prior": [
            "real_x",
            "real_y",
            "real_speed",
            "real_node",
            "sim1_x",
            "sim1_y",
            "sim1_speed",
            "sim1_node",
            "real_sim_distance_m",
            "close_distance_flag",
            "very_close_flag",
            "same_physical_edge",
            "opposite_edge_flag",
            "planned_real_x",
            "planned_real_y",
            "planned_real_u",
            "planned_real_v",
            "planned_real_tau",
            "planned_real_available",
            "planned_to_sim_distance_m",
            "planned_same_sim_edge",
            "planned_distance_delta_1s",
        ],
        "Full event-free knowledge": list(features),
    }

    return {k: [c for c in v if c in features] for k, v in groups.items()}


def run_feature_ablation(model_df, features):
    print_header("EXTRA VALIDATION: FEATURE-GROUP ABLATION")

    train, test = split_train_test(model_df)
    groups = get_feature_groups(features)
    rows = []

    for task, label_col in [("binary_risk", "risk_binary"), ("multiclass", "state_label")]:
        for group_name, cols in groups.items():
            if not cols:
                continue

            scores, _, _ = _fit_predict_for_task(train, test, cols, label_col, model_name="ExtraTrees")

            row = {
                "task": task,
                "model": "ExtraTrees",
                "feature_set": group_name,
                "n_features": len(cols),
            }
            row.update(scores)
            rows.append(row)

            print(
                f"[{task}] {group_name:28s} features={len(cols):2d} "
                f"acc={scores['accuracy']:.4f} macro_f1={scores['macro_f1']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "feature_group_ablation_v4.csv", index=False)

    return df


def rebuild_model_table_for_thresholds(combined, close_thr, very_close_thr):
    tmp = combined.copy()

    tmp["close_distance_flag"] = (tmp["real_sim_distance_m"] < close_thr).astype(int)
    tmp["very_close_flag"] = (tmp["real_sim_distance_m"] < very_close_thr).astype(int)

    tmp["state_label"] = tmp.apply(make_state_label, axis=1)
    tmp["risk_binary"] = tmp["state_label"].isin(RISK_LABELS).astype(int)

    return prepare_model_table(tmp)


def run_threshold_sensitivity(combined):
    print_header("EXTRA VALIDATION: THRESHOLD SENSITIVITY")

    rows = []

    for close_thr, very_thr in [(0.20, 0.05), (0.30, 0.10), (0.40, 0.15)]:
        model_df_t, features_t = rebuild_model_table_for_thresholds(combined, close_thr, very_thr)
        event_free_features_t = [c for c in features_t if c not in EVENT_AWARE_FEATURES]

        train, test = split_train_test(model_df_t)

        scores, _, _ = _fit_predict_for_task(
            train,
            test,
            event_free_features_t,
            "risk_binary",
            model_name="ExtraTrees",
        )

        row = {
            "close_threshold_m": close_thr,
            "very_close_threshold_m": very_thr,
            "model": "ExtraTrees",
        }
        row.update(scores)
        rows.append(row)

        print(
            f"thresholds close={close_thr:.2f} very_close={very_thr:.2f} "
            f"acc={scores['accuracy']:.4f} macro_f1={scores['macro_f1']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_sensitivity_v4.csv", index=False)

    return df


def run_runtime_latency(model_df, features):
    print_header("EXTRA VALIDATION: RUNTIME LATENCY")

    train, test = split_train_test(model_df)
    rows = []

    for model_name in ["RandomForest", "ExtraTrees", "HistGradientBoosting", "MLP"]:
        scores, model, scaler = _fit_predict_for_task(
            train,
            test,
            features,
            "risk_binary",
            model_name=model_name,
        )

        X_one = test[features].iloc[[0]].values

        X_pred = scaler.transform(X_one) if model_name in ["HistGradientBoosting", "MLP"] else X_one

        for _ in range(20):
            model.predict(X_pred)

        n = 500
        t0 = time.perf_counter()

        for _ in range(n):
            model.predict(X_pred)

        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0 / n

        rows.append(
            {
                "model": model_name,
                "mean_inference_time_ms_per_sample": ms,
                "suitable_for_1Hz": "Yes" if ms < 1000 else "No",
                "binary_accuracy": scores["accuracy"],
                "binary_macro_f1": scores["macro_f1"],
            }
        )

        print(f"{model_name:22s} latency={ms:.4f} ms/sample suitable_1Hz={'Yes' if ms < 1000 else 'No'}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "runtime_latency_v4.csv", index=False)

    return df


def build_future_horizon_table(model_df, horizons):
    rows = []
    base = model_df.sort_values(["run_id", "sec"]).copy()

    for run_id, group in base.groupby("run_id", sort=False):
        group = group.sort_values("sec").copy()

        for horizon in horizons:
            temp = group.copy()

            if horizon == 0:
                temp["future_risk_binary"] = temp["risk_binary"]
                temp["future_state_label"] = temp["state_label"]
            else:
                temp["future_risk_binary"] = temp["risk_binary"].shift(-horizon)
                temp["future_state_label"] = temp["state_label"].shift(-horizon)

            temp["prediction_horizon_s"] = horizon
            temp = temp.dropna(subset=["future_risk_binary", "future_state_label"]).copy()
            temp["future_risk_binary"] = temp["future_risk_binary"].astype(int)

            rows.append(temp)

    if not rows:
        raise RuntimeError("No future-horizon rows were created.")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA_DIR / "future_horizon_model_table_v1.csv", index=False)

    return out


def run_future_horizon_prediction(model_df, event_free_features):
    print_header("EXTRA VALIDATION: EVENT-FREE FUTURE-HORIZON RISK PREDICTION")

    horizon_df = build_future_horizon_table(model_df, FUTURE_HORIZONS_S)
    rows = []

    for horizon in FUTURE_HORIZONS_S:
        df_h = horizon_df[horizon_df["prediction_horizon_s"] == horizon].copy()

        train = df_h[df_h["run_id"].isin(["dataset1", "dataset2"])].copy()
        test = df_h[df_h["run_id"].isin(["dataset3"])].copy()

        if train.empty or test.empty:
            print(f"[SKIP] Horizon {horizon}s has empty train/test.")
            continue

        Xtr = train[event_free_features].values
        Xte = test[event_free_features].values

        ytr = train["future_risk_binary"].values.astype(int)
        yte = test["future_risk_binary"].values.astype(int)

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            print(f"[SKIP] Horizon {horizon}s has fewer than two classes.")
            continue

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=500,
                random_state=GLOBAL_SEED,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=500,
                random_state=GLOBAL_SEED,
                class_weight="balanced",
                min_samples_leaf=2,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                random_state=GLOBAL_SEED,
                max_iter=300,
                learning_rate=0.05,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=800,
                learning_rate_init=1e-3,
                random_state=GLOBAL_SEED,
            ),
        }

        for model_name, model in models.items():
            if model_name in ["HistGradientBoosting", "MLP"]:
                model.fit(Xtr_s, ytr)
                pred = model.predict(Xte_s)
                score = model.predict_proba(Xte_s)[:, 1] if hasattr(model, "predict_proba") else None
            else:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                score = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = evaluate_binary_with_extra_metrics(yte, pred, score)

            report = classification_report(
                yte,
                pred,
                target_names=["non_risk", "risk_action"],
                zero_division=0,
            )

            row = {
                "setting": "event_free_future_horizon",
                "task": "future_binary_risk",
                "prediction_horizon_s": horizon,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
            }
            row.update(metrics)
            rows.append(row)

            report_path = EXTRA_DIR / f"classification_report_future_binary_{model_name}_{horizon}s.txt"
            report_path.write_text(report, encoding="utf-8")

            if model_name == "ExtraTrees":
                cm = confusion_matrix(yte, pred, labels=[0, 1])

                plt.figure(figsize=(6, 5))
                plt.imshow(cm, interpolation="nearest")
                plt.title(f"Event-free future risk prediction ({horizon}s)")
                plt.colorbar()
                plt.xticks([0, 1], ["non_risk", "risk_action"], rotation=25, ha="right")
                plt.yticks([0, 1], ["non_risk", "risk_action"])

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)

                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"fig_future_binary_confusion_extratrees_{horizon}s.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            print(
                f"[future {horizon}s] {model_name:20s} "
                f"acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"risk_recall={metrics['risk_recall']:.4f}"
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "future_horizon_binary_risk_results_v1.csv", index=False)

    if not result_df.empty:
        best_df = (
            result_df.sort_values(
                ["prediction_horizon_s", "macro_f1"],
                ascending=[True, False],
            )
            .groupby("prediction_horizon_s")
            .head(1)
            .reset_index(drop=True)
        )

        best_df.to_csv(RESULTS_DIR / "future_horizon_binary_risk_best_v1.csv", index=False)
        plot_future_horizon_results(result_df)

    return result_df


def plot_future_horizon_results(result_df):
    if result_df is None or result_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for model_name, group in result_df.groupby("model"):
        group = group.sort_values("prediction_horizon_s")
        plt.plot(
            group["prediction_horizon_s"],
            group["macro_f1"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.05)
    plt.title("Event-free future risk/action prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_future_horizon_binary_risk.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_future_distance_targets(model_df, horizons):
    if "real_sim_distance_m" not in model_df.columns:
        raise RuntimeError("real_sim_distance_m not found in model_df.")

    rows = []
    base = model_df.sort_values(["run_id", "sec"]).copy()

    for run_id, group in base.groupby("run_id", sort=False):
        group = group.sort_values("sec").copy()
        dist = group["real_sim_distance_m"].values.astype(float)

        for horizon in horizons:
            temp = group.copy()
            min_dists = []

            for i in range(len(group)):
                j_end = min(i + horizon + 1, len(group))
                window = dist[i:j_end]

                if len(window) == 0 or np.all(np.isnan(window)):
                    min_dists.append(np.nan)
                else:
                    min_dists.append(np.nanmin(window))

            temp["prediction_horizon_s"] = horizon
            temp["min_future_distance_m"] = min_dists
            temp = temp.dropna(subset=["min_future_distance_m"]).copy()

            rows.append(temp)

    if not rows:
        raise RuntimeError("No future-distance rows were created.")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA_DIR / "future_distance_regression_table_v1.csv", index=False)

    return out


def run_future_distance_regression(model_df, event_free_features):
    print_header("EXTRA VALIDATION: EVENT-FREE FUTURE DISTANCE REGRESSION")

    horizons = [1, 3, 5]
    reg_df = build_future_distance_targets(model_df, horizons)
    rows = []

    for horizon in horizons:
        df_h = reg_df[reg_df["prediction_horizon_s"] == horizon].copy()

        train = df_h[df_h["run_id"].isin(["dataset1", "dataset2"])].copy()
        test = df_h[df_h["run_id"].isin(["dataset3"])].copy()

        if train.empty or test.empty:
            print(f"[SKIP] Distance horizon {horizon}s has empty train/test.")
            continue

        Xtr = train[event_free_features].values
        Xte = test[event_free_features].values

        ytr = train["min_future_distance_m"].values.astype(float)
        yte = test["min_future_distance_m"].values.astype(float)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        models = {
            "ExtraTreesRegressor": ExtraTreesRegressor(
                n_estimators=500,
                random_state=GLOBAL_SEED,
                min_samples_leaf=2,
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=500,
                random_state=GLOBAL_SEED,
                min_samples_leaf=2,
            ),
            "MLPRegressor": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=800,
                learning_rate_init=1e-3,
                random_state=GLOBAL_SEED,
            ),
        }

        for model_name, model in models.items():
            if model_name == "MLPRegressor":
                model.fit(Xtr_s, ytr)
                pred = model.predict(Xte_s)
            else:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)

            mae = mean_absolute_error(yte, pred)
            rmse = float(np.sqrt(mean_squared_error(yte, pred)))
            r2 = r2_score(yte, pred)

            rows.append(
                {
                    "setting": "event_free_future_distance",
                    "task": "future_distance_regression",
                    "prediction_horizon_s": horizon,
                    "model": model_name,
                    "mae_m": mae,
                    "rmse_m": rmse,
                    "r2": r2,
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )

            if model_name == "ExtraTreesRegressor":
                plt.figure(figsize=(6, 5))
                plt.scatter(yte, pred, s=12, alpha=0.7)

                min_val = min(float(np.min(yte)), float(np.min(pred)))
                max_val = max(float(np.max(yte)), float(np.max(pred)))

                plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
                plt.xlabel("True minimum future distance (m)")
                plt.ylabel("Predicted minimum future distance (m)")
                plt.title(f"Future distance prediction ({horizon}s)")
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"fig_future_distance_regression_extratrees_{horizon}s.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            print(
                f"[distance {horizon}s] {model_name:22s} "
                f"MAE={mae:.4f}m RMSE={rmse:.4f}m R2={r2:.4f}"
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "future_distance_regression_results_v1.csv", index=False)

    if not result_df.empty:
        plot_future_distance_results(result_df)

    return result_df


def plot_future_distance_results(result_df):
    if result_df is None or result_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for model_name, group in result_df.groupby("model"):
        group = group.sort_values("prediction_horizon_s")
        plt.plot(
            group["prediction_horizon_s"],
            group["mae_m"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("MAE of minimum future distance (m)")
    plt.title("Event-free future distance regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_future_distance_regression_mae.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_event_feature_leakage_audit(model_df, all_features, event_free_features):
    print_header("EXTRA VALIDATION: EVENT-FEATURE LEAKAGE AUDIT")

    event_only_features = [c for c in all_features if c in EVENT_AWARE_FEATURES]

    feature_sets = {
        "event_only": event_only_features,
        "event_free": event_free_features,
        "all_features": all_features,
    }

    rows = []

    for feature_set_name, cols in feature_sets.items():
        if not cols:
            print(f"[SKIP] {feature_set_name}: no features.")
            continue

        train, test = split_train_test(model_df)

        Xtr = train[cols].values
        Xte = test[cols].values

        ytr = train["risk_binary"].values.astype(int)
        yte = test["risk_binary"].values.astype(int)

        model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
            min_samples_leaf=2,
        )

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        y_score = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_binary_with_extra_metrics(yte, pred, y_score)

        row = {
            "task": "binary_risk",
            "model": "ExtraTrees",
            "feature_set": feature_set_name,
            "n_features": len(cols),
        }
        row.update(metrics)
        rows.append(row)

        print(
            f"[{feature_set_name}] features={len(cols)} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"risk_recall={metrics['risk_recall']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "event_feature_leakage_audit_v1.csv", index=False)

    return df


def run_decision_policy_evaluation(model_df, event_free_features):
    print_header("EXTRA VALIDATION: DECISION-LEVEL TMS POLICY EVALUATION")

    train, test = split_train_test(model_df)

    Xtr = train[event_free_features].values
    Xte = test[event_free_features].values

    ytr = train["risk_binary"].values.astype(int)
    yte = test["risk_binary"].values.astype(int)

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=GLOBAL_SEED,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]

    rows = []

    for high_thr in [0.50, 0.60, 0.70, 0.80]:
        pred = (prob >= high_thr).astype(int)
        metrics = evaluate_binary_with_extra_metrics(yte, pred, prob)

        row = {
            "policy": "binary_hold_if_probability_above_threshold",
            "high_risk_threshold": high_thr,
        }
        row.update(metrics)
        rows.append(row)

        print(
            f"[threshold {high_thr:.2f}] "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"risk_recall={metrics['risk_recall']:.4f} "
            f"risk_precision={metrics['risk_precision']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "decision_policy_threshold_results_v1.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["high_risk_threshold"], df["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(df["high_risk_threshold"], df["risk_recall"], marker="s", label="Risk recall")
    plt.plot(df["high_risk_threshold"], df["risk_precision"], marker="^", label="Risk precision")
    plt.xlabel("High-risk probability threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Decision-level TMS policy sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_decision_policy_thresholds.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df


def _parse_edge_tuple_any(x):
    if pd.isna(x):
        return None

    if isinstance(x, tuple) and len(x) == 2:
        try:
            return int(x[0]), int(x[1])
        except Exception:
            return None

    s = str(x).strip()

    if not s or s.lower() in ["none", "nan", "", "-", "nat"]:
        return None

    try:
        val = ast.literal_eval(s)

        if isinstance(val, tuple) and len(val) == 2:
            return int(val[0]), int(val[1])

    except Exception:
        pass

    return edge_key(s)


def build_topology_from_combined(combined):
    edges = set()
    nodes = set()

    for col in ["real_edge_tuple", "sim1_edge_tuple"]:
        if col in combined.columns:
            for val in combined[col].dropna().values:
                e = _parse_edge_tuple_any(val)

                if e is not None:
                    edges.add(e)
                    nodes.update(e)

    if "planned_real_u" in combined.columns and "planned_real_v" in combined.columns:
        for u, v in zip(combined["planned_real_u"], combined["planned_real_v"]):
            if pd.notna(u) and pd.notna(v):
                try:
                    e = int(float(u)), int(float(v))
                    edges.add(e)
                    nodes.update(e)
                except Exception:
                    pass

    for col in ["real_node", "sim1_node"]:
        if col in combined.columns:
            for n in pd.to_numeric(combined[col], errors="coerce").dropna().values:
                nodes.add(int(n))

    if not nodes:
        raise RuntimeError("No topology nodes found for GNN baseline.")

    raw_node_list = sorted(nodes)
    node_to_idx = {n: i + 1 for i, n in enumerate(raw_node_list)}

    node_list = ["UNKNOWN"] + raw_node_list
    n = len(node_list)

    A = np.eye(n, dtype=np.float32)

    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0

    deg = A.sum(axis=1)

    deg_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    A_norm = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

    node_ids = np.array(raw_node_list, dtype=np.float32)
    node_id_norm_real = node_ids / max(float(node_ids.max()), 1.0)
    degree_norm = deg / max(float(deg.max()), 1.0)

    X_real = np.stack([node_id_norm_real, degree_norm[1:]], axis=1).astype(np.float32)
    X_node = np.vstack([np.zeros((1, 2), dtype=np.float32), X_real]).astype(np.float32)

    return node_list, node_to_idx, A_norm.astype(np.float32), X_node


def _nodes_to_indices(values, node_to_idx):
    out = []

    for v in values:
        if pd.isna(v):
            out.append(0)
            continue

        try:
            out.append(node_to_idx.get(int(float(v)), 0))
        except Exception:
            out.append(0)

    return np.array(out, dtype=np.int64)


def _prepare_topology_task_data(model_df, features, label_col, node_to_idx):
    df = model_df.copy()

    if label_col == "state_label":
        train, test = split_train_test(df)
        train, test, valid_labels = filter_common_labels(train, test)

        le = LabelEncoder()
        le.fit(valid_labels)

        ytr = le.transform(train[label_col].values).astype(np.int64)
        yte = le.transform(test[label_col].values).astype(np.int64)
        class_names = list(le.classes_)

    else:
        train, test = split_train_test(df)

        ytr = train[label_col].values.astype(np.int64)
        yte = test[label_col].values.astype(np.int64)
        class_names = ["non_risk", "risk_action"]

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(train[features].values).astype(np.float32)
    Xte = scaler.transform(test[features].values).astype(np.float32)

    tr_nodes = np.stack(
        [
            _nodes_to_indices(train.get("real_node", pd.Series(np.nan, index=train.index)), node_to_idx),
            _nodes_to_indices(train.get("sim1_node", pd.Series(np.nan, index=train.index)), node_to_idx),
            _nodes_to_indices(train.get("planned_real_u", pd.Series(np.nan, index=train.index)), node_to_idx),
            _nodes_to_indices(train.get("planned_real_v", pd.Series(np.nan, index=train.index)), node_to_idx),
        ],
        axis=1,
    )

    te_nodes = np.stack(
        [
            _nodes_to_indices(test.get("real_node", pd.Series(np.nan, index=test.index)), node_to_idx),
            _nodes_to_indices(test.get("sim1_node", pd.Series(np.nan, index=test.index)), node_to_idx),
            _nodes_to_indices(test.get("planned_real_u", pd.Series(np.nan, index=test.index)), node_to_idx),
            _nodes_to_indices(test.get("planned_real_v", pd.Series(np.nan, index=test.index)), node_to_idx),
        ],
        axis=1,
    )

    return Xtr, Xte, ytr, yte, tr_nodes, te_nodes, class_names


def train_topology_graph_family_task(model_df, combined, features, label_col, task_name, variant="GCN"):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        print(f"[WARN] PyTorch unavailable; skipping topology-aware {variant} baseline: {e}")
        return None

    set_global_seed(GLOBAL_SEED)

    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)

    variant = str(variant).strip()

    if variant not in {"GCN", "GGNN", "GAT", "GraphSAGE"}:
        raise ValueError(f"Unknown topology GNN variant: {variant}")

    node_list, node_to_idx, A_norm, X_node = build_topology_from_combined(combined)

    Xtr, Xte, ytr, yte, tr_nodes, te_nodes, class_names = _prepare_topology_task_data(
        model_df,
        features,
        label_col,
        node_to_idx,
    )

    if len(Xtr) < 30 or len(Xte) < 10 or len(np.unique(ytr)) < 2:
        print(f"[WARN] Not enough data for topology-aware {variant} baseline ({task_name}).")
        return None

    print_header(f"TOPOLOGY-AWARE {variant} BASELINE: {task_name}")
    print(f"Topology nodes={len(node_list)} edges-derived adjacency shape={A_norm.shape}")
    print(f"Train={Xtr.shape} Test={Xte.shape} Classes={class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)
    Xnode_t = torch.tensor(X_node, dtype=torch.float32, device=device)

    class TopologyGraphFamilyClassifier(nn.Module):
        def __init__(self, tab_dim, node_feat_dim, hidden_dim, emb_dim, n_classes, variant_name):
            super().__init__()

            self.variant_name = variant_name
            self.hidden_dim = hidden_dim

            self.in_proj = nn.Linear(node_feat_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, emb_dim)

            self.act = nn.ReLU()
            self.dropout = nn.Dropout(0.25)

            self.ggnn_msg = nn.Linear(hidden_dim, hidden_dim)
            self.ggnn_gru = nn.GRUCell(hidden_dim, hidden_dim)

            self.gat_a = nn.Linear(2 * hidden_dim, 1, bias=False)

            self.sage_self = nn.Linear(hidden_dim, hidden_dim)
            self.sage_neigh = nn.Linear(hidden_dim, hidden_dim)

            self.head = nn.Sequential(
                nn.Linear(tab_dim + 4 * emb_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def _gcn_embeddings(self, A, Xn):
            H = self.act(A @ self.in_proj(Xn))
            H = self.dropout(H)
            H = self.act(A @ H)
            return self.act(self.out_proj(H))

        def _ggnn_embeddings(self, A, Xn, steps=3):
            H = self.in_proj(Xn)

            for _ in range(steps):
                M = A @ self.ggnn_msg(H)
                H = self.ggnn_gru(M, H)

            return self.act(self.out_proj(self.dropout(H)))

        def _gat_embeddings(self, A, Xn):
            H = self.act(self.in_proj(Xn))
            n = H.shape[0]

            hi = H.unsqueeze(1).expand(n, n, self.hidden_dim)
            hj = H.unsqueeze(0).expand(n, n, self.hidden_dim)

            e = torch.cat([hi, hj], dim=-1)

            scores = torch.nn.functional.leaky_relu(
                self.gat_a(e).squeeze(-1),
                negative_slope=0.2,
            )

            scores = scores.masked_fill(~(A > 0), -1e9)
            att = torch.softmax(scores, dim=1)

            return self.act(self.out_proj(self.dropout(self.act(att @ H))))

        def _graphsage_embeddings(self, A, Xn):
            H = self.act(self.in_proj(Xn))
            diag = torch.diagonal(A).unsqueeze(1)
            neigh = A @ H - H * diag
            H2 = self.act(self.sage_self(H) + self.sage_neigh(neigh))
            return self.act(self.out_proj(self.dropout(H2)))

        def node_embeddings(self, A, Xn):
            if self.variant_name == "GCN":
                return self._gcn_embeddings(A, Xn)

            if self.variant_name == "GGNN":
                return self._ggnn_embeddings(A, Xn)

            if self.variant_name == "GAT":
                return self._gat_embeddings(A, Xn)

            if self.variant_name == "GraphSAGE":
                return self._graphsage_embeddings(A, Xn)

            raise ValueError(self.variant_name)

        def forward(self, tab_x, node_idx, A, Xn):
            H = self.node_embeddings(A, Xn)
            emb = H[node_idx.reshape(-1)].reshape(node_idx.shape[0], -1)
            return self.head(torch.cat([tab_x, emb], dim=1))

    n_classes = len(np.unique(ytr)) if label_col != "state_label" else len(class_names)

    model = TopologyGraphFamilyClassifier(
        len(features),
        X_node.shape[1],
        32,
        16,
        n_classes,
        variant,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)

    classes = np.arange(n_classes)

    try:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
        crit = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    except Exception:
        crit = torch.nn.CrossEntropyLoss()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(tr_nodes), torch.tensor(ytr)),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xte), torch.tensor(te_nodes), torch.tensor(yte)),
        batch_size=128,
        shuffle=False,
    )

    hist = []
    best_state = None
    best_loss = float("inf")
    patience = 25
    no_improve = 0

    for epoch in range(1, 141):
        model.train()
        losses = []

        for xb, nb, yb in train_loader:
            xb, nb, yb = xb.to(device), nb.to(device), yb.to(device)

            opt.zero_grad()
            loss = crit(model(xb, nb, A_t, Xnode_t), yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            opt.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        hist.append(epoch_loss)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"[Topology{variant} {task_name}] epoch={epoch:03d} loss={epoch_loss:.4f}")

        if no_improve >= patience and epoch >= 60:
            print(f"[Topology{variant} {task_name}] early_stop epoch={epoch:03d} best_loss={best_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, nb, yb in test_loader:
            logits = model(xb.to(device), nb.to(device), A_t, Xnode_t)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)

    report = classification_report(trues, preds, target_names=class_names, zero_division=0)

    safe_name = f"Topology{variant}_{task_name}"

    (EXTRA_DIR / f"classification_report_{safe_name}.txt").write_text(report, encoding="utf-8")

    save_cm(np.array(trues), np.array(preds), class_names, safe_name)

    plt.figure(figsize=(8, 5))
    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"Topology-aware {variant} training loss ({task_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"training_history_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": features,
            "node_list": node_list,
            "class_names": class_names,
            "task_name": task_name,
            "variant": variant,
            "note": "Topology-aware TMS-state/action recognition baseline, not ESWA edge-cost model.",
        },
        MODELS_DIR / f"{safe_name}.pt",
    )

    print(report)

    return {
        "task": task_name,
        "model": f"Topology{variant}_baseline",
        "accuracy": acc,
        "macro_f1": f1,
    }


def train_topology_gnn_baselines(model_df, combined, features):
    rows = []

    for task_name, label_col in [("multiclass", "state_label"), ("binary_risk", "risk_binary")]:
        for variant in ["GCN", "GGNN", "GAT", "GraphSAGE"]:
            try:
                res = train_topology_graph_family_task(
                    model_df,
                    combined,
                    features,
                    label_col,
                    task_name,
                    variant=variant,
                )

                if res is not None:
                    rows.append(res)

            except Exception as e:
                print(f"[WARN] Topology{variant} failed for {task_name}: {e}")

    return pd.DataFrame(rows)


def _safe_read_csv(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _save_metric_figure(csv_path, out_path, task_filter=None, title="Model performance"):
    df = _safe_read_csv(csv_path)

    if df is None or df.empty:
        return

    if "task" not in df.columns or "model" not in df.columns or "accuracy" not in df.columns or "macro_f1" not in df.columns:
        return

    if task_filter:
        df = df[df["task"].astype(str).str.lower().str.contains(task_filter.lower(), na=False)].copy()

    if df.empty:
        return

    df = df.head(12).copy()

    labels = (
        df["model"]
        .astype(str)
        .str.replace("_v2", "", regex=False)
        .str.replace("_binary_risk", "", regex=False)
    )

    x = np.arange(len(df))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, df["macro_f1"], width, label="Macro-F1")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_label_distribution_figure(csv_path, out_path):
    df = _safe_read_csv(csv_path)

    if df is None or df.empty:
        return

    if "state_label" in df.columns and "count" in df.columns:
        labels = df["state_label"].astype(str)
        counts = df["count"]
    else:
        labels = df.iloc[:, 0].astype(str)
        counts = df.iloc[:, 1]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of samples")
    plt.title("Runtime state-label distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_ablation_figure(csv_path, out_path, task_filter, title):
    df = _safe_read_csv(csv_path)

    if df is None or df.empty:
        return

    if "task" in df.columns:
        df = df[df["task"].astype(str).str.lower().str.contains(task_filter.lower(), na=False)].copy()

    if df.empty or "feature_set" not in df.columns or "accuracy" not in df.columns or "macro_f1" not in df.columns:
        return

    x = np.arange(len(df))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, df["macro_f1"], width, label="Macro-F1")
    plt.xticks(x, df["feature_set"].astype(str), rotation=25, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_threshold_figure(csv_path, out_path):
    df = _safe_read_csv(csv_path)

    if df is None or df.empty:
        return

    if "accuracy" not in df.columns or "macro_f1" not in df.columns:
        return

    labels = (
        df["close_threshold_m"].astype(str) + "/" + df["very_close_threshold_m"].astype(str)
        if "close_threshold_m" in df.columns and "very_close_threshold_m" in df.columns
        else [str(i + 1) for i in range(len(df))]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(labels, df["accuracy"], marker="o", label="Accuracy")
    plt.plot(labels, df["macro_f1"], marker="s", label="Macro-F1")
    plt.xlabel("Close / very-close threshold (m)")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Threshold sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_latency_figure(csv_path, out_path):
    df = _safe_read_csv(csv_path)

    if df is None or df.empty:
        return

    if "model" not in df.columns or "mean_inference_time_ms_per_sample" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"].astype(str), df["mean_inference_time_ms_per_sample"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Latency (ms/sample)")
    plt.title("Runtime inference latency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()





def run_submission_level_experiments(combined, model_df, event_free_features):
    all_features = [
        c
        for c in model_df.columns
        if c not in ["run_id", "sec", "state_label", "risk_binary"]
    ]

    loro_df, loro_summary = run_leave_one_run_out(model_df, event_free_features)
    ablation_df = run_feature_ablation(model_df, event_free_features)
    threshold_df = run_threshold_sensitivity(combined)
    latency_df = run_runtime_latency(model_df, event_free_features)

    future_df = run_future_horizon_prediction(model_df, event_free_features)
    leakage_df = run_event_feature_leakage_audit(model_df, all_features, event_free_features)
    distance_reg_df = run_future_distance_regression(model_df, event_free_features)
    decision_policy_df = run_decision_policy_evaluation(model_df, event_free_features)

    print_header("SUBMISSION EXTRAS DONE")

    for name in [
        "leave_one_run_out_results_v4.csv",
        "leave_one_run_out_summary_v4.csv",
        "feature_group_ablation_v4.csv",
        "threshold_sensitivity_v4.csv",
        "runtime_latency_v4.csv",
        "future_horizon_binary_risk_results_v1.csv",
        "future_horizon_binary_risk_best_v1.csv",
        "event_feature_leakage_audit_v1.csv",
        "future_distance_regression_results_v1.csv",
        "decision_policy_threshold_results_v1.csv",
    ]:
        print(" -", RESULTS_DIR / name)

    print(" -", DATA_DIR / "future_horizon_model_table_v1.csv")
    print(" -", DATA_DIR / "future_distance_regression_table_v1.csv")
    #print(" -", EXTRA_DIR / "submission_latex_tables_v4.tex")


def organize_submission_outputs():
    _save_metric_figure(
        RESULTS_DIR / "model_results_v5_event_aware_and_event_free.csv",
        FIGURES_DIR / "fig_main_multiclass_results.png",
        "multiclass",
        "TMS-state recognition performance",
    )

    _save_label_distribution_figure(
        RESULTS_DIR / "label_distribution_combined_v4.csv",
        FIGURES_DIR / "fig_label_distribution.png",
    )

    _save_ablation_figure(
        RESULTS_DIR / "feature_group_ablation_v4.csv",
        FIGURES_DIR / "fig_ablation_binary_risk.png",
        "binary",
        "Binary feature-group ablation",
    )

    _save_ablation_figure(
        RESULTS_DIR / "feature_group_ablation_v4.csv",
        FIGURES_DIR / "fig_ablation_multiclass.png",
        "multiclass",
        "Multiclass feature-group ablation",
    )

    _save_threshold_figure(
        RESULTS_DIR / "threshold_sensitivity_v4.csv",
        FIGURES_DIR / "fig_threshold_sensitivity.png",
    )

    _save_latency_figure(
        RESULTS_DIR / "runtime_latency_v4.csv",
        FIGURES_DIR / "fig_runtime_latency.png",
    )

    future_path = RESULTS_DIR / "future_horizon_binary_risk_results_v1.csv"

    if future_path.exists():
        plot_future_horizon_results(pd.read_csv(future_path))

    distance_path = RESULTS_DIR / "future_distance_regression_results_v1.csv"

    if distance_path.exists():
        plot_future_distance_results(pd.read_csv(distance_path))

    print("\nSubmission package ready")
    print("Data folder       :", DATA_DIR)
    print("Results folder    :", RESULTS_DIR)
    print("Figures folder    :", FIGURES_DIR)
    print("Models folder     :", MODELS_DIR)
    print("Extra outputs     :", EXTRA_DIR)
   


def main():
    print_header("AGV DYNAMIC NEURAL TMS - JOURNAL PIPELINE V4")
    print("Base:", BASE_DIR)
    print("Out :", OUT_DIR)

    all_runs = []

    for run_name, folder in RUN_FOLDERS.items():
        if not folder.exists():
            print(f"[WARN] Missing folder {folder}")
            continue

        all_runs.append(build_run_dataset(run_name, folder))

    if not all_runs:
        raise RuntimeError("No datasets built. Check dataset1/dataset2/dataset3 folders.")

    combined = pd.concat(all_runs, ignore_index=True)
    combined.to_csv(DATA_DIR / "combined_dynamic_tms_dataset_1hz_v4.csv", index=False)

    save_summaries(combined)

    model_df, features = prepare_model_table(combined)
    model_df.to_csv(DATA_DIR / "combined_model_table_v4.csv", index=False)

    for ev_col in [
        "operator_event",
        "hard_stop_event",
        "hard_stop_operator",
        "hard_stop_scanner",
        "urgent_control_event",
        "virtual_scanner_event",
    ]:
        if ev_col in combined.columns:
            pd.crosstab(combined["state_label"], combined[ev_col]).to_csv(
                EXTRA_DIR / f"audit_label_vs_{ev_col}.csv"
            )

    print_header("FEATURES V4")
    print(features)
    print("Rows:", len(model_df))
    print("\nCombined labels:\n", combined["state_label"].value_counts())

    print_header("SETTING A: EVENT-AWARE DIAGNOSTIC RECOGNITION")

    multiclass = train_multiclass_models(model_df, features)
    binary = train_binary_risk_model(model_df, features)
    topology_gnn = train_topology_gnn_baselines(model_df, combined, features)

    result_frames = [multiclass, binary]

    if not topology_gnn.empty:
        result_frames.append(topology_gnn)

    results_event_aware = pd.concat(result_frames, ignore_index=True)
    results_event_aware.insert(0, "setting", "event_aware_diagnostic")
    results_event_aware.to_csv(RESULTS_DIR / "model_results_event_aware_v5.csv", index=False)

    event_free_features = [c for c in features if c not in EVENT_AWARE_FEATURES]

    print_header("SETTING B: EVENT-FREE EARLY RECOGNITION")
    print("Event-free features:", event_free_features)

    mc_free = train_multiclass_models(model_df, event_free_features)
    bin_free = train_binary_risk_model(model_df, event_free_features)
    gnn_free = train_topology_gnn_baselines(model_df, combined, event_free_features)

    free_frames = [mc_free, bin_free]

    if not gnn_free.empty:
        free_frames.append(gnn_free)

    results_event_free = pd.concat(free_frames, ignore_index=True)
    results_event_free.insert(0, "setting", "event_free_early")
    results_event_free.to_csv(RESULTS_DIR / "model_results_event_free_v5.csv", index=False)

    run_submission_level_experiments(combined, model_df, event_free_features)

    results = pd.concat([results_event_aware, results_event_free], ignore_index=True)
    results.to_csv(RESULTS_DIR / "model_results_v5_event_aware_and_event_free.csv", index=False)

    organize_submission_outputs()

    print_header("FINAL RESULTS V4")
    print(results)

    print_header("DONE")
    print("Outputs saved in:", OUT_DIR.resolve())
    print("Data:", DATA_DIR.resolve())
    print("Results:", RESULTS_DIR.resolve())
    print("Figures:", FIGURES_DIR.resolve())
    print("Models:", MODELS_DIR.resolve())
    print("Extra:", EXTRA_DIR.resolve())


if __name__ == "__main__":
    main()