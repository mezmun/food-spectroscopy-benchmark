# -*- coding: utf-8 -*-
"""
Spectroscopy Benchmark (Regression): Chemometrics vs Simple ANN / 1D-CNN

Nested evaluation + learning curve analysis for comparative benchmarking.

OUTPUT STRUCTURE
========================
For each dataset:
  outputs/<dataset_name>/
    tables/   -> each table is saved as a separate .xlsx
    figures/  -> .png figures

Tables:
  - bench_raw.xlsx
  - bench_summary.xlsx
  - best_config_freq.xlsx
  - run_log.xlsx
  - learning_curve_selected_config.xlsx
  - learning_curve_raw.xlsx
  - learning_curve_agg.xlsx
  - learning_curve_summary.xlsx
  - dl_complexity.xlsx

Figures:
  - bench_rmse_bar_<dataset>.png
  - bench_r2_bar_<dataset>.png
  - bench_rmse_box_<dataset>.png
  - bench_rmse_r2_scatter_<dataset>.png
  - learning_curve_train_cv_panels_<dataset>.png
  - dl_complexity_rmse_panels_<dataset>.png
"""

import os
import gc
import random
import ast
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# =========================================================
# 0) Basic helpers
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_clear_tf():
    tf.keras.backend.clear_session()


def build_early_stopping(
    patience: int = 30, monitor: str = "val_rmse"
) -> callbacks.Callback:
    return callbacks.EarlyStopping(
        monitor=monitor, mode="min", patience=patience, restore_best_weights=True
    )


def build_reduce_lr(monitor: str = "val_rmse") -> callbacks.Callback:
    return callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0,
    )


def _median(x: List[float]) -> float:
    return float(np.median(np.array(x, dtype=float)))


def _mean(x: List[float]) -> float:
    return float(np.mean(np.array(x, dtype=float)))


def _std(x: List[float]) -> float:
    x = np.array(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


# =========================================================
# 1) Dataset loading (Excel / CSV)
# =========================================================

@dataclass
class DatasetConfig:
    name: str
    kind: str  # "excel" or "csv"
    path: str
    satir_ilk: int
    satir_son: int
    bas_sutun: int
    bit_sutun: int
    col_Y: int
    sep: str = ","  # csv only


def load_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.kind.lower() == "excel":
        df = pd.read_excel(cfg.path, engine="openpyxl", header=None)
        data = df.values
    elif cfg.kind.lower() == "csv":
        df = pd.read_csv(cfg.path, sep=cfg.sep)
        data = df.values
    else:
        raise ValueError(f"Unknown kind: {cfg.kind}")

    y = data[cfg.satir_ilk:cfg.satir_son, cfg.col_Y].astype("float32").ravel()
    X = data[
        cfg.satir_ilk:cfg.satir_son, cfg.bas_sutun:cfg.bit_sutun
    ].astype("float32")
    return X, y


# =========================================================
# 2) Preprocess (leakage-safe)
# =========================================================

@dataclass
class PreprocessConfig:
    use_autoscale: bool = True  # StandardScaler


def preprocess_fit_apply(
    X_tr_raw: np.ndarray,
    X_other_raw_list: List[np.ndarray],
    cfg: PreprocessConfig,
):
    X_tr = X_tr_raw.copy()
    X_others = [Xo.copy() for Xo in X_other_raw_list]

    scaler = None
    if cfg.use_autoscale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X_tr)
        X_others = [scaler.transform(Xo) for Xo in X_others]

    return X_tr, X_others, scaler


def y_fit_apply(y_tr: np.ndarray, y_other_list: List[np.ndarray]):
    """
    Fold-safe y scaling for DL stability.
    Train y is scaled; predictions are inverse-transformed for reporting
    RMSE/R2 in original units.
    """
    ys = StandardScaler(with_mean=True, with_std=True)
    y_tr_s = ys.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_others_s = [ys.transform(y.reshape(-1, 1)).ravel() for y in y_other_list]
    return y_tr_s, y_others_s, ys


# =========================================================
# 3) Models
# =========================================================

def build_mlp(
    input_len: int, units: List[int], dropout: float = 0.3
) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len,), name="mlp_input")
    x = inp
    for u in units:
        x = layers.Dense(u, activation="relu")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inp, out, name=f"MLP_{'-'.join(map(str, units))}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_cnn1d(
    input_len: int,
    filters: List[int],
    kernel_size: int,
    dropout: float = 0.2,
    use_layernorm: bool = True,
) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, 1), name="cnn_input")
    x = inp
    for f in filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same")(x)
        if use_layernorm:
            x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(
        inp, out, name=f"CNN1D_{'-'.join(map(str, filters))}_k{kernel_size}"
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def fit_predict_plsr(X_tr, y_tr, X_te, n_comp: int):
    m = PLSRegression(n_components=n_comp)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel(), m


def fit_predict_svr(X_tr, y_tr, X_te, C: float, gamma="scale", epsilon=0.1):
    m = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel(), m


def fit_predict_ridge(X_tr, y_tr, X_te, alpha: float):
    m = Ridge(alpha=alpha)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel(), m


# =========================================================
# 3b) Complexity helpers (ANN / CNN1D)
# =========================================================

VERIFY_COMPLEXITY_WITH_KERAS = True
_DL_COMPLEXITY_CACHE: Dict[Tuple, Tuple[int, int]] = {}


def _safe_literal_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def ann_param_count_analytic(input_len: int, units: List[int]) -> int:
    prev = int(input_len)
    total = 0
    for u in units:
        u = int(u)
        total += (prev + 1) * u
        prev = u
    total += (prev + 1) * 1
    return int(total)


def cnn1d_param_count_analytic(
    input_len: int,
    filters: List[int],
    kernel_size: int,
    use_layernorm: bool = True,
) -> int:
    k = int(kernel_size)
    in_ch = 1
    total = 0
    for f in filters:
        f = int(f)
        total += (k * in_ch + 1) * f
        if use_layernorm:
            total += 2 * f
        in_ch = f
    total += (in_ch + 1) * 1
    return int(total)


def ann_flops_approx(input_len: int, units: List[int]) -> int:
    prev = int(input_len)
    flops = 0
    for u in units:
        u = int(u)
        flops += 2 * prev * u
        prev = u
    flops += 2 * prev * 1
    return int(flops)


def cnn1d_flops_approx(input_len: int, filters: List[int], kernel_size: int) -> int:
    L = int(input_len)
    k = int(kernel_size)
    in_ch = 1
    flops = 0
    for f in filters:
        f = int(f)
        flops += L * 2 * (k * in_ch * f)
        in_ch = f
    flops += 2 * in_ch * 1
    return int(flops)


def dl_complexity_cached(
    fam: str, cand: Dict[str, Any], input_len: int
) -> Tuple[int, int]:
    """
    Returns:
      param_count (verified with Keras if enabled),
      approx_flops (theoretical, per forward pass).
    """
    cand_str = str(cand)

    if fam == "ANN":
        key = ("ANN", int(input_len), cand_str, float(MODEL_REGISTRY["ANN"]["dropout"]))
        if key in _DL_COMPLEXITY_CACHE:
            return _DL_COMPLEXITY_CACHE[key]

        params_a = ann_param_count_analytic(input_len, cand["units"])
        flops_a = ann_flops_approx(input_len, cand["units"])
        params_final = params_a

        if VERIFY_COMPLEXITY_WITH_KERAS:
            try:
                safe_clear_tf()
                m = build_mlp(
                    input_len=input_len,
                    units=cand["units"],
                    dropout=MODEL_REGISTRY["ANN"]["dropout"],
                )
                params_k = int(m.count_params())
                if params_k != params_a:
                    print(
                        f"[WARN] ANN param mismatch analytic={params_a} "
                        f"vs keras={params_k} | cand={cand}"
                    )
                    params_final = params_k
            except Exception as e:
                print(f"[WARN] ANN keras param verify failed: {e} | cand={cand}")

        _DL_COMPLEXITY_CACHE[key] = (int(params_final), int(flops_a))
        return _DL_COMPLEXITY_CACHE[key]

    if fam == "CNN1D":
        key = (
            "CNN1D",
            int(input_len),
            cand_str,
            float(MODEL_REGISTRY["CNN1D"]["dropout"]),
            bool(MODEL_REGISTRY["CNN1D"]["use_layernorm"]),
        )
        if key in _DL_COMPLEXITY_CACHE:
            return _DL_COMPLEXITY_CACHE[key]

        params_a = cnn1d_param_count_analytic(
            input_len=input_len,
            filters=cand["filters"],
            kernel_size=cand["kernel"],
            use_layernorm=MODEL_REGISTRY["CNN1D"]["use_layernorm"],
        )
        flops_a = cnn1d_flops_approx(input_len, cand["filters"], cand["kernel"])
        params_final = params_a

        if VERIFY_COMPLEXITY_WITH_KERAS:
            try:
                safe_clear_tf()
                m = build_cnn1d(
                    input_len=input_len,
                    filters=cand["filters"],
                    kernel_size=cand["kernel"],
                    dropout=MODEL_REGISTRY["CNN1D"]["dropout"],
                    use_layernorm=MODEL_REGISTRY["CNN1D"]["use_layernorm"],
                )
                params_k = int(m.count_params())
                if params_k != params_a:
                    print(
                        f"[WARN] CNN1D param mismatch analytic={params_a} "
                        f"vs keras={params_k} | cand={cand}"
                    )
                    params_final = params_k
            except Exception as e:
                print(f"[WARN] CNN1D keras param verify failed: {e} | cand={cand}")

        _DL_COMPLEXITY_CACHE[key] = (int(params_final), int(flops_a))
        return _DL_COMPLEXITY_CACHE[key]

    return (None, None)


def save_multi_table_xlsx(path: str, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def build_dl_complexity_tables(
    dataset_name: str,
    bench_df: pd.DataFrame,
    candlog_df: pd.DataFrame,
    input_len: int,
) -> Dict[str, pd.DataFrame]:
    inner_raw = candlog_df.copy()

    g = inner_raw.groupby(["dataset", "family", "cand_str"], as_index=False)
    inner_agg = g.agg(
        param_count=("param_count", "first"),
        approx_flops=("approx_flops", "first"),
        inner_cv_rmse_mean=("inner_cv_rmse", "mean"),
        inner_cv_rmse_std=("inner_cv_rmse", "std"),
        n_outer=("outer_fold", "nunique"),
    )
    inner_agg["inner_cv_rmse_std"] = inner_agg["inner_cv_rmse_std"].fillna(0.0)

    dl_best = bench_df[bench_df["family"].isin(["ANN", "CNN1D"])].copy()
    params_list, flops_list = [], []
    for s in dl_best["best_cand"].astype(str).tolist():
        cand = _safe_literal_eval(s)
        if isinstance(cand, dict):
            fam = "ANN" if "units" in cand else "CNN1D"
            pcount, aflops = dl_complexity_cached(fam, cand, input_len=input_len)
            params_list.append(pcount)
            flops_list.append(aflops)
        else:
            params_list.append(None)
            flops_list.append(None)
    dl_best["param_count"] = params_list
    dl_best["approx_flops"] = flops_list

    return {
        "inner_candidates_raw": inner_raw,
        "inner_candidates_agg": inner_agg.sort_values(
            ["family", "param_count"], ascending=[True, True]
        ),
        "outer_best": dl_best,
    }


def fig_dl_complexity_rmse_panels(
    inner_agg_df: pd.DataFrame, out_dir_figs: str, dataset_name: str
):
    sf = inner_agg_df[inner_agg_df["dataset"] == dataset_name].copy()
    ann = sf[sf["family"] == "ANN"].dropna(subset=["param_count"])
    cnn = sf[sf["family"] == "CNN1D"].dropna(subset=["param_count"])

    if ann.empty and cnn.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1, ax2 = axes

    if not ann.empty:
        ann = ann.sort_values("param_count")
        ax1.errorbar(
            ann["param_count"],
            ann["inner_cv_rmse_mean"],
            yerr=ann["inner_cv_rmse_std"],
            fmt="o",
        )
        ax1.set_xscale("log")
        ax1.set_title(f"{dataset_name} | ANN | Complexity vs RMSE (Inner-CV)")
        ax1.set_xlabel("Param count (log)")
        ax1.set_ylabel("Inner-CV RMSE (mean±std across outer folds)")
        ax1.grid(True, alpha=0.25)
    else:
        ax1.set_axis_off()

    if not cnn.empty:
        cnn = cnn.sort_values("param_count")
        ax2.errorbar(
            cnn["param_count"],
            cnn["inner_cv_rmse_mean"],
            yerr=cnn["inner_cv_rmse_std"],
            fmt="o",
        )
        ax2.set_xscale("log")
        ax2.set_title(f"{dataset_name} | CNN1D | Complexity vs RMSE (Inner-CV)")
        ax2.set_xlabel("Param count (log)")
        ax2.set_ylabel("Inner-CV RMSE (mean±std across outer folds)")
        ax2.grid(True, alpha=0.25)
    else:
        ax2.set_axis_off()

    plt.tight_layout()
    path = os.path.join(out_dir_figs, f"dl_complexity_rmse_panels_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================
# 4) Model Registry
# =========================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "PLSR": {
        "enabled": True,
        "candidates": [{"n_comp": n} for n in [5, 10, 15, 20, 25, 30]],
    },
    "SVR": {
        "enabled": True,
        "candidates": [
            {"C": C, "gamma": "scale", "epsilon": 0.1}
            for C in [0.3, 1, 3, 10, 30, 100]
        ],
    },
    "Ridge": {
        "enabled": True,
        "candidates": [{"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]],
    },
    "ANN": {
        "enabled": True,
        "candidates": [
            {"units": [8]},
            {"units": [12]},
            {"units": [16]},
            {"units": [24]},
            {"units": [32]},
            {"units": [48]},
            {"units": [64]},
            {"units": [16, 8]},
            {"units": [32, 16]},
            {"units": [64, 32]},
            {"units": [96, 48]},
            {"units": [128, 64]},
        ],
        "dropout": 0.3,
    },
    "CNN1D": {
        "enabled": True,
        "candidates": [
            {"filters": [8], "kernel": 3},
            {"filters": [16], "kernel": 3},
            {"filters": [32], "kernel": 3},
            {"filters": [8], "kernel": 5},
            {"filters": [16], "kernel": 5},
            {"filters": [32], "kernel": 5},
            {"filters": [16], "kernel": 7},
            {"filters": [32], "kernel": 7},
            {"filters": [16, 8], "kernel": 3},
            {"filters": [32, 16], "kernel": 3},
            {"filters": [16, 8], "kernel": 5},
            {"filters": [32, 16], "kernel": 5},
        ],
        "dropout": 0.2,
        "use_layernorm": True,
    },
}


def is_dl_family(fam: str) -> bool:
    return fam in ["ANN", "CNN1D"]


# =========================================================
# 5) Experiment Config
# =========================================================

@dataclass
class ExperimentConfig:
    outer_folds: int = 3
    inner_folds: int = 3

    dl_seeds_inner: List[int] = None
    dl_seeds_final: List[int] = None

    epochs: int = 400
    batch_size: int = 16
    patience: int = 30

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    run_learning_curve: bool = True
    learning_curve_points: int = 10
    lc_families: List[str] = None
    lc_seed: int = 0
    lc_select_seed: int = 0

    lc_cv_folds: int = 3
    lc_repeats: int = 2
    lc_repeat_seeds: List[int] = None

    dl_validation_split: float = 0.15


def default_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.dl_seeds_inner = [0, 1]
    cfg.dl_seeds_final = [0, 1]
    cfg.lc_families = ["PLSR", "ANN", "CNN1D"]
    cfg.lc_repeat_seeds = [0, 1]
    return cfg


# =========================================================
# 6) Output dirs per dataset + saving tables
# =========================================================

def ensure_dataset_dirs(dataset_name: str) -> Dict[str, str]:
    if dataset_name is None or dataset_name.strip() == "":
        raise ValueError(
            "dataset_name is empty/blank. Refusing to create outputs/ "
            "without dataset folder."
        )
    base = os.path.join("outputs", dataset_name)
    tables = os.path.join(base, "tables")
    figs = os.path.join(base, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    return {"base": base, "tables": tables, "figs": figs}


def save_table_xlsx(path: str, df: pd.DataFrame, sheet_name: str = "data"):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# =========================================================
# 7) Progress accounting (total fits + remaining) + log to Excel
# =========================================================

class FitProgress:
    def __init__(self, total_fits: int):
        self.total = int(total_fits)
        self.done = 0
        self.records: List[Dict[str, Any]] = []

    def step(self, msg: str):
        self.done += 1
        left = self.total - self.done
        line = f"[FIT {self.done}/{self.total} | left={left}] {msg}"
        print(line)
        self.records.append(
            {
                "fit_done": self.done,
                "fit_total": self.total,
                "left": left,
                "message": msg,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def estimate_total_fits_for_dataset(
    cfg: ExperimentConfig, families: List[str], do_lc: bool
) -> int:
    total = 0

    for _ in range(cfg.outer_folds):
        for fam in families:
            cands = len(MODEL_REGISTRY[fam]["candidates"])
            if is_dl_family(fam):
                total += cands * cfg.inner_folds * len(cfg.dl_seeds_inner)
                total += len(cfg.dl_seeds_final)
            else:
                total += cands * cfg.inner_folds
                total += 1

    if do_lc:
        points = cfg.learning_curve_points
        lc_fams = [f for f in cfg.lc_families if f in families]
        for fam in lc_fams:
            cands = len(MODEL_REGISTRY[fam]["candidates"])
            if is_dl_family(fam):
                total += cands * cfg.inner_folds * 1
            else:
                total += cands * cfg.inner_folds
            total += points * cfg.lc_repeats * cfg.lc_cv_folds

    return int(total)


# =========================================================
# 8) Inner CV selection (3-fold)
# =========================================================

def kfold_splits(n: int, n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def one_line_cand(fam: str, cand: Dict[str, Any]) -> str:
    return f"{fam} cand={cand}"


def eval_candidate_sklearn_cv(
    fam: str,
    cand: Dict[str, Any],
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
    cv_splits,
    dataset_name: str,
    outer_fold_i: int,
    progress: FitProgress,
    stage: str,
) -> float:
    scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
        X_in_tr_raw, y_in_tr = X_tr_raw[tr_idx], y_tr[tr_idx]
        X_in_va_raw, y_in_va = X_tr_raw[va_idx], y_tr[va_idx]

        X_in_tr, [X_in_va], _ = preprocess_fit_apply(
            X_in_tr_raw, [X_in_va_raw], cfg.preprocess
        )

        if fam == "PLSR":
            yhat, _ = fit_predict_plsr(X_in_tr, y_in_tr, X_in_va, cand["n_comp"])
        elif fam == "SVR":
            yhat, _ = fit_predict_svr(
                X_in_tr, y_in_tr, X_in_va, cand["C"], cand["gamma"], cand["epsilon"]
            )
        elif fam == "Ridge":
            yhat, _ = fit_predict_ridge(X_in_tr, y_in_tr, X_in_va, cand["alpha"])
        else:
            raise ValueError(f"Unknown sklearn family: {fam}")

        scores.append(rmse(y_in_va, yhat))

        progress.step(
            f"DS={dataset_name} | outer={outer_fold_i}/{cfg.outer_folds} | {stage}"
            f" | inner_fold={fold_i}/{cfg.inner_folds} | {one_line_cand(fam, cand)}"
        )

    return float(np.mean(scores))


def eval_candidate_dl_cv(
    fam: str,
    cand: Dict[str, Any],
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
    cv_splits,
    input_len: int,
    seeds: List[int],
    dataset_name: str,
    outer_fold_i: int,
    progress: FitProgress,
    stage: str,
) -> float:
    seed_scores = []
    for s in seeds:
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
            set_seed(s)
            safe_clear_tf()

            X_in_tr_raw, y_in_tr = X_tr_raw[tr_idx], y_tr[tr_idx]
            X_in_va_raw, y_in_va = X_tr_raw[va_idx], y_tr[va_idx]

            X_in_tr, [X_in_va], _ = preprocess_fit_apply(
                X_in_tr_raw, [X_in_va_raw], cfg.preprocess
            )

            y_in_tr_s, [y_in_va_s], ysc = y_fit_apply(y_in_tr, [y_in_va])

            es = build_early_stopping(patience=cfg.patience, monitor="val_rmse")
            rlrop = build_reduce_lr(monitor="val_rmse")

            if fam == "ANN":
                model = build_mlp(
                    input_len=input_len,
                    units=cand["units"],
                    dropout=MODEL_REGISTRY["ANN"]["dropout"],
                )
                Xtr_in, Xva_in = X_in_tr, X_in_va
            elif fam == "CNN1D":
                model = build_cnn1d(
                    input_len=input_len,
                    filters=cand["filters"],
                    kernel_size=cand["kernel"],
                    dropout=MODEL_REGISTRY["CNN1D"]["dropout"],
                    use_layernorm=MODEL_REGISTRY["CNN1D"]["use_layernorm"],
                )
                Xtr_in, Xva_in = X_in_tr[..., None], X_in_va[..., None]
            else:
                raise ValueError(f"Unknown DL family: {fam}")

            model.fit(
                Xtr_in,
                y_in_tr_s,
                validation_data=(Xva_in, y_in_va_s),
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                callbacks=[es, rlrop],
                verbose=0,
            )

            yhat_s = model.predict(Xva_in, verbose=0).ravel()
            yhat = ysc.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
            fold_scores.append(rmse(y_in_va, yhat))

            progress.step(
                f"DS={dataset_name} | outer={outer_fold_i}/{cfg.outer_folds} | {stage}"
                f" | seed={s} | inner_fold={fold_i}/{cfg.inner_folds} | "
                f"{one_line_cand(fam, cand)}"
            )

        seed_scores.append(float(np.mean(fold_scores)))

    return _median(seed_scores)


def select_best_config_inner(
    fam: str,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
    input_len: int,
    inner_seed: int,
    dataset_name: str,
    outer_fold_i: int,
    progress: FitProgress,
    cand_log_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], float]:
    cv_splits = kfold_splits(len(X_tr_raw), cfg.inner_folds, seed=inner_seed)
    candidates = MODEL_REGISTRY[fam]["candidates"]

    best_cand, best_score = None, None
    for cand in candidates:
        if is_dl_family(fam):
            score = eval_candidate_dl_cv(
                fam=fam,
                cand=cand,
                X_tr_raw=X_tr_raw,
                y_tr=y_tr,
                cfg=cfg,
                cv_splits=cv_splits,
                input_len=input_len,
                seeds=cfg.dl_seeds_inner,
                dataset_name=dataset_name,
                outer_fold_i=outer_fold_i,
                progress=progress,
                stage="INNER_SELECT",
            )
        else:
            score = eval_candidate_sklearn_cv(
                fam=fam,
                cand=cand,
                X_tr_raw=X_tr_raw,
                y_tr=y_tr,
                cfg=cfg,
                cv_splits=cv_splits,
                dataset_name=dataset_name,
                outer_fold_i=outer_fold_i,
                progress=progress,
                stage="INNER_SELECT",
            )

        if cand_log_rows is not None:
            pcount, aflops = (None, None)
            if is_dl_family(fam):
                pcount, aflops = dl_complexity_cached(fam, cand, input_len=input_len)

            cand_log_rows.append(
                {
                    "dataset": dataset_name,
                    "outer_fold": int(outer_fold_i),
                    "family": fam,
                    "cand_str": str(cand),
                    "param_count": pcount,
                    "approx_flops": aflops,
                    "inner_cv_rmse": float(score),
                    "inner_seed": int(inner_seed),
                }
            )

        if best_score is None or score < best_score:
            best_score = score
            best_cand = cand

    return best_cand, float(best_score)


# =========================================================
# 9) Final fit on outer train, eval on outer test
# =========================================================

def fit_and_eval_on_outer_test(
    fam: str,
    best_cand: Dict[str, Any],
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_te_raw: np.ndarray,
    y_te: np.ndarray,
    cfg: ExperimentConfig,
    input_len: int,
    dataset_name: str,
    outer_fold_i: int,
    progress: FitProgress,
) -> Dict[str, Any]:
    X_tr, [X_te], _ = preprocess_fit_apply(X_tr_raw, [X_te_raw], cfg.preprocess)

    if is_dl_family(fam):
        test_rmses, test_r2s = [], []
        for s in cfg.dl_seeds_final:
            set_seed(1000 + s + outer_fold_i)
            safe_clear_tf()

            y_tr_s, [y_te_s], ysc = y_fit_apply(y_tr, [y_te])

            es = build_early_stopping(patience=cfg.patience, monitor="val_rmse")
            rlrop = build_reduce_lr(monitor="val_rmse")

            if fam == "ANN":
                model = build_mlp(
                    input_len=input_len,
                    units=best_cand["units"],
                    dropout=MODEL_REGISTRY["ANN"]["dropout"],
                )
                Xtr_in, Xte_in = X_tr, X_te
            else:
                model = build_cnn1d(
                    input_len=input_len,
                    filters=best_cand["filters"],
                    kernel_size=best_cand["kernel"],
                    dropout=MODEL_REGISTRY["CNN1D"]["dropout"],
                    use_layernorm=MODEL_REGISTRY["CNN1D"]["use_layernorm"],
                )
                Xtr_in, Xte_in = X_tr[..., None], X_te[..., None]

            model.fit(
                Xtr_in,
                y_tr_s,
                validation_split=cfg.dl_validation_split,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                callbacks=[es, rlrop],
                verbose=0,
            )

            yhat_s = model.predict(Xte_in, verbose=0).ravel()
            yhat = ysc.inverse_transform(yhat_s.reshape(-1, 1)).ravel()

            test_rmses.append(rmse(y_te, yhat))
            test_r2s.append(float(r2_score(y_te, yhat)))

            progress.step(
                f"DS={dataset_name} | outer={outer_fold_i}/{cfg.outer_folds} | "
                f"FINAL_EVAL | seed={s} | {one_line_cand(fam, best_cand)}"
            )

        return {
            "family": fam,
            "best_cand": str(best_cand),
            "test_rmse_median": _median(test_rmses),
            "test_rmse_mean": _mean(test_rmses),
            "test_rmse_std": _std(test_rmses),
            "test_r2_median": _median(test_r2s),
            "test_r2_mean": _mean(test_r2s),
            "test_r2_std": _std(test_r2s),
        }

    else:
        if fam == "PLSR":
            yhat, _ = fit_predict_plsr(X_tr, y_tr, X_te, best_cand["n_comp"])
        elif fam == "SVR":
            yhat, _ = fit_predict_svr(
                X_tr,
                y_tr,
                X_te,
                best_cand["C"],
                best_cand["gamma"],
                best_cand["epsilon"],
            )
        elif fam == "Ridge":
            yhat, _ = fit_predict_ridge(X_tr, y_tr, X_te, best_cand["alpha"])
        else:
            raise ValueError(f"Unknown family: {fam}")

        progress.step(
            f"DS={dataset_name} | outer={outer_fold_i}/{cfg.outer_folds} | "
            f"FINAL_EVAL | {one_line_cand(fam, best_cand)}"
        )

        r = rmse(y_te, yhat)
        r2 = float(r2_score(y_te, yhat))
        return {
            "family": fam,
            "best_cand": str(best_cand),
            "test_rmse_median": r,
            "test_rmse_mean": r,
            "test_rmse_std": 0.0,
            "test_r2_median": r2,
            "test_r2_mean": r2,
            "test_r2_std": 0.0,
        }


# =========================================================
# 10) Benchmark runner (Outer 3-fold)
# =========================================================

def run_outer_cv_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    cfg: ExperimentConfig,
    progress: FitProgress,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_len = X.shape[1]

    outer = KFold(n_splits=cfg.outer_folds, shuffle=True, random_state=42)
    families = [k for k, v in MODEL_REGISTRY.items() if v.get("enabled", False)]

    rows = []
    cand_log_rows: List[Dict[str, Any]] = []
    outer_i = 0

    for tr_idx, te_idx in outer.split(np.arange(len(X))):
        outer_i += 1
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_te_raw, y_te = X[te_idx], y[te_idx]

        best_map: Dict[str, Tuple[Dict[str, Any], float]] = {}

        for fam in families:
            inner_seed = 100 + outer_i
            best_cand, best_val = select_best_config_inner(
                fam=fam,
                X_tr_raw=X_tr_raw,
                y_tr=y_tr,
                cfg=cfg,
                input_len=input_len,
                inner_seed=inner_seed,
                dataset_name=dataset_name,
                outer_fold_i=outer_i,
                progress=progress,
                cand_log_rows=cand_log_rows,
            )
            best_map[fam] = (best_cand, best_val)

        for fam in families:
            best_cand, best_val = best_map[fam]
            out = fit_and_eval_on_outer_test(
                fam=fam,
                best_cand=best_cand,
                X_tr_raw=X_tr_raw,
                y_tr=y_tr,
                X_te_raw=X_te_raw,
                y_te=y_te,
                cfg=cfg,
                input_len=input_len,
                dataset_name=dataset_name,
                outer_fold_i=outer_i,
                progress=progress,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "outer_fold": outer_i,
                    "family": fam,
                    "inner_cv_rmse": float(best_val),
                    "test_rmse_median": out["test_rmse_median"],
                    "test_rmse_mean": out["test_rmse_mean"],
                    "test_rmse_std": out["test_rmse_std"],
                    "test_r2_median": out["test_r2_median"],
                    "test_r2_mean": out["test_r2_mean"],
                    "test_r2_std": out["test_r2_std"],
                    "best_cand": out["best_cand"],
                }
            )

        del X_tr_raw, y_tr, X_te_raw, y_te, best_map
        gc.collect()
        safe_clear_tf()

    bench_df = pd.DataFrame(rows)
    candlog_df = pd.DataFrame(cand_log_rows)
    return bench_df, candlog_df


def make_benchmark_summary(bench_df: pd.DataFrame) -> pd.DataFrame:
    g = bench_df.groupby(["dataset", "family"])
    out = g.agg(
        rmse_mean=("test_rmse_median", "mean"),
        rmse_std=("test_rmse_median", "std"),
        r2_mean=("test_r2_median", "mean"),
        r2_std=("test_r2_median", "std"),
        n_folds=("test_rmse_median", "count"),
    ).reset_index()
    return out


def best_config_frequencies(bench_df: pd.DataFrame) -> pd.DataFrame:
    freq = (
        bench_df.groupby(["dataset", "family", "best_cand"])
        .size()
        .reset_index(name="count")
    )
    freq = freq.sort_values(
        ["dataset", "family", "count"], ascending=[True, True, False]
    )
    return freq


# =========================================================
# 11) Learning Curve (Train error + CV error)
# =========================================================

def select_best_config_for_lc_once(
    fam: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    input_len: int,
    dataset_name: str,
    progress: FitProgress,
) -> Tuple[Dict[str, Any], float]:
    cv_splits = kfold_splits(len(X_raw), cfg.inner_folds, seed=777)
    candidates = MODEL_REGISTRY[fam]["candidates"]

    best_cand, best_score = None, None
    for cand in candidates:
        if is_dl_family(fam):
            score = eval_candidate_dl_cv(
                fam=fam,
                cand=cand,
                X_tr_raw=X_raw,
                y_tr=y,
                cfg=cfg,
                cv_splits=cv_splits,
                input_len=input_len,
                seeds=[cfg.lc_select_seed],
                dataset_name=dataset_name,
                outer_fold_i=0,
                progress=progress,
                stage="LC_SELECT_ONCE",
            )
        else:
            score = eval_candidate_sklearn_cv(
                fam=fam,
                cand=cand,
                X_tr_raw=X_raw,
                y_tr=y,
                cfg=cfg,
                cv_splits=cv_splits,
                dataset_name=dataset_name,
                outer_fold_i=0,
                progress=progress,
                stage="LC_SELECT_ONCE",
            )

        if best_score is None or score < best_score:
            best_score = score
            best_cand = cand

    return best_cand, float(best_score)


def _fit_predict_one_fold_for_lc(
    fam: str,
    cand: Dict[str, Any],
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_va_raw: np.ndarray,
    y_va: np.ndarray,
    cfg: ExperimentConfig,
    input_len: int,
    seed: int,
    dataset_name: str,
    point_i: int,
    repeat_i: int,
    fold_i: int,
    progress: FitProgress,
) -> Dict[str, float]:
    """
    Returns train_rmse, val_rmse, train_r2, val_r2 in original y units.
    """
    X_tr, [X_va], _ = preprocess_fit_apply(X_tr_raw, [X_va_raw], cfg.preprocess)

    if fam == "PLSR":
        yhat_tr, _ = fit_predict_plsr(X_tr, y_tr, X_tr, cand["n_comp"])
        yhat_va, _ = fit_predict_plsr(X_tr, y_tr, X_va, cand["n_comp"])
        progress.step(
            f"DS={dataset_name} | LC_FIT point={point_i} rep={repeat_i} "
            f"fold={fold_i} | {one_line_cand(fam, cand)}"
        )
        return {
            "train_rmse": rmse(y_tr, yhat_tr),
            "val_rmse": rmse(y_va, yhat_va),
            "train_r2": float(r2_score(y_tr, yhat_tr)),
            "val_r2": float(r2_score(y_va, yhat_va)),
        }

    if fam == "ANN":
        set_seed(seed)
        safe_clear_tf()
        y_tr_s, [y_va_s], ysc = y_fit_apply(y_tr, [y_va])

        es = build_early_stopping(patience=cfg.patience, monitor="val_rmse")
        rlrop = build_reduce_lr(monitor="val_rmse")

        model = build_mlp(
            input_len=input_len,
            units=cand["units"],
            dropout=MODEL_REGISTRY["ANN"]["dropout"],
        )
        model.fit(
            X_tr,
            y_tr_s,
            validation_data=(X_va, y_va_s),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[es, rlrop],
            verbose=0,
        )
        yhat_tr_s = model.predict(X_tr, verbose=0).ravel()
        yhat_va_s = model.predict(X_va, verbose=0).ravel()
        yhat_tr = ysc.inverse_transform(yhat_tr_s.reshape(-1, 1)).ravel()
        yhat_va = ysc.inverse_transform(yhat_va_s.reshape(-1, 1)).ravel()

        progress.step(
            f"DS={dataset_name} | LC_FIT point={point_i} rep={repeat_i} "
            f"fold={fold_i} | seed={seed} | {one_line_cand(fam, cand)}"
        )
        return {
            "train_rmse": rmse(y_tr, yhat_tr),
            "val_rmse": rmse(y_va, yhat_va),
            "train_r2": float(r2_score(y_tr, yhat_tr)),
            "val_r2": float(r2_score(y_va, yhat_va)),
        }

    if fam == "CNN1D":
        set_seed(seed)
        safe_clear_tf()
        y_tr_s, [y_va_s], ysc = y_fit_apply(y_tr, [y_va])

        es = build_early_stopping(patience=cfg.patience, monitor="val_rmse")
        rlrop = build_reduce_lr(monitor="val_rmse")

        model = build_cnn1d(
            input_len=input_len,
            filters=cand["filters"],
            kernel_size=cand["kernel"],
            dropout=MODEL_REGISTRY["CNN1D"]["dropout"],
            use_layernorm=MODEL_REGISTRY["CNN1D"]["use_layernorm"],
        )
        model.fit(
            X_tr[..., None],
            y_tr_s,
            validation_data=(X_va[..., None], y_va_s),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[es, rlrop],
            verbose=0,
        )
        yhat_tr_s = model.predict(X_tr[..., None], verbose=0).ravel()
        yhat_va_s = model.predict(X_va[..., None], verbose=0).ravel()
        yhat_tr = ysc.inverse_transform(yhat_tr_s.reshape(-1, 1)).ravel()
        yhat_va = ysc.inverse_transform(yhat_va_s.reshape(-1, 1)).ravel()

        progress.step(
            f"DS={dataset_name} | LC_FIT point={point_i} rep={repeat_i} "
            f"fold={fold_i} | seed={seed} | {one_line_cand(fam, cand)}"
        )
        return {
            "train_rmse": rmse(y_tr, yhat_tr),
            "val_rmse": rmse(y_va, yhat_va),
            "train_r2": float(r2_score(y_tr, yhat_tr)),
            "val_r2": float(r2_score(y_va, yhat_va)),
        }

    raise ValueError(f"Unknown LC family: {fam}")


def run_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    cfg: ExperimentConfig,
    progress: FitProgress,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_len = X.shape[1]
    families = [f for f in cfg.lc_families if MODEL_REGISTRY.get(f, {}).get("enabled")]

    fracs = np.linspace(0.1, 1.0, cfg.learning_curve_points).tolist()

    select_rows = []
    lc_rows = []

    best_map = {}
    for fam in families:
        best_cand, best_cv_rmse = select_best_config_for_lc_once(
            fam=fam,
            X_raw=X,
            y=y,
            cfg=cfg,
            input_len=input_len,
            dataset_name=dataset_name,
            progress=progress,
        )
        best_map[fam] = best_cand
        select_rows.append(
            {
                "dataset": dataset_name,
                "family": fam,
                "selected_best_cand": str(best_cand),
                "inner_cv_rmse": float(best_cv_rmse),
                "note": "Selected once on full data; fixed across LC points.",
            }
        )

    rng = np.random.RandomState(cfg.lc_seed)

    for point_i, frac in enumerate(fracs, start=1):
        m = max(20, int(round(len(X) * frac)))
        sub_idx = rng.choice(len(X), size=m, replace=False)
        X_sub_raw, y_sub = X[sub_idx], y[sub_idx]

        kf = KFold(n_splits=cfg.lc_cv_folds, shuffle=True, random_state=999 + point_i)
        splits = list(kf.split(np.arange(m)))

        for repeat_i in range(1, cfg.lc_repeats + 1):
            seed = (
                cfg.lc_repeat_seeds[repeat_i - 1]
                if cfg.lc_repeat_seeds
                else (cfg.lc_seed + repeat_i)
            )

            for fam in families:
                cand = best_map[fam]
                fold_metrics = []
                for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
                    X_tr_raw, y_tr = X_sub_raw[tr_idx], y_sub[tr_idx]
                    X_va_raw, y_va = X_sub_raw[va_idx], y_sub[va_idx]

                    met = _fit_predict_one_fold_for_lc(
                        fam=fam,
                        cand=cand,
                        X_tr_raw=X_tr_raw,
                        y_tr=y_tr,
                        X_va_raw=X_va_raw,
                        y_va=y_va,
                        cfg=cfg,
                        input_len=input_len,
                        seed=seed,
                        dataset_name=dataset_name,
                        point_i=point_i,
                        repeat_i=repeat_i,
                        fold_i=fold_i,
                        progress=progress,
                    )
                    fold_metrics.append(met)

                train_rmse = float(np.mean([m_["train_rmse"] for m_ in fold_metrics]))
                val_rmse = float(np.mean([m_["val_rmse"] for m_ in fold_metrics]))
                train_r2 = float(np.mean([m_["train_r2"] for m_ in fold_metrics]))
                val_r2 = float(np.mean([m_["val_r2"] for m_ in fold_metrics]))

                lc_rows.append(
                    {
                        "dataset": dataset_name,
                        "family": fam,
                        "point": point_i,
                        "train_fraction": float(frac),
                        "train_n": int(m),
                        "repeat": int(repeat_i),
                        "seed": int(seed),
                        "cv_folds": int(cfg.lc_cv_folds),
                        "train_rmse": train_rmse,
                        "cv_rmse": val_rmse,
                        "train_r2": train_r2,
                        "cv_r2": val_r2,
                        "fixed_best_cand": str(cand),
                    }
                )

        del X_sub_raw, y_sub
        gc.collect()
        safe_clear_tf()

    sel_df = pd.DataFrame(select_rows)
    lc_raw_df = pd.DataFrame(lc_rows)

    g = lc_raw_df.groupby(
        ["dataset", "family", "point", "train_fraction", "train_n"], as_index=False
    )
    lc_agg = g.agg(
        train_rmse_mean=("train_rmse", "mean"),
        train_rmse_std=("train_rmse", "std"),
        cv_rmse_mean=("cv_rmse", "mean"),
        cv_rmse_std=("cv_rmse", "std"),
        train_r2_mean=("train_r2", "mean"),
        train_r2_std=("train_r2", "std"),
        cv_r2_mean=("cv_r2", "mean"),
        cv_r2_std=("cv_r2", "std"),
        n_repeats=("repeat", "nunique"),
    )

    for c in ["train_rmse_std", "cv_rmse_std", "train_r2_std", "cv_r2_std"]:
        lc_agg[c] = lc_agg[c].fillna(0.0)

    return sel_df, lc_raw_df, lc_agg


def summarize_learning_curve(lc_agg_df: pd.DataFrame) -> pd.DataFrame:
    g = lc_agg_df.groupby(["dataset", "family"])
    out = g.agg(
        cv_rmse_best=("cv_rmse_mean", "min"),
        cv_rmse_last=("cv_rmse_mean", lambda s: float(s.iloc[-1])),
        cv_r2_last=("cv_r2_mean", lambda s: float(s.iloc[-1])),
        n_points=("cv_rmse_mean", "count"),
    ).reset_index()
    return out


# =========================================================
# 12) Figures (saved per dataset)
# =========================================================

def fig_bench_rmse_bar(
    bench_summary_df: pd.DataFrame, out_dir_figs: str, dataset_name: str
):
    sub = bench_summary_df.sort_values("rmse_mean")
    plt.figure()
    plt.bar(sub["family"], sub["rmse_mean"], yerr=sub["rmse_std"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE (mean ± std across outer folds)")
    plt.title(f"{dataset_name} | RMSE")
    plt.tight_layout()
    path = os.path.join(out_dir_figs, f"bench_rmse_bar_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def fig_bench_r2_bar(
    bench_summary_df: pd.DataFrame, out_dir_figs: str, dataset_name: str
):
    sub = bench_summary_df.sort_values("r2_mean", ascending=False)
    plt.figure()
    plt.bar(sub["family"], sub["r2_mean"], yerr=sub["r2_std"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R² (mean ± std across outer folds)")
    plt.title(f"{dataset_name} | R²")
    plt.tight_layout()
    path = os.path.join(out_dir_figs, f"bench_r2_bar_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def fig_bench_rmse_box(bench_df: pd.DataFrame, out_dir_figs: str, dataset_name: str):
    families = list(bench_df["family"].unique())
    data = [bench_df[bench_df["family"] == f]["test_rmse_median"].values for f in families]
    plt.figure()
    plt.boxplot(data, labels=families, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE (outer-test per fold)")
    plt.title(f"{dataset_name} | RMSE distribution")
    plt.tight_layout()
    path = os.path.join(out_dir_figs, f"bench_rmse_box_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def fig_bench_rmse_r2_scatter(
    bench_df: pd.DataFrame, out_dir_figs: str, dataset_name: str
):
    plt.figure()
    for fam in bench_df["family"].unique():
        sf = bench_df[bench_df["family"] == fam]
        plt.scatter(sf["test_rmse_median"], sf["test_r2_median"], label=fam, alpha=0.7)
    plt.xlabel("RMSE (outer-test)")
    plt.ylabel("R² (outer-test)")
    plt.title(f"{dataset_name} | RMSE vs R² (folds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir_figs, f"bench_rmse_r2_scatter_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def fig_learning_curve_train_cv_panels(
    lc_agg_df: pd.DataFrame, out_dir_figs: str, dataset_name: str
):
    fams = ["PLSR", "ANN", "CNN1D"]
    fams = [f for f in fams if f in lc_agg_df["family"].unique()]

    n_panels = len(fams)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, fam in zip(axes, fams):
        sf = lc_agg_df[lc_agg_df["family"] == fam].sort_values("train_n")

        ax.errorbar(
            sf["train_n"],
            sf["train_rmse_mean"],
            yerr=sf["train_rmse_std"],
            marker="o",
            label="Train RMSE",
        )
        ax.errorbar(
            sf["train_n"],
            sf["cv_rmse_mean"],
            yerr=sf["cv_rmse_std"],
            marker="o",
            label="CV RMSE",
        )

        ax.set_title(f"{dataset_name} | {fam} | Learning Curve (Train vs CV)")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Train N (subsample size)")
    plt.tight_layout()

    path = os.path.join(out_dir_figs, f"learning_curve_train_cv_panels_{dataset_name}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================
# 13) Main runner
# =========================================================

def main():
    cfg = default_cfg()

    DATASETS: List[DatasetConfig] = [
        DatasetConfig(
            name="mangos_TA_Vit_C_TA",
            kind="excel",
            path="data/mangos_TA_Vit_C.xlsx",
            satir_ilk=1,
            satir_son=59,
            bas_sutun=4,
            bit_sutun=1560,
            col_Y=2,
        ),
        DatasetConfig(
            name="mangos_TA_Vit_C_VitC",
            kind="excel",
            path="data/mangos_TA_Vit_C.xlsx",
            satir_ilk=1,
            satir_son=59,
            bas_sutun=4,
            bit_sutun=1560,
            col_Y=3,
        ),
        DatasetConfig(
            name="Cucurbitaceae_Fruits_Water",
            kind="excel",
            path="data/Cucurbitaceae_Fruits.xlsx",
            satir_ilk=1,
            satir_son=301,
            bas_sutun=3,
            bit_sutun=232,
            col_Y=2,
        ),
        DatasetConfig(
            name="Cucurbitaceae_Fruits_Brix",
            kind="excel",
            path="data/Cucurbitaceae_Fruits.xlsx",
            satir_ilk=1,
            satir_son=301,
            bas_sutun=3,
            bit_sutun=232,
            col_Y=3,
        ),
        DatasetConfig(
            name="Milk",
            kind="csv",
            path="data/milk.csv",
            satir_ilk=0,
            satir_son=1224,
            bas_sutun=270,
            bit_sutun=526,
            col_Y=1,
            sep=",",
        ),
        DatasetConfig(
            name="Mangoes_Cvit",
            kind="excel",
            path="data/Mangoes.xlsx",
            satir_ilk=1,
            satir_son=187,
            bas_sutun=5,
            bit_sutun=1561,
            col_Y=2,
        ),
        DatasetConfig(
            name="Mangoes_TA",
            kind="excel",
            path="data/Mangoes.xlsx",
            satir_ilk=1,
            satir_son=187,
            bas_sutun=5,
            bit_sutun=1561,
            col_Y=3,
        ),
        DatasetConfig(
            name="Mangoes_Brix",
            kind="excel",
            path="data/Mangoes.xlsx",
            satir_ilk=1,
            satir_son=187,
            bas_sutun=5,
            bit_sutun=1561,
            col_Y=4,
        ),
        DatasetConfig(
            name="DATASET_csv",
            kind="csv",
            path="data/DATASET.csv",
            satir_ilk=0,
            satir_son=75,
            bas_sutun=3,
            bit_sutun=206,
            col_Y=2,
            sep=";",
        ),
        DatasetConfig(
            name="Cary_centrifuged_Brix_averaged",
            kind="excel",
            path="data/Cary_centrifuged_Brix_averaged.xlsx",
            satir_ilk=2,
            satir_son=645,
            bas_sutun=1198,
            bit_sutun=1437,
            col_Y=1772,
        ),
    ]

    if len(DATASETS) == 0:
        raise RuntimeError("DATASETS listesi boş.")

    for dcfg in DATASETS:
        dirs = ensure_dataset_dirs(dcfg.name)
        print("\n" + "=" * 90)
        print(f"DATASET: {dcfg.name}")
        print(f"Saving to: {dirs['base']}")
        print("=" * 90)

        X, y = load_dataset(dcfg)
        n, p = X.shape
        families = [k for k, v in MODEL_REGISTRY.items() if v.get("enabled", False)]
        do_lc = bool(cfg.run_learning_curve)

        total_fits = estimate_total_fits_for_dataset(cfg, families, do_lc)
        print(f"N={n} | p={p} | Families={families}")
        print(f"TOTAL FITS (estimated) for this dataset = {total_fits}")
        progress = FitProgress(total_fits)

        bench_df, candlog_df = run_outer_cv_benchmark(X, y, dcfg.name, cfg, progress)
        bench_summary_df = make_benchmark_summary(bench_df)
        freq_df = best_config_frequencies(bench_df)

        save_table_xlsx(
            os.path.join(dirs["tables"], "bench_raw.xlsx"), bench_df, "bench_raw"
        )
        save_table_xlsx(
            os.path.join(dirs["tables"], "bench_summary.xlsx"),
            bench_summary_df,
            "bench_summary",
        )
        save_table_xlsx(
            os.path.join(dirs["tables"], "best_config_freq.xlsx"),
            freq_df,
            "best_config_freq",
        )

        run_log_df = progress.to_dataframe()
        save_table_xlsx(
            os.path.join(dirs["tables"], "run_log.xlsx"), run_log_df, "run_log"
        )

        input_len = X.shape[1]
        complexity_tables = build_dl_complexity_tables(
            dataset_name=dcfg.name,
            bench_df=bench_df,
            candlog_df=candlog_df,
            input_len=input_len,
        )

        save_multi_table_xlsx(
            os.path.join(dirs["tables"], "dl_complexity.xlsx"),
            complexity_tables,
        )

        fig_dl_complexity_rmse_panels(
            complexity_tables["inner_candidates_agg"],
            dirs["figs"],
            dcfg.name,
        )

        fig_bench_rmse_bar(bench_summary_df, dirs["figs"], dcfg.name)
        fig_bench_r2_bar(bench_summary_df, dirs["figs"], dcfg.name)
        fig_bench_rmse_box(bench_df, dirs["figs"], dcfg.name)
        fig_bench_rmse_r2_scatter(bench_df, dirs["figs"], dcfg.name)

        if do_lc:
            lc_select_df, lc_raw_df, lc_agg_df = run_learning_curve(
                X, y, dcfg.name, cfg, progress
            )
            lc_summary_df = summarize_learning_curve(lc_agg_df)

            save_table_xlsx(
                os.path.join(dirs["tables"], "learning_curve_selected_config.xlsx"),
                lc_select_df,
                "lc_selected",
            )
            save_table_xlsx(
                os.path.join(dirs["tables"], "learning_curve_raw.xlsx"),
                lc_raw_df,
                "lc_raw",
            )
            save_table_xlsx(
                os.path.join(dirs["tables"], "learning_curve_agg.xlsx"),
                lc_agg_df,
                "lc_agg",
            )
            save_table_xlsx(
                os.path.join(dirs["tables"], "learning_curve_summary.xlsx"),
                lc_summary_df,
                "lc_summary",
            )

            fig_learning_curve_train_cv_panels(lc_agg_df, dirs["figs"], dcfg.name)

        del X, y, bench_df, candlog_df, complexity_tables, bench_summary_df
        del freq_df, run_log_df
        if do_lc:
            del lc_select_df, lc_raw_df, lc_agg_df, lc_summary_df
        gc.collect()
        safe_clear_tf()

        print(f"\n[{dcfg.name}] DONE. Outputs saved under: {dirs['base']}")

    print("\nALL DATASETS DONE. Check outputs/<dataset_name>/ folders.")


if __name__ == "__main__":
    main()
