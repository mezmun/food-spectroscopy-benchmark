# -*- coding: utf-8 -*-
"""
Food NIR spectroscopy regression benchmark.

The pipeline provides two complementary evaluation tracks:

1. Leakage-safe nested cross-validation on the complete dataset, preserving the
   original benchmark design for repeated model-selection/performance estimation.
2. A fixed independent hold-out test set. Hyperparameter selection is performed
   only on the development subset, and the locked test subset is used only for
   final prediction assessment and test-based learning curves.

The script compares PLSR, Ridge, SVR, ANN, and 1D-CNN models and exports raw
predictions, summary tables, paired randomization tests, learning curves,
complexity analyses, publication figures, CSV files, Excel workbooks, and LaTeX
code for quantitative manuscript tables.

Datasets are not distributed with this repository. Use --synthetic for a fully
self-contained simulation and --smoke-test for a fast end-to-end check.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers
    TF_AVAILABLE = True
except Exception:
    tf = None
    callbacks = layers = models = optimizers = None
    TF_AVAILABLE = False

MODEL_ORDER = ["PLSR", "Ridge", "SVR", "ANN", "CNN1D"]
TASK_ORDER = [
    "Mango-A (TA)",
    "Mango-A (Vitamin C)",
    "Cucurbitaceae (Water)",
    "Cucurbitaceae (Brix)",
    "Milk (Fat)",
    "Mango-B (TA)",
    "Mango-B (Vitamin C)",
    "Mango-B (Brix)",
    "Grapes (Sugar)",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def safe_clear_tf() -> None:
    if TF_AVAILABLE:
        tf.keras.backend.clear_session()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def sample_std(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    return 0.0 if x.size <= 1 else float(np.std(x, ddof=1))


def safe_slug(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def reorder_tasks(df: pd.DataFrame, task_col: str = "task") -> pd.DataFrame:
    if df.empty or task_col not in df.columns:
        return df
    order = {name: i for i, name in enumerate(TASK_ORDER)}
    out = df.copy()
    out["__order"] = out[task_col].map(order).fillna(10_000)
    out = out.sort_values(["__order"]).drop(columns="__order").reset_index(drop=True)
    return out


def reorder_models(df: pd.DataFrame, model_col: str = "family") -> pd.DataFrame:
    if df.empty or model_col not in df.columns:
        return df
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    out = df.copy()
    out["__order"] = out[model_col].map(order).fillna(10_000)
    out = out.sort_values(["__order"]).drop(columns="__order").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class DatasetConfig:
    task: str
    kind: str
    path: str
    row_start: int
    row_stop: int
    feature_start: int
    feature_stop: int
    target_col: int
    sep: str = ","
    sheet_name: Optional[str] = None
    target_unit: str = ""
    spectral_range: str = ""
    expected_n: Optional[int] = None
    expected_p: Optional[int] = None


def manuscript_datasets() -> List[DatasetConfig]:
    """Return the nine regression tasks used in the manuscript benchmark.

    Column indices follow zero-based Python indexing. Excel sheet names are
    specified explicitly to make data loading independent of workbook sheet order.
    """
    return [
        DatasetConfig(
            "Mango-A (TA)", "excel", "data/mangos_TA_Vit_C.xlsx",
            1, 59, 4, 1561, 2, sheet_name="Raw Spectra data",
            target_unit="mg/100gr FM", spectral_range="999.9-2500.2 nm",
            expected_n=58, expected_p=1557,
        ),
        DatasetConfig(
            "Mango-A (Vitamin C)", "excel", "data/mangos_TA_Vit_C.xlsx",
            1, 59, 4, 1561, 3, sheet_name="Raw Spectra data",
            target_unit="mg/100gr FM", spectral_range="999.9-2500.2 nm",
            expected_n=58, expected_p=1557,
        ),
        DatasetConfig(
            "Cucurbitaceae (Water)", "excel", "data/Cucurbitaceae_Fruits.xlsx",
            1, 301, 3, 232, 1, sheet_name="Calibration Set",
            target_unit="%", spectral_range="381-1065 nm",
            expected_n=300, expected_p=229,
        ),
        DatasetConfig(
            "Cucurbitaceae (Brix)", "excel", "data/Cucurbitaceae_Fruits.xlsx",
            1, 301, 3, 232, 2, sheet_name="Calibration Set",
            target_unit="% Brix", spectral_range="381-1065 nm",
            expected_n=300, expected_p=229,
        ),
        DatasetConfig(
            "Milk (Fat)", "csv", "data/milk.csv",
            0, 1224, 270, 526, 1, sep=",",
            target_unit="%", spectral_range="960-1690 nm",
            expected_n=1224, expected_p=256,
        ),
        DatasetConfig(
            "Mango-B (TA)", "excel", "data/Mangoes.xlsx",
            1, 187, 5, 1562, 3, sheet_name="RawData",
            target_unit="mg/100g", spectral_range="999.9-2500.2 nm",
            expected_n=186, expected_p=1557,
        ),
        DatasetConfig(
            "Mango-B (Vitamin C)", "excel", "data/Mangoes.xlsx",
            1, 187, 5, 1562, 2, sheet_name="RawData",
            target_unit="mg/100g", spectral_range="999.9-2500.2 nm",
            expected_n=186, expected_p=1557,
        ),
        DatasetConfig(
            "Mango-B (Brix)", "excel", "data/Mangoes.xlsx",
            1, 187, 5, 1562, 4, sheet_name="RawData",
            target_unit="deg Brix", spectral_range="999.9-2500.2 nm",
            expected_n=186, expected_p=1557,
        ),
        DatasetConfig(
            "Grapes (Sugar)", "csv", "data/DATASET.csv",
            0, 274, 3, 207, 2, sep=";",
            target_unit="g/L", spectral_range="397.32-1003.5 nm",
            expected_n=274, expected_p=204,
        ),
    ]


def validate_column_definition(cfg: DatasetConfig) -> None:
    if cfg.feature_start <= cfg.target_col < cfg.feature_stop:
        raise ValueError(
            f"Target leakage in {cfg.task}: target_col={cfg.target_col} lies inside "
            f"feature slice [{cfg.feature_start}:{cfg.feature_stop})."
        )
    if cfg.row_stop <= cfg.row_start or cfg.feature_stop <= cfg.feature_start:
        raise ValueError(f"Invalid slice definition for {cfg.task}.")


def load_dataset(cfg: DatasetConfig, repo_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    validate_column_definition(cfg)
    path = repo_root / cfg.path
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    if cfg.kind == "excel":
        df = pd.read_excel(
            path, engine="openpyxl", header=None,
            sheet_name=cfg.sheet_name if cfg.sheet_name is not None else 0,
        )
    elif cfg.kind == "csv":
        df = pd.read_csv(path, sep=cfg.sep)
    else:
        raise ValueError(f"Unknown dataset kind: {cfg.kind}")
    data = df.values
    if cfg.row_stop > data.shape[0] or max(cfg.feature_stop - 1, cfg.target_col) >= data.shape[1]:
        raise ValueError(f"Configured slice exceeds file dimensions for {cfg.task}: shape={data.shape}.")
    y = data[cfg.row_start:cfg.row_stop, cfg.target_col].astype("float32").ravel()
    X = data[cfg.row_start:cfg.row_stop, cfg.feature_start:cfg.feature_stop].astype("float32")
    return X, y


def synthetic_dataset(cfg: DatasetConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(max(54, cfg.expected_n or 72), 90)
    p = 64
    latent = rng.normal(size=(n, 6))
    X = latent @ rng.normal(size=(6, p)) + 0.20 * rng.normal(size=(n, p))
    X = (np.roll(X, 1, axis=1) + 2 * X + np.roll(X, -1, axis=1)) / 4.0
    difficulty = {
        "Mango-A (TA)": 0.35,
        "Mango-A (Vitamin C)": 0.75,
        "Cucurbitaceae (Water)": 0.30,
        "Cucurbitaceae (Brix)": 0.45,
        "Milk (Fat)": 0.25,
        "Mango-B (TA)": 1.20,
        "Mango-B (Vitamin C)": 0.90,
        "Mango-B (Brix)": 1.10,
        "Grapes (Sugar)": 0.70,
    }.get(cfg.task, 0.70)
    beta = rng.normal(size=p) * np.exp(-np.linspace(0, 2.5, p))
    signal = X @ beta
    nonlinear = 0.12 * latent[:, 0] ** 2 - 0.08 * latent[:, 1] * latent[:, 2]
    y = signal + nonlinear + difficulty * np.std(signal) * rng.normal(size=n)
    return X.astype("float32"), y.astype("float32")


def audit_dataset(X: np.ndarray, y: np.ndarray, cfg: DatasetConfig, synthetic: bool) -> Dict[str, Any]:
    validate_column_definition(cfg)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{cfg.task}: inconsistent X/y dimensions.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{cfg.task}: non-finite values detected.")
    if np.std(y) == 0:
        raise ValueError(f"{cfg.task}: constant target.")
    exact_matches = sum(np.allclose(X[:, j], y, rtol=0.0, atol=1e-10) for j in range(X.shape[1]))
    if exact_matches:
        raise ValueError(f"{cfg.task}: a predictor exactly reproduces the target; check column mapping.")
    _, inv, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    dup_rows = int(np.sum(np.maximum(counts - 1, 0)))
    conflict = 0
    for gid in np.where(counts > 1)[0]:
        if np.ptp(y[inv == gid]) > 1e-8:
            conflict += 1
    n, p = X.shape
    return {
        "task": cfg.task, "file": cfg.path, "N": n, "p": p, "N_over_p": n / p,
        "target_unit": cfg.target_unit, "spectral_range": cfg.spectral_range,
        "target_min": float(y.min()), "target_max": float(y.max()),
        "target_mean": float(y.mean()), "target_sd": float(np.std(y, ddof=1)),
        "duplicate_spectra_rows": dup_rows,
        "duplicate_spectra_conflicting_target_groups": conflict,
        "exact_target_feature_matches": exact_matches,
        "expected_N": cfg.expected_n, "expected_p": cfg.expected_p,
        "expected_N_match": None if synthetic or cfg.expected_n is None else bool(n == cfg.expected_n),
        "expected_p_match": None if synthetic or cfg.expected_p is None else bool(p == cfg.expected_p),
    }


def make_locked_holdout_split(n: int, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    dev, test = train_test_split(np.arange(n), test_size=test_fraction, random_state=seed, shuffle=True)
    return np.sort(dev), np.sort(test)


@dataclass
class PreprocessConfig:
    use_autoscale: bool = True


def preprocess_fit_apply(X_train_raw: np.ndarray, X_others_raw: Sequence[np.ndarray], cfg: PreprocessConfig):
    Xtr = X_train_raw.copy()
    others = [x.copy() for x in X_others_raw]
    scaler = None
    if cfg.use_autoscale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        others = [scaler.transform(x) for x in others]
    return Xtr, others, scaler


def fit_y_scaler(y: np.ndarray) -> StandardScaler:
    s = StandardScaler()
    s.fit(y.reshape(-1, 1))
    return s


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "PLSR": {"enabled": True, "candidates": [{"n_comp": n} for n in [5, 10, 15, 20, 25, 30]]},
    "Ridge": {"enabled": True, "candidates": [{"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]]},
    "SVR": {"enabled": True, "candidates": [{"C": c, "gamma": "scale", "epsilon": 0.1} for c in [0.3, 1, 3, 10, 30, 100]]},
    "ANN": {"enabled": True, "candidates": [{"units": u} for u in [[8], [12], [16], [24], [32], [48], [64], [16,8], [32,16], [64,32], [96,48], [128,64]]], "dropout": 0.3},
    "CNN1D": {"enabled": True, "candidates": [
        {"filters":[8],"kernel":3},{"filters":[16],"kernel":3},{"filters":[32],"kernel":3},
        {"filters":[8],"kernel":5},{"filters":[16],"kernel":5},{"filters":[32],"kernel":5},
        {"filters":[16],"kernel":7},{"filters":[32],"kernel":7},
        {"filters":[16,8],"kernel":3},{"filters":[32,16],"kernel":3},
        {"filters":[16,8],"kernel":5},{"filters":[32,16],"kernel":5}],
        "dropout":0.2, "use_layernorm":True},
}


def is_dl_family(family: str) -> bool:
    return family in {"ANN", "CNN1D"}


def build_mlp(input_len: int, units: List[int], dropout: float):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for ANN/CNN1D models.")
    inp = layers.Input(shape=(input_len,))
    x = inp
    for u in units:
        x = layers.Dense(u, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse",
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def build_cnn1d(input_len: int, filters: List[int], kernel: int, dropout: float, use_layernorm: bool):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for ANN/CNN1D models.")
    inp = layers.Input(shape=(input_len, 1))
    x = inp
    for f in filters:
        x = layers.Conv1D(f, kernel, padding="same")(x)
        if use_layernorm:
            x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse",
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def build_callbacks(patience: int):
    return [callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_rmse", mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=0)]


def fit_classical_model(family: str, cand: Dict[str, Any], X: np.ndarray, y: np.ndarray):
    if family == "PLSR":
        m = PLSRegression(n_components=int(cand["n_comp"]), scale=False)
    elif family == "Ridge":
        m = Ridge(alpha=float(cand["alpha"]))
    elif family == "SVR":
        m = SVR(kernel="rbf", C=float(cand["C"]), gamma=cand["gamma"], epsilon=float(cand["epsilon"]))
    else:
        raise ValueError(f"Unsupported family: {family}")
    m.fit(X, y)
    return m


def complexity_for_candidate(family: str, cand: Dict[str, Any], input_len: int):
    if family == "ANN":
        prev, params, flops = input_len, 0, 0
        for u in cand["units"]:
            params += (prev + 1) * u; flops += 2 * prev * u; prev = u
        params += prev + 1; flops += 2 * prev
        return int(params), int(flops)
    if family == "CNN1D":
        in_ch, params, flops = 1, 0, 0
        for f in cand["filters"]:
            params += (cand["kernel"] * in_ch + 1) * f
            if MODEL_REGISTRY["CNN1D"]["use_layernorm"]:
                params += 2 * f
            flops += input_len * 2 * cand["kernel"] * in_ch * f
            in_ch = f
        params += in_ch + 1; flops += 2 * in_ch
        return int(params), int(flops)
    return None, None


@dataclass
class ExperimentConfig:
    outer_folds: int = 3
    inner_folds: int = 3
    epochs: int = 400
    batch_size: int = 16
    patience: int = 30
    dl_seeds_inner: List[int] = field(default_factory=lambda: [0, 1])
    dl_seeds_final: List[int] = field(default_factory=lambda: [0, 1])
    dl_validation_fraction: float = 0.15
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    holdout_fraction: float = 0.20
    holdout_seed: int = 2026
    run_learning_curves: bool = True
    learning_curve_points: int = 10
    learning_curve_families: List[str] = field(default_factory=lambda: ["PLSR", "ANN", "CNN1D"])
    learning_curve_repeats: int = 2
    learning_curve_cv_folds: int = 3
    learning_curve_seed: int = 700
    randomization_trials: int = 9999
    randomization_seed: int = 2026
    output_root: str = "outputs/revision_results"
    disable_dl: bool = False
    synthetic: bool = False
    smoke_test: bool = False


def active_families(cfg: ExperimentConfig) -> List[str]:
    fams = [f for f in MODEL_ORDER if MODEL_REGISTRY[f]["enabled"]]
    if cfg.disable_dl:
        fams = [f for f in fams if not is_dl_family(f)]
    if any(is_dl_family(f) for f in fams) and not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install requirements or use --disable-dl for a classical-model test.")
    return fams


def apply_smoke_profile(cfg: ExperimentConfig) -> None:
    cfg.outer_folds = 2; cfg.inner_folds = 2; cfg.epochs = 8; cfg.patience = 2
    cfg.dl_seeds_inner = [0]; cfg.dl_seeds_final = [0]
    cfg.learning_curve_points = 3; cfg.learning_curve_repeats = 1; cfg.learning_curve_cv_folds = 2
    cfg.randomization_trials = 199
    MODEL_REGISTRY["PLSR"]["candidates"] = [{"n_comp":2},{"n_comp":4}]
    MODEL_REGISTRY["Ridge"]["candidates"] = [{"alpha":0.1},{"alpha":1.0}]
    MODEL_REGISTRY["SVR"]["candidates"] = [{"C":1.0,"gamma":"scale","epsilon":0.1},{"C":10.0,"gamma":"scale","epsilon":0.1}]
    MODEL_REGISTRY["ANN"]["candidates"] = [{"units":[8]},{"units":[16]}]
    MODEL_REGISTRY["CNN1D"]["candidates"] = [{"filters":[8],"kernel":3},{"filters":[8],"kernel":5}]


class FitProgress:
    def __init__(self): self.done = 0; self.records: List[Dict[str, Any]] = []
    def step(self, msg: str):
        self.done += 1; print(f"[FIT {self.done}] {msg}"); self.records.append({"fit":self.done,"message":msg})
    def to_dataframe(self): return pd.DataFrame(self.records)


def kfold_splits(n: int, n_splits: int, seed: int):
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def candidate_valid_for_splits(family: str, cand: Dict[str, Any], splits, n_features: int) -> bool:
    if family != "PLSR": return True
    max_comp = min(n_features, min(len(tr) for tr, _ in splits) - 1)
    return cand["n_comp"] <= max_comp


def eval_candidate_classical_cv(family, cand, X, y, splits, cfg, task, stage, progress):
    scores = []
    for fold, (tr, va) in enumerate(splits, 1):
        Xtr, [Xva], _ = preprocess_fit_apply(X[tr], [X[va]], cfg.preprocess)
        m = fit_classical_model(family, cand, Xtr, y[tr]); pred = np.asarray(m.predict(Xva)).ravel()
        scores.append(rmse(y[va], pred)); progress.step(f"{task} | {stage} | {family} | fold={fold} | {cand}")
    return float(np.mean(scores))


def eval_candidate_dl_cv(family, cand, X, y, splits, cfg, task, stage, progress):
    seed_scores = []
    for seed in cfg.dl_seeds_inner:
        fold_scores = []
        for fold, (tr, va) in enumerate(splits, 1):
            set_seed(seed); safe_clear_tf()
            Xtr, [Xva], _ = preprocess_fit_apply(X[tr], [X[va]], cfg.preprocess)
            ys = fit_y_scaler(y[tr]); ytr_s = ys.transform(y[tr].reshape(-1,1)).ravel(); yva_s = ys.transform(y[va].reshape(-1,1)).ravel()
            if family == "ANN":
                m = build_mlp(Xtr.shape[1], cand["units"], MODEL_REGISTRY["ANN"]["dropout"]); Xtr_i, Xva_i = Xtr, Xva
            else:
                m = build_cnn1d(Xtr.shape[1], cand["filters"], cand["kernel"], MODEL_REGISTRY["CNN1D"]["dropout"], MODEL_REGISTRY["CNN1D"]["use_layernorm"]); Xtr_i, Xva_i = Xtr[...,None], Xva[...,None]
            m.fit(Xtr_i, ytr_s, validation_data=(Xva_i,yva_s), epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=build_callbacks(cfg.patience), verbose=0)
            pred_s = m.predict(Xva_i, verbose=0).ravel(); pred = ys.inverse_transform(pred_s.reshape(-1,1)).ravel()
            fold_scores.append(rmse(y[va], pred)); progress.step(f"{task} | {stage} | {family} | seed={seed} | fold={fold} | {cand}")
        seed_scores.append(float(np.mean(fold_scores)))
    return float(np.median(seed_scores))


def select_best_config(family, X, y, cfg, seed, task, stage, progress):
    splits = kfold_splits(len(X), cfg.inner_folds, seed)
    rows, best_cand, best_score = [], None, math.inf
    for cand in MODEL_REGISTRY[family]["candidates"]:
        params, flops = complexity_for_candidate(family, cand, X.shape[1])
        if not candidate_valid_for_splits(family, cand, splits, X.shape[1]):
            rows.append({"task":task,"stage":stage,"family":family,"candidate":str(cand),"inner_cv_rmse":np.nan,"param_count":params,"approx_flops":flops,"status":"skipped_invalid_for_fold_size"}); continue
        score = eval_candidate_dl_cv(family,cand,X,y,splits,cfg,task,stage,progress) if is_dl_family(family) else eval_candidate_classical_cv(family,cand,X,y,splits,cfg,task,stage,progress)
        rows.append({"task":task,"stage":stage,"family":family,"candidate":str(cand),"inner_cv_rmse":score,"param_count":params,"approx_flops":flops,"status":"evaluated"})
        if score < best_score: best_score, best_cand = score, dict(cand)
    if best_cand is None: raise RuntimeError(f"No valid candidate for {task}/{family}.")
    return best_cand, float(best_score), rows


def fit_dl_predict_final(
    family: str,
    cand: Dict[str, Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    task: str,
    stage: str,
    progress: FitProgress,
) -> Tuple[np.ndarray, int]:
    """Fit a DL model without using the final test set for stopping decisions.

    An internal validation subset of the supplied training data is used only to
    estimate a suitable training duration. The same architecture is then rebuilt
    and refit on the complete training data for that number of epochs before
    predictions are generated for the external evaluation subset.
    """
    set_seed(seed)
    safe_clear_tf()

    indices = np.arange(len(X_train_raw))
    tr_idx, va_idx = train_test_split(
        indices,
        test_size=cfg.dl_validation_fraction,
        random_state=seed + 991,
        shuffle=True,
    )

    X_sub, [X_val], _ = preprocess_fit_apply(
        X_train_raw[tr_idx], [X_train_raw[va_idx]], cfg.preprocess
    )
    y_scaler = fit_y_scaler(y_train[tr_idx])
    y_sub = y_scaler.transform(y_train[tr_idx].reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_train[va_idx].reshape(-1, 1)).ravel()

    if family == "ANN":
        model = build_mlp(X_sub.shape[1], cand["units"], MODEL_REGISTRY["ANN"]["dropout"])
        X_sub_i, X_val_i = X_sub, X_val
    elif family == "CNN1D":
        model = build_cnn1d(
            X_sub.shape[1],
            cand["filters"],
            cand["kernel"],
            MODEL_REGISTRY["CNN1D"]["dropout"],
            MODEL_REGISTRY["CNN1D"]["use_layernorm"],
        )
        X_sub_i, X_val_i = X_sub[..., None], X_val[..., None]
    else:
        raise ValueError(f"Unsupported DL family: {family}")

    history = model.fit(
        X_sub_i,
        y_sub,
        validation_data=(X_val_i, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=build_callbacks(cfg.patience),
        verbose=0,
    )
    val_curve = history.history.get("val_rmse", [])
    best_epoch = int(np.argmin(val_curve) + 1) if val_curve else int(len(history.history.get("loss", [])))
    best_epoch = max(1, min(best_epoch, cfg.epochs))
    progress.step(f"{task} | {stage} | {family} | seed={seed} | epoch-selection")

    # Refit on all available training samples. The external test set remains locked.
    set_seed(seed)
    safe_clear_tf()
    X_full, [X_test], _ = preprocess_fit_apply(X_train_raw, [X_test_raw], cfg.preprocess)
    y_scaler_full = fit_y_scaler(y_train)
    y_full = y_scaler_full.transform(y_train.reshape(-1, 1)).ravel()

    if family == "ANN":
        final_model = build_mlp(X_full.shape[1], cand["units"], MODEL_REGISTRY["ANN"]["dropout"])
        X_full_i, X_test_i = X_full, X_test
    else:
        final_model = build_cnn1d(
            X_full.shape[1],
            cand["filters"],
            cand["kernel"],
            MODEL_REGISTRY["CNN1D"]["dropout"],
            MODEL_REGISTRY["CNN1D"]["use_layernorm"],
        )
        X_full_i, X_test_i = X_full[..., None], X_test[..., None]

    final_model.fit(
        X_full_i,
        y_full,
        epochs=best_epoch,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    pred_scaled = final_model.predict(X_test_i, verbose=0).ravel()
    pred = y_scaler_full.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    progress.step(f"{task} | {stage} | {family} | seed={seed} | final-fit")
    return pred.astype(float), best_epoch


def evaluate_selected_model(
    family: str,
    cand: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
    cfg: ExperimentConfig,
    task: str,
    stage: str,
    progress: FitProgress,
    seed_override: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    if is_dl_family(family):
        seeds = list(seed_override) if seed_override is not None else list(cfg.dl_seeds_final)
        for seed in seeds:
            pred, best_epoch = fit_dl_predict_final(
                family, cand, X_train, y_train, X_test, cfg, int(seed), task, stage, progress
            )
            metric_rows.append(
                {
                    "task": task,
                    "stage": stage,
                    "family": family,
                    "seed": int(seed),
                    "rmse": rmse(y_test, pred),
                    "r2": float(r2_score(y_test, pred)),
                    "best_epoch": int(best_epoch),
                    "best_config": str(cand),
                }
            )
            for sample_index, yt, yp in zip(test_indices, y_test, pred):
                err = float(yt - yp)
                prediction_rows.append(
                    {
                        "task": task,
                        "stage": stage,
                        "family": family,
                        "seed": int(seed),
                        "sample_index": int(sample_index),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "residual": err,
                        "squared_error": err * err,
                        "best_epoch": int(best_epoch),
                        "best_config": str(cand),
                    }
                )
    else:
        Xtr, [Xte], _ = preprocess_fit_apply(X_train, [X_test], cfg.preprocess)
        model = fit_classical_model(family, cand, Xtr, y_train)
        pred = np.asarray(model.predict(Xte)).ravel().astype(float)
        progress.step(f"{task} | {stage} | {family} | final-fit")
        metric_rows.append(
            {
                "task": task,
                "stage": stage,
                "family": family,
                "seed": np.nan,
                "rmse": rmse(y_test, pred),
                "r2": float(r2_score(y_test, pred)),
                "best_epoch": np.nan,
                "best_config": str(cand),
            }
        )
        for sample_index, yt, yp in zip(test_indices, y_test, pred):
            err = float(yt - yp)
            prediction_rows.append(
                {
                    "task": task,
                    "stage": stage,
                    "family": family,
                    "seed": np.nan,
                    "sample_index": int(sample_index),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "residual": err,
                    "squared_error": err * err,
                    "best_epoch": np.nan,
                    "best_config": str(cand),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    task: str,
    progress: FitProgress,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    candidate_rows: List[Dict[str, Any]] = []
    outer = KFold(n_splits=cfg.outer_folds, shuffle=True, random_state=42)

    for outer_fold, (tr, te) in enumerate(outer.split(X), 1):
        for family in active_families(cfg):
            best, inner_score, rows = select_best_config(
                family,
                X[tr],
                y[tr],
                cfg,
                seed=100 + outer_fold,
                task=task,
                stage=f"NESTED_OUTER_{outer_fold}_SELECT",
                progress=progress,
            )
            for row in rows:
                row["outer_fold"] = outer_fold
                candidate_rows.append(row)

            metrics, predictions = evaluate_selected_model(
                family,
                best,
                X[tr],
                y[tr],
                X[te],
                y[te],
                np.asarray(te),
                cfg,
                task,
                stage=f"NESTED_OUTER_{outer_fold}_TEST",
                progress=progress,
            )
            rmse_values = metrics["rmse"].astype(float).values
            r2_values = metrics["r2"].astype(float).values
            raw_rows.append(
                {
                    "task": task,
                    "outer_fold": outer_fold,
                    "family": family,
                    "inner_cv_rmse": inner_score,
                    "best_config": str(best),
                    "test_rmse_mean": float(np.mean(rmse_values)),
                    "test_rmse_median": float(np.median(rmse_values)),
                    "test_rmse_sd_across_seeds": sample_std(rmse_values),
                    "test_r2_mean": float(np.mean(r2_values)),
                    "test_r2_median": float(np.median(r2_values)),
                    "test_r2_sd_across_seeds": sample_std(r2_values),
                    "n_test": int(len(te)),
                    "n_final_seeds": int(len(metrics)),
                }
            )
            predictions = predictions.copy()
            predictions["outer_fold"] = outer_fold
            pred_frames.append(predictions)

    return (
        pd.DataFrame(raw_rows),
        pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
        pd.DataFrame(candidate_rows),
    )


def summarize_nested_cv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (task, family), g in raw.groupby(["task", "family"], sort=False):
        rm = g["test_rmse_median"].astype(float).values
        rr = g["test_r2_median"].astype(float).values
        rows.append(
            {
                "task": task,
                "family": family,
                "rmse_mean": float(np.mean(rm)),
                "rmse_std": sample_std(rm),
                "r2_mean": float(np.mean(rr)),
                "r2_std": sample_std(rr),
                "n_outer_folds": int(len(g)),
            }
        )
    return reorder_models(reorder_tasks(pd.DataFrame(rows)))


def aggregate_test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    keys = ["task", "stage", "sample_index", "family"]
    for key, g in predictions.groupby(keys, sort=False, dropna=False):
        yp = g["y_pred"].astype(float).values
        se = g["squared_error"].astype(float).values
        yt = float(g["y_true"].iloc[0])
        yp_mean = float(np.mean(yp))
        rows.append(
            {
                "task": key[0],
                "stage": key[1],
                "sample_index": int(key[2]),
                "family": key[3],
                "y_true": yt,
                "y_pred_mean": yp_mean,
                "y_pred_sd": sample_std(yp),
                "squared_error_mean_across_seeds": float(np.mean(se)),
                "squared_error_from_mean_prediction": float((yt - yp_mean) ** 2),
                "n_prediction_seeds": int(len(g)),
            }
        )
    return reorder_models(reorder_tasks(pd.DataFrame(rows)))


def run_independent_test(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    task: str,
    progress: FitProgress,
):
    dev_idx, test_idx = make_locked_holdout_split(len(X), cfg.holdout_fraction, cfg.holdout_seed)
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    summary_rows: List[Dict[str, Any]] = []
    seed_metric_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    candidate_rows: List[Dict[str, Any]] = []
    best_map: Dict[str, Dict[str, Any]] = {}

    for family in active_families(cfg):
        best, inner_score, rows = select_best_config(
            family,
            X_dev,
            y_dev,
            cfg,
            seed=cfg.holdout_seed + 17,
            task=task,
            stage="HOLDOUT_SELECT",
            progress=progress,
        )
        best_map[family] = best
        candidate_rows.extend(rows)

        metrics, predictions = evaluate_selected_model(
            family,
            best,
            X_dev,
            y_dev,
            X_test,
            y_test,
            test_idx,
            cfg,
            task,
            stage="LOCKED_TEST",
            progress=progress,
        )
        metrics = metrics.copy()
        metrics["inner_cv_rmse"] = inner_score
        seed_metric_frames.append(metrics)
        prediction_frames.append(predictions)

    seed_metrics = pd.concat(seed_metric_frames, ignore_index=True) if seed_metric_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    pred_agg = aggregate_test_predictions(predictions)

    for family in active_families(cfg):
        g = seed_metrics[seed_metrics["family"] == family]
        pg = pred_agg[pred_agg["family"] == family]
        if g.empty or pg.empty:
            continue
        rmse_seed = g["rmse"].astype(float).values
        r2_seed = g["r2"].astype(float).values
        y_true = pg["y_true"].astype(float).values
        y_pred_mean = pg["y_pred_mean"].astype(float).values
        rmsep_pooled = float(np.sqrt(pg["squared_error_mean_across_seeds"].astype(float).mean()))
        summary_rows.append(
            {
                "task": task,
                "family": family,
                "n_development": int(len(dev_idx)),
                "n_test": int(len(test_idx)),
                "inner_cv_rmse_selected": float(g["inner_cv_rmse"].iloc[0]),
                "rmsep_pooled": rmsep_pooled,
                "rmsep_mean_prediction": rmse(y_true, y_pred_mean),
                "rmsep_seed_mean": float(np.mean(rmse_seed)),
                "rmsep_seed_median": float(np.median(rmse_seed)),
                "rmsep_seed_sd": sample_std(rmse_seed),
                "r2_mean_prediction": float(r2_score(y_true, y_pred_mean)),
                "r2_seed_mean": float(np.mean(r2_seed)),
                "r2_seed_median": float(np.median(r2_seed)),
                "r2_seed_sd": sample_std(r2_seed),
                "n_seeds": int(len(g)),
                "best_config": str(best_map[family]),
            }
        )

    return (
        reorder_models(pd.DataFrame(summary_rows)),
        seed_metrics,
        predictions,
        pred_agg,
        best_map,
        dev_idx,
        test_idx,
        pd.DataFrame(candidate_rows),
    )


def random_sign_pvalues(differences: np.ndarray, trials: int, seed: int) -> Tuple[float, float]:
    """Monte Carlo sign-randomization p-values for paired squared-error differences.

    The statistic is the mean paired difference in squared prediction error,
    comparator minus reference. Random sign exchange implements the paired
    randomization principle described by van der Voet (1994),
    doi:10.1016/0169-7439(94)85050-X.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan
    observed = float(np.mean(d))
    rng = np.random.default_rng(seed)
    ge = 0
    ge_abs = 0
    remaining = int(trials)
    chunk = 2000
    while remaining > 0:
        m = min(chunk, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, d.size))
        sim = np.mean(signs * d[None, :], axis=1)
        ge += int(np.sum(sim >= observed))
        ge_abs += int(np.sum(np.abs(sim) >= abs(observed)))
        remaining -= m
    p_one = (ge + 1.0) / (trials + 1.0)
    p_two = (ge_abs + 1.0) / (trials + 1.0)
    return float(p_one), float(p_two)


def van_der_voet_table(pred_agg: pd.DataFrame, trials: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if pred_agg.empty:
        return pd.DataFrame()
    for task, tg in pred_agg.groupby("task", sort=False):
        losses = (
            tg.groupby("family", sort=False)["squared_error_mean_across_seeds"]
            .mean()
            .sort_values()
        )
        if losses.empty:
            continue
        reference = str(losses.index[0])
        ref = tg[tg["family"] == reference][
            ["sample_index", "squared_error_mean_across_seeds"]
        ].rename(columns={"squared_error_mean_across_seeds": "ref_se"})
        ref_rmsep = float(np.sqrt(losses.iloc[0]))

        for i, family in enumerate([f for f in MODEL_ORDER if f in set(tg["family"])]):
            comp = tg[tg["family"] == family][
                ["sample_index", "squared_error_mean_across_seeds"]
            ].rename(columns={"squared_error_mean_across_seeds": "comp_se"})
            pair = ref.merge(comp, on="sample_index", how="inner")
            comp_msep = float(pair["comp_se"].mean()) if not pair.empty else np.nan
            if family == reference:
                p_one, p_two = 1.0, 1.0
            else:
                p_one, p_two = random_sign_pvalues(
                    pair["comp_se"].values - pair["ref_se"].values,
                    trials=trials,
                    seed=seed + i + 1000 * TASK_ORDER.index(task),
                )
            rows.append(
                {
                    "task": task,
                    "reference_family": reference,
                    "comparator_family": family,
                    "n_test": int(len(pair)),
                    "reference_rmsep": ref_rmsep,
                    "comparator_rmsep": float(np.sqrt(comp_msep)) if np.isfinite(comp_msep) else np.nan,
                    "delta_msep_comparator_minus_reference": (
                        float(comp_msep - pair["ref_se"].mean()) if not pair.empty else np.nan
                    ),
                    "p_one_sided_comparator_worse": p_one,
                    "p_two_sided": p_two,
                    "significantly_worse_at_0_05": bool(p_one < 0.05) if np.isfinite(p_one) else False,
                    "loss_basis": "mean squared prediction error per sample; DL averaged across final seeds",
                }
            )
    return reorder_tasks(pd.DataFrame(rows))


def learning_curve_sizes(n_dev: int, n_points: int) -> List[int]:
    minimum = min(n_dev, max(20, int(math.ceil(0.20 * n_dev))))
    if n_points <= 1 or minimum >= n_dev:
        return [n_dev]
    sizes = np.linspace(minimum, n_dev, n_points).round().astype(int)
    return sorted(set(int(x) for x in sizes))


def fit_predict_direct_for_cv_curve(
    family: str,
    cand: Dict[str, Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    task: str,
    progress: FitProgress,
) -> Tuple[np.ndarray, np.ndarray]:
    Xtr, [Xva], _ = preprocess_fit_apply(X_train_raw, [X_val_raw], cfg.preprocess)
    if not is_dl_family(family):
        model = fit_classical_model(family, cand, Xtr, y_train)
        tr_pred = np.asarray(model.predict(Xtr)).ravel()
        va_pred = np.asarray(model.predict(Xva)).ravel()
        progress.step(f"{task} | LC_CV | {family} | seed={seed}")
        return tr_pred.astype(float), va_pred.astype(float)

    set_seed(seed)
    safe_clear_tf()
    ys = fit_y_scaler(y_train)
    ytr_s = ys.transform(y_train.reshape(-1, 1)).ravel()
    yva_s = ys.transform(y_val.reshape(-1, 1)).ravel()
    if family == "ANN":
        model = build_mlp(Xtr.shape[1], cand["units"], MODEL_REGISTRY["ANN"]["dropout"])
        Xtr_i, Xva_i = Xtr, Xva
    else:
        model = build_cnn1d(
            Xtr.shape[1], cand["filters"], cand["kernel"],
            MODEL_REGISTRY["CNN1D"]["dropout"], MODEL_REGISTRY["CNN1D"]["use_layernorm"]
        )
        Xtr_i, Xva_i = Xtr[..., None], Xva[..., None]
    model.fit(
        Xtr_i,
        ytr_s,
        validation_data=(Xva_i, yva_s),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=build_callbacks(cfg.patience),
        verbose=0,
    )
    tr_s = model.predict(Xtr_i, verbose=0).ravel()
    va_s = model.predict(Xva_i, verbose=0).ravel()
    tr_pred = ys.inverse_transform(tr_s.reshape(-1, 1)).ravel()
    va_pred = ys.inverse_transform(va_s.reshape(-1, 1)).ravel()
    progress.step(f"{task} | LC_CV | {family} | seed={seed}")
    return tr_pred.astype(float), va_pred.astype(float)


def run_test_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    best_map: Dict[str, Dict[str, Any]],
    cfg: ExperimentConfig,
    task: str,
    progress: FitProgress,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.run_learning_curves:
        return pd.DataFrame(), pd.DataFrame()
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    sizes = learning_curve_sizes(len(dev_idx), cfg.learning_curve_points)
    rows: List[Dict[str, Any]] = []

    families = [f for f in cfg.learning_curve_families if f in best_map and f in active_families(cfg)]
    for family in families:
        cand = best_map[family]
        for repeat in range(cfg.learning_curve_repeats):
            rng = np.random.default_rng(cfg.learning_curve_seed + repeat + 101 * TASK_ORDER.index(task))
            order = rng.permutation(len(dev_idx))
            for point, train_n in enumerate(sizes, 1):
                sub = order[:train_n]
                seed = cfg.learning_curve_seed + 1000 * repeat + point
                metrics, _ = evaluate_selected_model(
                    family,
                    cand,
                    X_dev[sub],
                    y_dev[sub],
                    X_test,
                    y_test,
                    test_idx,
                    cfg,
                    task,
                    stage="LC_LOCKED_TEST",
                    progress=progress,
                    seed_override=[seed] if is_dl_family(family) else None,
                )
                rows.append(
                    {
                        "task": task,
                        "family": family,
                        "point": point,
                        "train_n": int(train_n),
                        "train_fraction_of_development": float(train_n / len(dev_idx)),
                        "repeat": repeat,
                        "test_rmse": float(metrics["rmse"].mean()),
                        "test_r2": float(metrics["r2"].mean()),
                        "fixed_best_config": str(cand),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()
    agg = (
        raw.groupby(["task", "family", "point", "train_n", "train_fraction_of_development"], as_index=False)
        .agg(test_rmse_mean=("test_rmse", "mean"), test_rmse_std=("test_rmse", "std"),
             test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"))
    )
    agg[["test_rmse_std", "test_r2_std"]] = agg[["test_rmse_std", "test_r2_std"]].fillna(0.0)
    return raw, reorder_models(reorder_tasks(agg))


def run_cv_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    dev_idx: np.ndarray,
    best_map: Dict[str, Dict[str, Any]],
    cfg: ExperimentConfig,
    task: str,
    progress: FitProgress,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Supplementary train/CV diagnostic restricted to the development data."""
    if not cfg.run_learning_curves:
        return pd.DataFrame(), pd.DataFrame()
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    sizes = learning_curve_sizes(len(dev_idx), cfg.learning_curve_points)
    rows: List[Dict[str, Any]] = []
    families = [f for f in cfg.learning_curve_families if f in best_map and f in active_families(cfg)]

    for family in families:
        cand = best_map[family]
        for repeat in range(cfg.learning_curve_repeats):
            rng = np.random.default_rng(cfg.learning_curve_seed + 500 + repeat + 101 * TASK_ORDER.index(task))
            order = rng.permutation(len(dev_idx))
            for point, train_n in enumerate(sizes, 1):
                sub = order[:train_n]
                X_sub, y_sub = X_dev[sub], y_dev[sub]
                splits = kfold_splits(len(sub), cfg.learning_curve_cv_folds, cfg.learning_curve_seed + point + repeat)
                if not candidate_valid_for_splits(family, cand, splits, X_sub.shape[1]):
                    continue
                for fold, (tr, va) in enumerate(splits, 1):
                    seed = cfg.learning_curve_seed + 1000 * repeat + 100 * point + fold
                    tr_pred, va_pred = fit_predict_direct_for_cv_curve(
                        family, cand, X_sub[tr], y_sub[tr], X_sub[va], y_sub[va],
                        cfg, seed, task, progress
                    )
                    rows.append(
                        {
                            "task": task,
                            "family": family,
                            "point": point,
                            "train_n": int(train_n),
                            "repeat": repeat,
                            "fold": fold,
                            "train_rmse": rmse(y_sub[tr], tr_pred),
                            "cv_rmse": rmse(y_sub[va], va_pred),
                            "fixed_best_config": str(cand),
                        }
                    )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()
    agg = (
        raw.groupby(["task", "family", "point", "train_n"], as_index=False)
        .agg(train_rmse_mean=("train_rmse", "mean"), train_rmse_std=("train_rmse", "std"),
             cv_rmse_mean=("cv_rmse", "mean"), cv_rmse_std=("cv_rmse", "std"))
    )
    agg[["train_rmse_std", "cv_rmse_std"]] = agg[["train_rmse_std", "cv_rmse_std"]].fillna(0.0)
    return raw, reorder_models(reorder_tasks(agg))


def numerical_rank_summary(summary: pd.DataFrame, metric_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    rank_rows: List[pd.DataFrame] = []
    for task, g in summary.groupby("task", sort=False):
        t = g[["task", "family", metric_col]].copy()
        t["rank"] = t[metric_col].rank(method="min", ascending=True)
        rank_rows.append(t)
    ranks = reorder_models(reorder_tasks(pd.concat(rank_rows, ignore_index=True)))
    counts = (
        ranks.groupby("family", as_index=False)
        .agg(
            numerical_wins=("rank", lambda s: int(np.sum(np.asarray(s) == 1))),
            top2_count=("rank", lambda s: int(np.sum(np.asarray(s) <= 2))),
            tasks=("task", "nunique"),
        )
    )
    counts = reorder_models(counts)
    return ranks, counts


def uncertainty_sig_digits(sd: float) -> int:
    if not np.isfinite(sd) or sd <= 0:
        return 2
    exponent = math.floor(math.log10(abs(sd)))
    scaled = abs(sd) / (10 ** exponent)
    first = int(scaled)
    second = int((scaled - first) * 10 + 1e-10)
    if first == 1 or (first == 2 and second < 5):
        return 2
    return 1


def format_mean_sd(mean: float, sd: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(sd) or sd <= 0:
        return format_sig(mean, 3)
    sig = uncertainty_sig_digits(sd)
    exponent = math.floor(math.log10(abs(sd)))
    decimal_place = exponent - sig + 1
    rounded_sd = round(sd, -decimal_place)
    rounded_mean = round(mean, -decimal_place)
    if abs(rounded_mean) >= 1e4 or (0 < abs(rounded_mean) < 1e-3) or abs(rounded_sd) < 1e-3:
        decimals = max(0, sig - 1)
        return f"{rounded_mean:.{decimals}e} +/- {rounded_sd:.{decimals}e}"
    decimals = max(0, -decimal_place)
    return f"{rounded_mean:.{decimals}f} +/- {rounded_sd:.{decimals}f}"


def format_sig(value: float, sig: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    decimals = sig - exponent - 1
    if exponent >= sig + 1 or exponent <= -4:
        return f"{value:.{sig - 1}e}"
    return f"{value:.{max(0, decimals)}f}"


def pivot_formatted_nested(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        g = summary[summary["task"] == task]
        if g.empty:
            continue
        row: Dict[str, Any] = {"Task": task}
        for family in MODEL_ORDER:
            h = g[g["family"] == family]
            if h.empty:
                row[family] = "NA"
            elif value == "rmse":
                row[family] = format_mean_sd(float(h["rmse_mean"].iloc[0]), float(h["rmse_std"].iloc[0]))
            else:
                row[family] = format_mean_sd(float(h["r2_mean"].iloc[0]), float(h["r2_std"].iloc[0]))
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_formatted_holdout(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        g = summary[summary["task"] == task]
        if g.empty:
            continue
        row: Dict[str, Any] = {"Task": task}
        for family in MODEL_ORDER:
            h = g[g["family"] == family]
            if h.empty:
                row[family] = "NA"
            elif value == "rmsep":
                # DL dispersion reflects repeated stochastic fits; deterministic models have SD=0.
                row[family] = format_mean_sd(float(h["rmsep_pooled"].iloc[0]), float(h["rmsep_seed_sd"].iloc[0]))
            else:
                row[family] = format_mean_sd(float(h["r2_mean_prediction"].iloc[0]), float(h["r2_seed_sd"].iloc[0]))
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_overview_table(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    out = audit[["task", "N", "p", "N_over_p"]].copy()
    out.columns = ["Task", "N", "p", "N/p"]
    out["N/p"] = out["N/p"].map(lambda x: f"{float(x):.3f}")
    return reorder_tasks(out.rename(columns={"Task": "task"})).rename(columns={"task": "Task"})


def latex_escape(value: Any) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(latex_escape)
    body = safe.to_latex(index=False, escape=False)
    body = body.replace("+/-", r"$\pm$")
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        + body
        + "\\end{table*}\n"
    )


def build_latex_bundle(
    audit: pd.DataFrame,
    nested_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    holdout_rank_counts: pd.DataFrame,
    vdvt: pd.DataFrame,
) -> str:
    parts: List[str] = []
    parts.append("% Automatically generated quantitative tables.\n")
    parts.append(dataframe_to_latex(
        dataset_overview_table(audit),
        "Overview of the regression tasks included in the spectroscopy benchmark.",
        "tab:datasets_generated",
    ))
    parts.append(dataframe_to_latex(
        pivot_formatted_nested(nested_summary, "rmse"),
        "Nested cross-validation outer-test RMSE (mean $\\pm$ SD across outer folds).",
        "tab:nested_rmse_generated",
    ))
    parts.append(dataframe_to_latex(
        pivot_formatted_nested(nested_summary, "r2"),
        "Nested cross-validation outer-test $R^2$ (mean $\\pm$ SD across outer folds).",
        "tab:nested_r2_generated",
    ))
    parts.append(dataframe_to_latex(
        pivot_formatted_holdout(holdout_summary, "rmsep"),
        "Locked independent-test RMSEP. For stochastic deep-learning models, the accompanying dispersion is across repeated final fits.",
        "tab:holdout_rmsep_generated",
    ))
    parts.append(dataframe_to_latex(
        pivot_formatted_holdout(holdout_summary, "r2"),
        "Locked independent-test $R^2$. For stochastic deep-learning models, the accompanying dispersion is across repeated final fits.",
        "tab:holdout_r2_generated",
    ))
    if not holdout_rank_counts.empty:
        rank_table = holdout_rank_counts.rename(columns={
            "family": "Model family", "numerical_wins": "Numerical wins",
            "top2_count": "Top-2 counts", "tasks": "Tasks"
        })
        parts.append(dataframe_to_latex(
            rank_table,
            "Numerical performance summary across tasks using locked independent-test RMSEP ranks.",
            "tab:holdout_ranks_generated",
        ))
    if not vdvt.empty:
        stat = vdvt[[
            "task", "reference_family", "comparator_family", "reference_rmsep",
            "comparator_rmsep", "p_one_sided_comparator_worse", "significantly_worse_at_0_05"
        ]].copy()
        stat.columns = ["Task", "Reference", "Comparator", "Reference RMSEP", "Comparator RMSEP", "p", "Significantly worse"]
        for col in ["Reference RMSEP", "Comparator RMSEP", "p"]:
            stat[col] = stat[col].map(lambda x: format_sig(float(x), 3) if np.isfinite(float(x)) else "NA")
        parts.append(dataframe_to_latex(
            stat,
            "Paired randomization comparisons of locked-test squared prediction errors. The reference is the numerically lowest-MSEP model within each task.",
            "tab:van_der_voet_generated",
        ))
    return "\n\n".join(parts)


def save_figure(fig, png_path: Path, pdf_path: Optional[Path] = None) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_task_benchmark(task: str, nested_summary: pd.DataFrame, out_dir: Path) -> None:
    g = nested_summary[nested_summary["task"] == task].copy()
    if g.empty:
        return
    g["family"] = pd.Categorical(g["family"], MODEL_ORDER, ordered=True)
    g = g.sort_values("family")
    for metric, err, ylabel, suffix in [
        ("rmse_mean", "rmse_std", "Outer-test RMSE", "rmse"),
        ("r2_mean", "r2_std", r"Outer-test $R^2$", "r2"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(g["family"].astype(str), g[metric], yerr=g[err], capsize=3)
        ax.set_title(task)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model family")
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, out_dir / f"{safe_slug(task)}_{suffix}.png")


def plot_global_complexity(candidate_log: pd.DataFrame, figures_dir: Path) -> None:
    if candidate_log.empty:
        return
    g = candidate_log[
        (candidate_log["family"].isin(["ANN", "CNN1D"]))
        & (candidate_log["status"] == "evaluated")
        & candidate_log["approx_flops"].notna()
    ].copy()
    if g.empty:
        return
    agg = (
        g.groupby(["task", "family", "candidate", "param_count", "approx_flops"], as_index=False)
        .agg(inner_cv_rmse_mean=("inner_cv_rmse", "mean"), inner_cv_rmse_std=("inner_cv_rmse", "std"))
    )
    agg["inner_cv_rmse_std"] = agg["inner_cv_rmse_std"].fillna(0.0)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    handles = []
    labels = []
    for ax, task in zip(axes.flat, TASK_ORDER):
        tg = agg[agg["task"] == task]
        for family in ["ANN", "CNN1D"]:
            fg = tg[tg["family"] == family].sort_values("approx_flops")
            if fg.empty:
                continue
            h = ax.errorbar(
                fg["approx_flops"], fg["inner_cv_rmse_mean"], yerr=fg["inner_cv_rmse_std"],
                marker="o", linewidth=1.2, capsize=2, label=family
            )
            if family not in labels:
                handles.append(h); labels.append(family)
        ax.set_xscale("log")
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("Approx. FLOPs / forward pass")
        ax.set_ylabel("Inner-CV RMSE")
        ax.grid(alpha=0.2)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Deep-learning complexity versus validation error", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(
        fig,
        figures_dir / "Figure1_complexity_panels.png",
        figures_dir / "Figure1_complexity_panels.pdf",
    )


def plot_global_test_learning_curve(lc_agg: pd.DataFrame, figures_dir: Path) -> None:
    if lc_agg.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    handles, labels = [], []
    for ax, task in zip(axes.flat, TASK_ORDER):
        tg = lc_agg[lc_agg["task"] == task]
        for family in ["PLSR", "ANN", "CNN1D"]:
            fg = tg[tg["family"] == family].sort_values("train_n")
            if fg.empty:
                continue
            h = ax.errorbar(
                fg["train_n"], fg["test_rmse_mean"], yerr=fg["test_rmse_std"],
                marker="o", linewidth=1.2, capsize=2, label=family
            )
            if family not in labels:
                handles.append(h); labels.append(family)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("Training samples from development set")
        ax.set_ylabel("Locked-test RMSE")
        ax.grid(alpha=0.2)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Independent-test learning curves", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(
        fig,
        figures_dir / "Figure2_independent_test_learning_curves.png",
        figures_dir / "Figure2_independent_test_learning_curves.pdf",
    )


def plot_global_cv_learning_curve(lc_agg: pd.DataFrame, figures_dir: Path) -> None:
    if lc_agg.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    family_handles: Dict[str, Any] = {}
    for ax, task in zip(axes.flat, TASK_ORDER):
        tg = lc_agg[lc_agg["task"] == task]
        for family in ["PLSR", "ANN", "CNN1D"]:
            fg = tg[tg["family"] == family].sort_values("train_n")
            if fg.empty:
                continue
            line, = ax.plot(fg["train_n"], fg["cv_rmse_mean"], marker="o", linewidth=1.2, label=family)
            ax.plot(fg["train_n"], fg["train_rmse_mean"], linestyle="--", linewidth=1.0, color=line.get_color())
            family_handles.setdefault(family, line)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("Training subset size")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.2)
    handles = list(family_handles.values())
    labels = list(family_handles.keys())
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Development-set learning curves: solid = CV, dashed = training", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(
        fig,
        figures_dir / "FigureS1_cv_learning_curves.png",
        figures_dir / "FigureS1_cv_learning_curves.pdf",
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_excel_workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def run_pipeline(cfg: ExperimentConfig, repo_root: Path) -> Path:
    if cfg.smoke_test:
        apply_smoke_profile(cfg)
    families = active_families(cfg)
    out_root = repo_root / cfg.output_root
    raw_dir = out_root / "raw"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    latex_dir = out_root / "latex"
    for d in [raw_dir, tables_dir, figures_dir, latex_dir]:
        d.mkdir(parents=True, exist_ok=True)

    progress = FitProgress()
    audit_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    nested_raw_frames: List[pd.DataFrame] = []
    nested_pred_frames: List[pd.DataFrame] = []
    nested_candidate_frames: List[pd.DataFrame] = []
    holdout_summary_frames: List[pd.DataFrame] = []
    holdout_seed_frames: List[pd.DataFrame] = []
    holdout_pred_frames: List[pd.DataFrame] = []
    holdout_pred_agg_frames: List[pd.DataFrame] = []
    holdout_candidate_frames: List[pd.DataFrame] = []
    lc_test_raw_frames: List[pd.DataFrame] = []
    lc_test_agg_frames: List[pd.DataFrame] = []
    lc_cv_raw_frames: List[pd.DataFrame] = []
    lc_cv_agg_frames: List[pd.DataFrame] = []

    for dataset_number, dcfg in enumerate(manuscript_datasets(), 1):
        print(f"\n=== [{dataset_number}/{len(TASK_ORDER)}] {dcfg.task} ===")
        if cfg.synthetic:
            X, y = synthetic_dataset(dcfg, seed=10000 + dataset_number)
        else:
            X, y = load_dataset(dcfg, repo_root)
        audit_rows.append(audit_dataset(X, y, dcfg, synthetic=cfg.synthetic))

        nested_raw, nested_pred, nested_candidates = run_nested_cv(X, y, cfg, dcfg.task, progress)
        nested_raw_frames.append(nested_raw)
        nested_pred_frames.append(nested_pred)
        nested_candidate_frames.append(nested_candidates)

        (
            holdout_summary,
            holdout_seed,
            holdout_pred,
            holdout_pred_agg,
            best_map,
            dev_idx,
            test_idx,
            holdout_candidates,
        ) = run_independent_test(X, y, cfg, dcfg.task, progress)
        holdout_summary_frames.append(holdout_summary)
        holdout_seed_frames.append(holdout_seed)
        holdout_pred_frames.append(holdout_pred)
        holdout_pred_agg_frames.append(holdout_pred_agg)
        holdout_candidate_frames.append(holdout_candidates)

        split_rows.extend(
            {"task": dcfg.task, "sample_index": int(i), "split": "development"} for i in dev_idx
        )
        split_rows.extend(
            {"task": dcfg.task, "sample_index": int(i), "split": "locked_test"} for i in test_idx
        )

        if cfg.run_learning_curves:
            lc_raw, lc_agg = run_test_learning_curve(
                X, y, dev_idx, test_idx, best_map, cfg, dcfg.task, progress
            )
            lc_test_raw_frames.append(lc_raw)
            lc_test_agg_frames.append(lc_agg)
            cv_raw, cv_agg = run_cv_learning_curve(
                X, y, dev_idx, best_map, cfg, dcfg.task, progress
            )
            lc_cv_raw_frames.append(cv_raw)
            lc_cv_agg_frames.append(cv_agg)

        safe_clear_tf()
        gc.collect()

    audit_df = reorder_tasks(pd.DataFrame(audit_rows))
    split_df = reorder_tasks(pd.DataFrame(split_rows))
    nested_raw_df = pd.concat(nested_raw_frames, ignore_index=True) if nested_raw_frames else pd.DataFrame()
    nested_predictions_df = pd.concat(nested_pred_frames, ignore_index=True) if nested_pred_frames else pd.DataFrame()
    nested_candidates_df = pd.concat(nested_candidate_frames, ignore_index=True) if nested_candidate_frames else pd.DataFrame()
    holdout_summary_df = reorder_models(reorder_tasks(pd.concat(holdout_summary_frames, ignore_index=True))) if holdout_summary_frames else pd.DataFrame()
    holdout_seed_df = pd.concat(holdout_seed_frames, ignore_index=True) if holdout_seed_frames else pd.DataFrame()
    holdout_predictions_df = pd.concat(holdout_pred_frames, ignore_index=True) if holdout_pred_frames else pd.DataFrame()
    holdout_pred_agg_df = pd.concat(holdout_pred_agg_frames, ignore_index=True) if holdout_pred_agg_frames else pd.DataFrame()
    holdout_candidates_df = pd.concat(holdout_candidate_frames, ignore_index=True) if holdout_candidate_frames else pd.DataFrame()
    lc_test_raw_df = pd.concat(lc_test_raw_frames, ignore_index=True) if lc_test_raw_frames else pd.DataFrame()
    lc_test_agg_df = pd.concat(lc_test_agg_frames, ignore_index=True) if lc_test_agg_frames else pd.DataFrame()
    lc_cv_raw_df = pd.concat(lc_cv_raw_frames, ignore_index=True) if lc_cv_raw_frames else pd.DataFrame()
    lc_cv_agg_df = pd.concat(lc_cv_agg_frames, ignore_index=True) if lc_cv_agg_frames else pd.DataFrame()

    nested_summary_df = summarize_nested_cv(nested_raw_df)
    vdvt_df = van_der_voet_table(
        holdout_pred_agg_df,
        trials=cfg.randomization_trials,
        seed=cfg.randomization_seed,
    )
    nested_ranks_df, nested_rank_counts_df = numerical_rank_summary(nested_summary_df, "rmse_mean")
    holdout_ranks_df, holdout_rank_counts_df = numerical_rank_summary(holdout_summary_df, "rmsep_pooled")

    complexity_df = pd.DataFrame()
    if not nested_candidates_df.empty:
        cg = nested_candidates_df[
            (nested_candidates_df["family"].isin(["ANN", "CNN1D"]))
            & (nested_candidates_df["status"] == "evaluated")
        ].copy()
        if not cg.empty:
            complexity_df = (
                cg.groupby(["task", "family", "candidate", "param_count", "approx_flops"], as_index=False)
                .agg(inner_cv_rmse_mean=("inner_cv_rmse", "mean"),
                     inner_cv_rmse_std=("inner_cv_rmse", "std"),
                     evaluations=("inner_cv_rmse", "count"))
            )
            complexity_df["inner_cv_rmse_std"] = complexity_df["inner_cv_rmse_std"].fillna(0.0)
            complexity_df = reorder_models(reorder_tasks(complexity_df))

    raw_outputs = {
        "dataset_audit.csv": audit_df,
        "split_manifest.csv": split_df,
        "nested_cv_raw.csv": nested_raw_df,
        "nested_cv_summary.csv": nested_summary_df,
        "nested_cv_predictions.csv": nested_predictions_df,
        "nested_cv_candidates.csv": nested_candidates_df,
        "independent_test_summary.csv": holdout_summary_df,
        "independent_test_seed_metrics.csv": holdout_seed_df,
        "independent_test_predictions.csv": holdout_predictions_df,
        "independent_test_predictions_aggregated.csv": holdout_pred_agg_df,
        "independent_test_candidates.csv": holdout_candidates_df,
        "van_der_voet_randomization.csv": vdvt_df,
        "nested_cv_task_ranks.csv": nested_ranks_df,
        "nested_cv_rank_summary.csv": nested_rank_counts_df,
        "independent_test_task_ranks.csv": holdout_ranks_df,
        "independent_test_rank_summary.csv": holdout_rank_counts_df,
        "learning_curve_independent_test_raw.csv": lc_test_raw_df,
        "learning_curve_independent_test_summary.csv": lc_test_agg_df,
        "learning_curve_cv_diagnostic_raw.csv": lc_cv_raw_df,
        "learning_curve_cv_diagnostic_summary.csv": lc_cv_agg_df,
        "dl_complexity.csv": complexity_df,
        "run_log.csv": progress.to_dataframe(),
    }
    for name, df in raw_outputs.items():
        save_csv(df, raw_dir / name)

    save_excel_workbook(
        tables_dir / "revision_tables.xlsx",
        {
            "Dataset_Audit": audit_df,
            "Nested_CV_Summary": nested_summary_df,
            "Independent_Test": holdout_summary_df,
            "Van_der_Voet": vdvt_df,
            "Nested_Ranks": nested_rank_counts_df,
            "Independent_Ranks": holdout_rank_counts_df,
            "LC_Independent_Test": lc_test_agg_df,
            "LC_CV_Diagnostic": lc_cv_agg_df,
            "DL_Complexity": complexity_df,
        },
    )

    latex_text = build_latex_bundle(
        audit_df, nested_summary_df, holdout_summary_df, holdout_rank_counts_df, vdvt_df
    )
    (latex_dir / "latex_tables.txt").write_text(latex_text, encoding="utf-8")

    for task in TASK_ORDER:
        plot_task_benchmark(task, nested_summary_df, figures_dir)
    plot_global_complexity(nested_candidates_df, figures_dir)
    plot_global_test_learning_curve(lc_test_agg_df, figures_dir)
    plot_global_cv_learning_curve(lc_cv_agg_df, figures_dir)

    run_config = asdict(cfg)
    run_config.update(
        {
            "tensorflow_available": TF_AVAILABLE,
            "active_families": families,
            "dataset_tasks": TASK_ORDER,
            "data_configs": [asdict(x) for x in manuscript_datasets()],
        }
    )
    save_json(run_config, out_root / "run_config.json")
    print(f"\nCompleted. Outputs written to: {out_root}")
    return out_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Food NIR spectroscopy regression benchmark")
    p.add_argument("--synthetic", action="store_true", help="Use generated spectroscopy-like data instead of local data files.")
    p.add_argument("--smoke-test", action="store_true", help="Use reduced grids/epochs for a fast end-to-end validation.")
    p.add_argument("--disable-dl", action="store_true", help="Run only PLSR, Ridge, and SVR.")
    p.add_argument("--skip-learning-curves", action="store_true", help="Skip learning-curve calculations and figures.")
    p.add_argument("--holdout-fraction", type=float, default=0.20, help="Fraction reserved as the locked independent test set.")
    p.add_argument("--holdout-seed", type=int, default=2026, help="Random seed for the locked independent split.")
    p.add_argument("--randomization-trials", type=int, default=9999, help="Monte Carlo trials for paired randomization tests.")
    p.add_argument("--output-root", type=str, default="outputs/revision_results", help="Output directory relative to repository root.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.05 <= args.holdout_fraction <= 0.50):
        raise ValueError("--holdout-fraction must be between 0.05 and 0.50.")
    cfg = ExperimentConfig(
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
        randomization_trials=args.randomization_trials,
        output_root=args.output_root,
        disable_dl=args.disable_dl,
        synthetic=args.synthetic,
        smoke_test=args.smoke_test,
        run_learning_curves=not args.skip_learning_curves,
    )
    repo_root = Path(__file__).resolve().parent
    run_pipeline(cfg, repo_root)


if __name__ == "__main__":
    main()
