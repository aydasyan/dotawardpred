"""CatBoost: 64-классовая классификация клеток карты + proba → тепловая карта 8x8.

Ключевые решения под жёсткий дисбаланс 64 классов:
  * loss_function=MultiClassOneVsAll — каждая клетка как отдельная бинарная
    задача, устойчивее к дисбалансу, чем softmax MultiClass;
  * class_weights = sqrt-балансировка с КАПОТИРОВАНИЕМ — клетки-пустыни не
    должны получать экстремальные веса и переобучаться на 1-2 примерах;
  * depth 6-7 + l2_leaf_reg + min_data_in_leaf — регуляризация против
    запоминания редких клеток;
  * метрика качества для UI — не MultiClass-лосс, а Top-K accuracy.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_COLS

GRID = 8

# hero_id — ключевая категория: у каждого героя свой стайл вардинга
CAT_FEATURES = ["slot", "team", "hero_id"]

BASE_PARAMS = dict(
    loss_function="MultiClassOneVsAll",
    eval_metric="MultiClassOneVsAll",  # должен совпадать с loss; TopKAccuracy(k=3) — тоже вариант
    iterations=4000,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=5.0,
    min_data_in_leaf=20,
    bootstrap_type="Bayesian",
    bagging_temperature=1.0,
    random_strength=1.0,
    leaf_estimation_iterations=10,
    random_seed=2026,
    thread_count=-1,
    allow_writing_files=False,
    metric_period=100,
)


class TopKAccuracy:
    """Кастомная eval-метрика CatBoost: доля снэпшотов, где истинная клетка в Top-K.

    Использование: BASE_PARAMS["eval_metric"] = TopKAccuracy(k=3).
    Для UI-задачи Top-3 информативнее лосса: карта покажет 3 «горячие» зоны.
    """

    def __init__(self, k: int = 3):
        self.k = k

    def is_max_optimal(self) -> bool:
        return True

    def get_final_error(self, error: float, weight: float) -> float:
        return error / (weight + 1e-12)

    def evaluate(self, approxes, target, weight):
        scores = np.vstack(list(approxes)).T            # (n_objects, n_classes)
        k = min(self.k, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        t = np.asarray(target).ravel()
        hits = (top == t[:, None]).any(axis=1).astype(np.float64)
        w = np.ones(len(t)) if weight is None else np.asarray(weight, dtype=np.float64)
        return float(np.dot(hits, w)), float(w.sum())


def build_class_weights(y: np.ndarray, n_cells: int = GRID * GRID,
                        cap: float = 25.0) -> Tuple[Dict[int, float], np.ndarray]:
    """sqrt-балансировка весов клеток с капотом + нормировка на среднее 1.

    Возвращает (веса для присутствующих классов, массив присутствующих клеток).
    Клетки без примеров нельзя отдавать в CatBoost — proba-вектор будет
    короче 64, маппинг нужен для перевода в тепловую карту.
    """
    counts = np.bincount(y, minlength=n_cells)
    present = np.nonzero(counts)[0]
    total = counts[present].sum()
    weights = np.sqrt(total / (len(present) * counts[present]))
    weights = np.minimum(weights, cap)
    weights = weights / weights.mean()
    return {int(cell): float(w) for cell, w in zip(present, weights)}, present


def train_ward_classifier(df: pd.DataFrame,
                          target_col: str = "cell",
                          group_col: str = "match_id",
                          val_fraction: float = 0.2,
                          seed: int = 2026,
                          params_override: Optional[dict] = None
                          ) -> Tuple[CatBoostClassifier, dict]:
    """Сплит ПО МАТЧАМ (не по строкам!) + обучение с весами классов."""
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[target_col].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=df[group_col]))
    y_train = y.iloc[train_idx]

    weights, present = build_class_weights(y_train.to_numpy())
    params = {**BASE_PARAMS, "class_weights": weights}
    if params_override:
        params.update(params_override)

    # классы, отсутствующие в train, из val убираем (CatBoost иначе упадёт)
    y_val = y.iloc[val_idx]
    unseen = ~y_val.isin(present)
    if unseen.any():
        import warnings
        warnings.warn(f"В val {int(unseen.sum())} снэпшотов с классами, "
                      f"которых нет в train — исключены из валидации")
        val_idx = val_idx[~unseen.to_numpy()]

    train_pool = Pool(X.iloc[train_idx], y_train, cat_features=cat_cols,
                      feature_names=feature_cols)
    val_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_cols,
                    feature_names=feature_cols)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool,
              early_stopping_rounds=200, verbose=params.get("metric_period", 100))

    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "present_cells": present.tolist(),
        "best_iteration": int(model.get_best_iteration()),
    }
    return model, meta


def evaluate_quality(model: CatBoostClassifier, df: pd.DataFrame,
                     target_col: str = "cell", group_col: str = "match_id",
                     val_fraction: float = 0.2, seed: int = 2026,
                     ks: Tuple[int, ...] = (1, 3)) -> dict:
    """Top-K accuracy на отложенных матчах + простые baselines (тот же сплит).

    Сплит повторяет train_ward_classifier (те же seed/val_fraction), поэтому
    val-множество совпадает с валидацией обучения. Baselines:
      * uniform — равномерный выбор из 64 клеток (теоретический уровень K/64);
      * frequency — предсказание самых частых клеток train-части;
      * last_position — клетка последней видимой позиции вардера (Top-1;
        считается только по снэпшотам с известной позицией).
    """
    from .features import xy_to_cell
    from .config import DEFAULT_CONFIG

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df[target_col].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=df[group_col]))
    y_train = y.iloc[train_idx]

    # те же ограничения, что при обучении: классы вне train из val убираем
    classes = np.asarray(model.classes_)
    y_val = y.iloc[val_idx]
    seen = y_val.isin(classes).to_numpy()
    X_val = X.iloc[val_idx][seen]
    y_val = y_val[seen]
    n = len(y_val)
    if n == 0:
        return {"n_val": 0}

    proba = model.predict_proba(X_val)                # (n, n_present)
    k_max = min(max(ks), proba.shape[1])
    top = np.argpartition(-proba, k_max - 1, axis=1)[:, :k_max]
    top_labels = classes[top]
    yt = y_val.to_numpy()

    metrics: Dict[str, float] = {"n_val": n}
    for k in ks:
        if k > k_max:
            continue
        metrics[f"top{k}"] = float((top_labels[:, :k] == yt[:, None]).any(axis=1).mean())
        metrics[f"uniform_top{k}"] = k / (GRID * GRID)

    counts = np.bincount(y_train.to_numpy(), minlength=GRID * GRID)
    freq_order = np.argsort(-counts)
    for k in ks:
        if k > k_max:
            continue
        metrics[f"frequency_top{k}"] = float(
            np.isin(yt, freq_order[:k]).mean())

    lx = X_val["last_x"].to_numpy(np.float64)
    ly = X_val["last_y"].to_numpy(np.float64)
    known = (lx != -1.0) & (ly != -1.0)
    if known.any():
        pred_cell = xy_to_cell(lx[known], ly[known], DEFAULT_CONFIG)
        metrics["last_position_top1"] = float((pred_cell == yt[known]).mean())
        metrics["last_position_coverage"] = float(known.mean())
    return metrics


def proba_to_heatmap(proba: np.ndarray, classes: np.ndarray,
                     sigma: float = 0.7) -> np.ndarray:
    """Вектор вероятностей (только присутствующие классы) → карта 8x8 с Gaussian blur.

    sigma=0.7 сглаживает соседние клетки; mode="constant" не заносит массу за край.
    """
    grid = np.zeros((GRID, GRID), dtype=np.float64)
    for cell, p in zip(classes, proba):
        row, col = divmod(int(cell), GRID)
        grid[row, col] = p
    heat = gaussian_filter(grid, sigma=sigma, mode="constant", cval=0.0)
    total = heat.sum()
    return heat / total if total > 0 else heat


def predict_heatmap(model: CatBoostClassifier, X: pd.DataFrame,
                    sigma: float = 0.7) -> np.ndarray:
    """X — один снэпшот (DataFrame 1×n_features) → сглаженная карта 8x8."""
    proba = model.predict_proba(X)
    return proba_to_heatmap(proba[0], np.asarray(model.classes_), sigma)


def save_meta(path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
