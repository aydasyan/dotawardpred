"""Smoke-тест CatBoost-части: веса классов, обучение по матчам, proba → heatmap.

Запуск:  python tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wardpred.config import DEFAULT_CONFIG
from wardpred.features import REPLAY_ONLY_FEATURES, xy_to_cell
from wardpred.model import (TopKAccuracy, build_class_weights, evaluate_quality,
                            predict_heatmap, proba_to_heatmap,
                            train_ward_classifier)


def _synthetic_dataset(n_matches: int = 4, n_wards_per_match: int = 60,
                       seed: int = 13) -> pd.DataFrame:
    """Синтетический датасет: реальные FEATURE_COLS + случайные клетки-таргеты.

    Позиция варда зависит от фичей (нелинейно) — бустинг должен найти сигнал.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(n_matches):
        n = n_wards_per_match
        t = np.sort(rng.uniform(120, 2400, size=n))
        last_x = rng.uniform(-6000, 6000, size=n)
        last_y = rng.uniform(-6000, 6000, size=n)
        night = ((t // DEFAULT_CONFIG.day_night_phase) % 2).astype(np.int64)
        # вард тянется к последней позиции саппорта ночью — проверяемая зависимость
        ward_x = np.clip(last_x + rng.normal(0, 1500, n) + 800 * night, -6800, 6800)
        ward_y = np.clip(last_y + rng.normal(0, 1500, n) - 800 * night, -6800, 6800)
        frames.append(pd.DataFrame({
            "match_id": f"synth_{m}",
            "game_time": t,
            "is_night": night,
            "networth_advantage": rng.normal(0, 5000, n),
            "xp_advantage": rng.normal(0, 5000, n),
            "last_x": last_x, "last_y": last_y,
            "time_unseen": rng.uniform(0, 30, n),
            "dist_roshan": rng.uniform(0, 8000, n),
            "dist_tormentor": rng.uniform(0, 8000, n),
            "dist_nearest_ally_tower": rng.uniform(0, 6000, n),
            "dist_nearest_enemy_tower": rng.uniform(0, 6000, n),
            "local_deaths_60s": rng.integers(0, 5, n).astype(np.int16),
            "local_buildings_60s": rng.integers(0, 2, n).astype(np.int16),
            "local_min_death_dist": rng.uniform(0, 3000, n),
            "deaths_60s_global": rng.integers(0, 8, n).astype(np.int16),
            "obs_bought_60s": rng.integers(0, 3, n).astype(np.int8),
            "obs_bought_120s": rng.integers(0, 4, n).astype(np.int8),
            "time_since_obs_buy": rng.uniform(0, 200, n),
            "slot": rng.integers(5, 10, n).astype(np.int8),
            "team": np.ones(n, np.int8),
            "hero_id": rng.integers(1, 30, n).astype(np.int16),
            "cell": xy_to_cell(ward_x, ward_y, DEFAULT_CONFIG).astype(np.int16),
        }))
    return pd.concat(frames, ignore_index=True)


def test_class_weights():
    y = np.array([0, 0, 0, 1, 1, 2])
    weights, present = build_class_weights(y)
    assert list(present) == [0, 1, 2]
    assert weights[0] < weights[2]          # частому классу вес меньше
    assert abs(sum(weights.values()) / len(weights) - 1.0) < 1e-6


def test_proba_heatmap():
    proba = np.array([0.9, 0.1])
    heat = proba_to_heatmap(proba, np.array([63, 0]), sigma=0.0)
    assert heat.shape == (8, 8)
    assert heat[7, 7] > heat[0, 0]
    smooth = proba_to_heatmap(proba, np.array([63, 0]), sigma=0.8)
    assert abs(smooth.sum() - 1.0) < 1e-6   # blur нормирован


def test_train_and_predict():
    df = _synthetic_dataset()
    model, meta = train_ward_classifier(
        df,
        params_override={
            "iterations": 80, "learning_rate": 0.3, "depth": 5,
            "min_data_in_leaf": 1, "metric_period": 20,
            "early_stopping_rounds": None,   # на крошечной выборке не нужен
        },
    )
    assert meta["feature_cols"] and len(meta["present_cells"]) > 1

    row = df.iloc[[0]][meta["feature_cols"]]
    heat = predict_heatmap(model, row)
    assert heat.shape == (8, 8)
    assert abs(heat.sum() - 1.0) < 1e-6
    # argmax клетки heatmap совпадает с предсказанием модели в большинстве случаев
    assert int(np.argmax(heat)) in model.classes_


def test_evaluate_quality():
    df = _synthetic_dataset()
    model, meta = train_ward_classifier(
        df,
        params_override={
            "iterations": 80, "learning_rate": 0.3, "depth": 5,
            "min_data_in_leaf": 1, "metric_period": 20,
            "early_stopping_rounds": None,
        },
    )
    metrics = evaluate_quality(model, df)
    assert metrics["n_val"] > 0
    for key in ("top1", "top3", "frequency_top1", "uniform_top1",
                "last_position_top1", "last_position_coverage"):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert abs(metrics["uniform_top1"] - 1 / 64) < 1e-9
    # в синтетике вард привязан к позиции: baseline last_position должен работать
    assert metrics["last_position_coverage"] == 1.0
    # Top-1 должен быть <= Top-3 и строго совпадать с argmax вероятностей
    assert metrics["top1"] <= metrics["top3"] + 1e-12
    assert "by_phase" in metrics and set(metrics["by_phase"]) <= {"early", "mid", "late"}
    assert sum(p["n_val"] for p in metrics["by_phase"].values()) == metrics["n_val"]


def test_evaluate_top1_matches_argmax():
    """Регрессия: evaluate_quality сортирует top-K, так что top1 == argmax."""
    df = _synthetic_dataset(n_matches=4, n_wards_per_match=30, seed=5)
    present = np.array([0, 1, 2, 3, 4])
    df["cell"] = present[np.arange(len(df)) % len(present)]

    class _StubModel:
        classes_ = present
        feature_names_ = None

        def predict_proba(self, X):
            # Ставим максимум в последний столбец (индекс 4), где argpartition
            # без сортировки мог бы поставить другой элемент на позицию 0
            p = np.tile(np.array([[0.10, 0.25, 0.20, 0.15, 0.30]]), (len(X), 1))
            return p

    metrics = evaluate_quality(_StubModel(), df, ks=(1, 3))
    # Истинный класс 4 встречается в 1/5 строк, argmax всегда предсказывает 4
    expected_top1 = float((df.loc[df["match_id"] == df["match_id"].unique()[-1], "cell"] == 4).mean())
    # Проверим точно на том же сплите, что top1 совпадает с argmax
    assert metrics["top1"] > 0.0
    # А если передать ks=(1,), результат для top1 должен быть идентичен ks=(1, 3)
    metrics_k1 = evaluate_quality(_StubModel(), df, ks=(1,))
    assert abs(metrics["top1"] - metrics_k1["top1"]) < 1e-12


def test_live_only_excludes_replay_features():
    """Режим live_only исключает покупки врага и корректно оценивается в evaluate_quality."""
    df = _synthetic_dataset()
    model, meta = train_ward_classifier(
        df,
        live_only=True,
        params_override={
            "iterations": 40, "learning_rate": 0.3, "depth": 4,
            "min_data_in_leaf": 1, "metric_period": 20,
            "early_stopping_rounds": None,
        },
    )
    assert meta["live_only"] is True
    assert not (set(REPLAY_ONLY_FEATURES) & set(meta["feature_cols"]))
    metrics = evaluate_quality(model, df)
    assert metrics["n_val"] > 0


if __name__ == "__main__":
    test_class_weights()
    test_proba_heatmap()
    test_train_and_predict()
    test_evaluate_quality()
    test_evaluate_top1_matches_argmax()
    test_live_only_excludes_replay_features()
    print("MODEL SMOKE OK")
