"""Smoke-тест всего пайплайна на синтетическом матче (без CatBoost — быстро).

Запуск:  python tests/test_smoke.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wardpred.config import DEFAULT_CONFIG
from wardpred.features import (FEATURE_COLS, LocalActivityWindow,
                               _prepare_slot_stream,
                               build_match_features, xy_to_cell)
from wardpred.loader import load_match


def _synthetic_match_file(path: Path, rng: np.random.Generator) -> None:
    rows = []
    # interval-срезы: 10 слотов каждые 2 секунды, 0..240 с
    for t in np.arange(0, 240, 2.0):
        for slot in range(10):
            rows.append({
                "type": "interval", "time": float(t), "slot": slot,
                "x": float(rng.uniform(-6000, 6000)),
                "y": float(rng.uniform(-6000, 6000)),
                "gold": 100 + slot, "xp": 1000 + t, "net_worth": 1000 + t + slot,
                "hero_id": slot + 1,
            })
    # покупки вардов слотом 7 (dire)
    rows.append({"type": "DOTA_COMBATLOG_PURCHASE", "time": 60.0, "slot": 7,
                 "valuename": "item_observer_ward"})
    rows.append({"type": "DOTA_COMBATLOG_PURCHASE", "time": 100.0, "slot": 7,
                 "valuename": "item_ward_dispenser"})
    # смерти (одна с координатами, одна без — проверка fill по потоку жертвы)
    rows.append({"type": "DOTA_COMBATLOG_DEATH", "time": 110.0, "slot": 2,
                 "x": 100.0, "y": 100.0})
    rows.append({"type": "DOTA_COMBATLOG_DEATH", "time": 115.0, "slot": 3})
    # разрушение башни (имя из конфига)
    rows.append({"type": "DOTA_COMBATLOG_TOWERKILL", "time": 118.0,
                 "targetname": "npc_dota_tower1_top_dire"})
    # дубликат obs — должен быть удалён дедупом
    rows.append({"type": "obs", "time": 120.0, "slot": 7, "x": 1000.0, "y": -500.0})
    rows.append({"type": "obs", "time": 120.4, "slot": 7, "x": 1000.3, "y": -500.2})
    rows.append({"type": "obs", "time": 200.0, "slot": 7, "x": -3000.0, "y": 2500.0})
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_loader_normalizes_and_dedups():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "match1.jsonl"
        _synthetic_match_file(path, np.random.default_rng(7))
        match = load_match(path)
        assert len(match.intervals) == 1200          # 120 срезов * 10 слотов
        assert set(match.purchases["item"]) <= DEFAULT_CONFIG.obs_items
        assert len(match.purchases) == 2
        assert len(match.obs) == 3                    # сырые obs; дедуп — в features
        # смерть с явными координатами сохранена (100 в сетке → (100-64)*128 world),
        # без координат — NaN (добор в features)
        assert match.deaths["x"].iloc[0] == (100 - 64) * 128.0
        assert np.isnan(match.deaths["x"].iloc[1])
        # башня из конфига получила координаты
        assert match.buildings["x"].notna().all()
        return match


def test_features_snapshot():
    match = test_loader_normalizes_and_dedups()
    df = build_match_features(match)
    assert len(df) == 2
    assert list(df.columns) == FEATURE_COLS + ["cell", "ward_x", "ward_y", "match_id"]
    # добор координат смерти по потоку жертвы работает
    from wardpred.features import _fill_death_coords
    assert _fill_death_coords(match.deaths, {3: _prepare_slot_stream(
        match.intervals, 3)})["x"].notna().all()

    row = df.iloc[0]
    assert 0 <= int(row["cell"]) <= 63
    assert int(row["is_night"]) in (0, 1)
    assert row["time_unseen"] == 0.0            # срез в t=120 есть → виден
    assert int(row["obs_bought_120s"]) == 2     # обе покупки в окне 120с
    assert int(row["obs_bought_60s"]) == 2      # окно [T-60, T] включительно: 60.0 входит
    assert abs(float(row["time_since_obs_buy"]) - 20.0) < 1e-3
    assert float(row["networth_advantage"]) == row["networth_advantage"]  # не NaN
    assert int(row["deaths_60s_global"]) == 2   # смерти в 110 и 115
    assert int(row["local_buildings_60s"]) in (0, 1)  # башня в 118, радиус решает
    assert not df[[*FEATURE_COLS]].isna().any().any()  # весь вектор без NaN
    return df


def test_cell_mapping():
    cfg = DEFAULT_CONFIG
    cells = xy_to_cell(np.array([cfg.map_min_x + 1, cfg.map_max_x - 1]),
                       np.array([cfg.map_min_y + 1, cfg.map_max_y - 1]), cfg)
    assert cells[0] == 0 and cells[1] == 63     # углы сетки


def test_local_window_boundary():
    """Граница окна ровно window_seconds: событие на T-60 остаётся, старше — нет.

    Регрессия на двойное вычитание окна (раньше окно фактически было ~120 с).
    """
    win = LocalActivityWindow(60.0)
    win.push(90.0, 0.0, 0.0, 0)    # старше окна (пуши идут по неубыванию времени)
    win.push(100.0, 0.0, 0.0, 0)   # смерть ровно на T-60 от момента проверки
    win.evict(160.0)
    deaths, _, _ = win.stats(0.0, 0.0, 1500.0)
    assert deaths == 1, "событие ровно на границе окна должно сохраняться"
    win.evict(160.1)
    deaths, _, _ = win.stats(0.0, 0.0, 1500.0)
    assert deaths == 0, "событие старше окна должно выселяться"


def test_hero_id_at_snapshot_time():
    """hero_id берётся с последнего видимого среза <= T, а не последнего в файле."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "match2.jsonl"
        rows = [
            {"type": "interval", "time": 100.0, "slot": 7, "x": 0, "y": 0,
             "hero_id": 7, "gold": 0, "xp": 0, "net_worth": 0},
            {"type": "interval", "time": 200.0, "slot": 7, "x": 0, "y": 0,
             "hero_id": 99, "gold": 0, "xp": 0, "net_worth": 0},
            {"type": "obs", "time": 150.0, "slot": 7, "x": 500.0, "y": 500.0},
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        df = build_match_features(load_match(path))
        assert int(df.iloc[0]["hero_id"]) == 7


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*флаг 'visible'.*")
    test_cell_mapping()
    test_local_window_boundary()
    test_hero_id_at_snapshot_time()
    df = test_features_snapshot()
    print("SMOKE OK:", len(df), "снэпшотов,", df['cell'].nunique(), "клеток")
