"""Потоковая загрузка jsonl-логов (стиль OpenDota/Clarity) в tidy-таблицы.

Принципы:
  * один проход по файлу, без накопления сырых dict'ов;
  * orjson, если установлен (парсинг jsonl — основная стоимость сборки датасета);
  * координаты нормализуются в world units один раз;
  * события боевого лога часто не содержат slot — сопоставляем по имени героя
    из interval-срезов (name_to_slot строится на лету, в порядке следования строк).

Паритет «реплей/live»: combatlog-покупки врага доступны в реплее, но могут
отсутствовать в live-интеграции — см. обсуждение утечек в README.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import (BUILDING_EVENT_TYPES, DEATH_EVENT_TYPES,
                     PURCHASE_EVENT_TYPES, DEFAULT_CONFIG, WardConfig)

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None


def _loads(raw: bytes):
    if _orjson is not None:
        return _orjson.loads(raw)
    return json.loads(raw)


@dataclass
class MatchEvents:
    """Нормализованные потоки одного матча."""

    match_id: str
    intervals: pd.DataFrame   # time, slot, x, y, gold, net_worth, xp, hero_id, visible?
    purchases: pd.DataFrame   # time, slot, item (только варды из cfg.obs_items)
    deaths: pd.DataFrame      # time, slot(victim; -1 если неизвестен), x, y
    obs: pd.DataFrame         # time, slot, x, y — установки Observer Wards
    buildings: pd.DataFrame   # time, slot, name, x, y, team — разрушения строений


_INTERVAL_COLS = ["time", "slot", "x", "y", "gold", "net_worth", "xp", "hero_id", "visible"]
_PURCHASE_COLS = ["time", "slot", "item"]
_DEATH_COLS = ["time", "slot", "x", "y"]
_OBS_COLS = ["time", "slot", "x", "y"]
_BUILDING_COLS = ["time", "slot", "name", "x", "y", "team"]

_HERO_NAME_KEYS = ("name", "unit", "hero_name", "unitname")
_NAME_KEYS = ("attackername", "targetname", "sourcename", "name", "unit", "unitname")


def _f(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_time(event: dict) -> float:
    return _f(event.get("time", event.get("game_time")))


def _resolve_slot(event: dict, name_to_slot: Dict[str, int],
                  prefer_name_key: Optional[str] = None) -> int:
    """Slot события: явное поле, затем имя героя, затем остальные имена."""
    if prefer_name_key is not None:
        name = event.get(prefer_name_key)
        if name in name_to_slot:
            return name_to_slot[name]
    slot = event.get("slot", event.get("player_slot"))
    if slot is not None:
        try:
            slot = int(slot)
            if 0 <= slot <= 9:
                return slot
        except (TypeError, ValueError):
            pass
    for key in _NAME_KEYS:
        name = event.get(key)
        if name in name_to_slot:
            return name_to_slot[name]
    return -1


def _normalize_xy(x: np.ndarray, y: np.ndarray, cfg: WardConfig):
    """Сетка 0..127 → world units; world units остаются как есть; NaN → NaN."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    is_world = (np.abs(x) > 200) | (np.abs(y) > 200)
    gx = np.where(is_world, x, (x - cfg.grid_offset) * cfg.grid_to_world)
    gy = np.where(is_world, y, (y - cfg.grid_offset) * cfg.grid_to_world)
    return gx.astype(np.float32), gy.astype(np.float32)


def load_match(path, cfg: WardConfig = DEFAULT_CONFIG) -> MatchEvents:
    """Прочитать один jsonl-файл матча → MatchEvents."""
    match_id = Path(path).stem
    iv_rows, buy_rows, death_rows, obs_rows, bld_rows = [], [], [], [], []
    name_to_slot: Dict[str, int] = {}

    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = _loads(raw)
            except ValueError:
                continue  # битая строка не должна ронять весь матч

            etype = event.get("type")
            t = _event_time(event)

            if etype == "interval":
                slot = _resolve_slot(event, name_to_slot)
                if 0 <= slot <= 9:
                    for key in _HERO_NAME_KEYS:
                        hero = event.get(key)
                        if isinstance(hero, str) and "npc_dota_hero_" in hero:
                            name_to_slot.setdefault(hero, slot)
                    iv_rows.append((
                        t, slot, _f(event.get("x")), _f(event.get("y")),
                        _f(event.get("gold")),
                        _f(event.get("net_worth", event.get("networth"))),
                        _f(event.get("xp")), _f(event.get("hero_id"), -1.0),
                        event.get("visible"),
                    ))

            elif etype in PURCHASE_EVENT_TYPES:
                item = event.get("valuename") or event.get("name") or event.get("item")
                if item in cfg.obs_items:
                    buy_rows.append((
                        t, _resolve_slot(event, name_to_slot, "attackername"), item,
                    ))

            elif etype in DEATH_EVENT_TYPES:
                # позиция смерти публична (kill feed), координаты могут отсутствовать
                death_rows.append((
                    t, _resolve_slot(event, name_to_slot, "targetname"),
                    _f(event.get("x")), _f(event.get("y")),
                ))

            elif etype == "obs":
                x, y = _f(event.get("x")), _f(event.get("y"))
                if not (math.isnan(x) or math.isnan(y)):
                    obs_rows.append((t, _resolve_slot(event, name_to_slot), x, y))

            elif etype in BUILDING_EVENT_TYPES:
                name = event.get("targetname") or event.get("valuename") \
                    or event.get("name") or ""
                pos = cfg.tower_positions.get(name)
                bld_rows.append((
                    t, _resolve_slot(event, name_to_slot), name,
                    _f(event.get("x")) if pos is None else pos[0],
                    _f(event.get("y")) if pos is None else pos[1],
                    pos[2] if pos is not None else int(event.get("team", -1)),
                ))

    intervals = pd.DataFrame(iv_rows, columns=_INTERVAL_COLS)
    if len(intervals):
        # строки без координат бесполезны для позиционных фичей (в реальных логах их нет)
        intervals = intervals.loc[intervals["x"].notna() & intervals["y"].notna()]
        intervals = intervals.reset_index(drop=True)
        if intervals["visible"].isna().all():
            intervals = intervals.drop(columns=["visible"])
    xs, ys = _normalize_xy(intervals["x"].to_numpy(np.float64),
                           intervals["y"].to_numpy(np.float64), cfg)
    intervals = intervals.assign(x=xs, y=ys)

    purchases = pd.DataFrame(buy_rows, columns=_PURCHASE_COLS)
    deaths = pd.DataFrame(death_rows, columns=_DEATH_COLS)
    obs = pd.DataFrame(obs_rows, columns=_OBS_COLS)
    buildings = pd.DataFrame(bld_rows, columns=_BUILDING_COLS)

    deaths = _with_normalized_coords(deaths, cfg)
    obs = _with_normalized_coords(obs, cfg)
    buildings = _with_normalized_coords(buildings, cfg)

    return MatchEvents(match_id, intervals, purchases, deaths, obs, buildings)


def _with_normalized_coords(df: pd.DataFrame, cfg: WardConfig) -> pd.DataFrame:
    if df.empty:
        return df
    xs, ys = _normalize_xy(df["x"].to_numpy(np.float64),
                           df["y"].to_numpy(np.float64), cfg)
    return df.assign(x=xs, y=ys)
