"""Адаптер gem-dota: .dem/.json реплей -> MatchEvents с честным туманом войны.

В отличие от jsonl-пути, здесь колонка ``visible`` в intervals — не эвристика,
а authoritative-видимость из реплея (``gem.hero_visibility_at``): видел ли
наблюдающая команда данного героя на этом срезе.

Один матч порождает ДВА MatchEvents — по одному на перспективу каждой команды:
  * intervals содержат всех 10 героев, ``visible`` посчитан от лица наблюдателя;
  * obs — только варды ВРАЖЕСКОЙ команды (это и есть целевая переменная:
    предсказываем, куда враг поставит вард).
  * match_id у обеих перспектив одинаковый — GroupShuffleSplit по match_id
    не разведёт перспективы одного матча в train и validation.

Система координат gem — 0..32768 со сдвигом; переводим в world units вычитанием
16384 (проверено: реальный разброс позиций 7.7k..25k центрируется в ±8.6k).

Покупки вардов врага берутся из combatlog — это replay-only информация, в live
её не будет (см. правило 5 в README/AGENTS.md). Поведение сознательно совпадает
с jsonl-путём.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, WardConfig
from .loader import MatchEvents

# gem отдаёт координаты в диапазоне ~0..32768; world units = coord - сдвиг
GEM_COORD_OFFSET = 16384.0

# gem: 2 = Radiant, 3 = Dire; внутреннее представление: 0 = Radiant, 1 = Dire
_GEM_TEAMS = (2, 3)
_GEM_TO_INTERNAL_TEAM = {2: 0, 3: 1}

_INTERVAL_COLS = ["time", "slot", "x", "y", "gold", "net_worth", "xp", "hero_id", "visible"]
_PURCHASE_COLS = ["time", "slot", "item"]
_DEATH_COLS = ["time", "slot", "x", "y"]
_OBS_COLS = ["time", "slot", "x", "y"]
_BUILDING_COLS = ["time", "slot", "name", "x", "y", "team"]

_TOWER_NAME_RE = re.compile(r"npc_dota_(goodguys|badguys)_tower(\d)(?:_(\w+))?$")
_TOWER_SIDE = {"goodguys": "radiant", "badguys": "dire"}


def _cfg_tower_name(gem_name: str) -> str:
    """'npc_dota_badguys_tower1_top' -> 'npc_dota_tower1_top_dire' (имена конфига)."""
    m = _TOWER_NAME_RE.match(gem_name)
    if not m or not m.group(3):
        return gem_name
    return f"npc_dota_tower{m.group(2)}_{m.group(3)}_{_TOWER_SIDE[m.group(1)]}"


def load_match_gem(path, cfg: WardConfig = DEFAULT_CONFIG) -> List[MatchEvents]:
    """Загрузить .dem (парсинг) или .json (кэш gem.to_json) -> список перспектив."""
    import gem  # локальный импорт: gem нужен только этому пути загрузки

    path = Path(path)
    if path.suffix.lower() == ".dem":
        match = gem.parse(str(path))
    else:
        match = gem.load_json(str(path))
    match_id = str(getattr(match, "match_id", None) or path.stem)
    return match_to_events(match, match_id, cfg)


def match_to_events(match, match_id: str,
                    cfg: WardConfig = DEFAULT_CONFIG) -> List[MatchEvents]:
    """Развернуть ParsedMatch в две перспективы (наблюдатель = каждая из команд)."""
    import gem

    clock = match.game_clock

    def game_time(tick: float) -> float:
        t = clock.game_time_at(int(tick))
        if t is None:
            t = (tick - (clock.game_start_tick or 0)) / 30.0
        return float(t)

    name_to_slot: Dict[str, int] = {p.hero_name: p.player_id for p in match.players}
    players = {p.player_id: p for p in match.players}

    # --- интервальные срезы: позиции + экономика на сетке ticks игрока ---------
    # visible зависит от перспективы, поэтому срезы собираем внутри цикла по командам
    base_rows: Dict[int, List[Tuple]] = {}
    for p in match.players:
        pos = {t: (x, y) for t, x, y in p.position_log}
        ticks = list(p.times)
        gold = list(p.gold_t) if p.gold_t is not None else [np.nan] * len(ticks)
        xp = list(p.xp_t) if p.xp_t is not None else [np.nan] * len(ticks)
        nw = list(p.net_worth_t) if p.net_worth_t is not None else [np.nan] * len(ticks)
        rows = []
        for tick, g, n_, x_ in zip(ticks, gold, nw, xp):
            xy = pos.get(tick)
            if xy is None:
                continue
            rows.append((game_time(tick), p.player_id,
                         xy[0] - GEM_COORD_OFFSET, xy[1] - GEM_COORD_OFFSET,
                         float(g), float(n_), float(x_), float(p.hero_id), tick))
        base_rows[p.player_id] = rows

    # --- покупки вардов (обе команды; replay-only, см. docstring модуля) -------
    purchases = pd.DataFrame(
        [(game_time(e.tick), p.player_id, e.value_name)
         for p in match.players
         for e in p.purchase_log
         if e.value_name in cfg.obs_items],
        columns=_PURCHASE_COLS)

    # --- смерти героев (kill feed публичен; координаты часто None — доберёт
    # _fill_death_coords по последней видимой позиции жертвы) -------------------
    death_rows = []
    for e in match.combat_log:
        if e.log_type is None or e.log_type.value != "DEATH" or not e.target_is_hero:
            continue
        slot = name_to_slot.get(e.target_name, -1)
        x = float(e.location_x) - GEM_COORD_OFFSET if e.location_x is not None else np.nan
        y = float(e.location_y) - GEM_COORD_OFFSET if e.location_y is not None else np.nan
        death_rows.append((game_time(e.tick), slot, x, y))
    deaths = pd.DataFrame(death_rows, columns=_DEATH_COLS)

    # --- разрушения башен -------------------------------------------------------
    bld_rows = []
    for t in match.towers:
        cfg_name = _cfg_tower_name(t.tower_name)
        pos = cfg.tower_positions.get(cfg_name)
        killer_slot = name_to_slot.get(t.killer, -1)
        bld_rows.append((
            game_time(t.tick), killer_slot, cfg_name,
            pos[0] if pos is not None else np.nan,
            pos[1] if pos is not None else np.nan,
            _GEM_TO_INTERNAL_TEAM.get(t.team, -1),
        ))
    buildings = pd.DataFrame(bld_rows, columns=_BUILDING_COLS)

    # --- перспективы -------------------------------------------------------------
    result: List[MatchEvents] = []
    for observing in _GEM_TEAMS:
        iv_rows = []
        for slot, rows in base_rows.items():
            for time_, slot_, x, y, g, nw, xp_, hero, tick in rows:
                state = gem.hero_visibility_at(
                    match, player_id=slot, observing_team=observing, tick=int(tick))
                visible = 1 if state is gem.VisibilityState.VISIBLE else 0
                iv_rows.append((time_, slot_, x, y, g, nw, xp_, hero, visible))
        intervals = pd.DataFrame(iv_rows, columns=_INTERVAL_COLS)
        intervals = intervals.sort_values("time", kind="stable").reset_index(drop=True)

        obs_rows = []
        for w in match.wards:
            if w.ward_type != "observer" or w.team != (5 - observing):
                continue
            if not (0 <= w.player_id <= 9):
                continue
            obs_rows.append((game_time(w.tick), w.player_id,
                             w.x - GEM_COORD_OFFSET, w.y - GEM_COORD_OFFSET))
        obs = pd.DataFrame(obs_rows, columns=_OBS_COLS)
        if obs.empty:
            continue  # перспектива без целевых событий бесполезна

        result.append(MatchEvents(match_id, intervals, purchases.copy(),
                                  deaths.copy(), obs, buildings.copy()))
    return result
