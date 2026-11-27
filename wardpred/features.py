"""Снэпшоты признаков в момент T (вражеский саппорт установил вард).

Инвариант анти-утечки: все запросы к событиям — строго time <= T. Целевые
obs-события никогда не попадают в признаки по построению (фичи берутся только
из потоков intervals/purchases/deaths/buildings).

Производительность:
  * «сколько событий в окне [T-Δt, T]» — разность np.searchsorted по
    отсортированным массивам, векторно по всем снэпшотам сразу → O((N+M)·log N)
    вместо по-событийных булевых масок O(M·N);
  * геометрическое окно (смерти/строения в радиусе) — collections.deque с
    амортизированным O(1) выселением, т.к. нужен попредметный радиус;
  * last-known position — searchsorted по потоку интервальных срезов слота.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, WardConfig
from .loader import MatchEvents

FEATURE_COLS: List[str] = [
    "game_time", "is_night",
    "networth_advantage", "xp_advantage",
    "last_x", "last_y", "time_unseen",
    "dist_roshan", "dist_tormentor",
    "dist_nearest_ally_tower", "dist_nearest_enemy_tower",
    "local_deaths_60s", "local_buildings_60s", "local_min_death_dist",
    "deaths_60s_global",
    "obs_bought_60s", "obs_bought_120s", "time_since_obs_buy",
    "slot", "team", "hero_id",
]

# Признаки из combatlog покупок врага: доступны в реплее, но отсутствуют в live
REPLAY_ONLY_FEATURES: Tuple[str, ...] = (
    "obs_bought_60s",
    "obs_bought_120s",
    "time_since_obs_buy",
)

_EMPTY_SNAPSHOTS = pd.DataFrame(columns=FEATURE_COLS + ["match_id"])
_EMPTY_RESULT = pd.DataFrame(columns=FEATURE_COLS + ["cell", "ward_x", "ward_y", "match_id"])

_WARNED = {"perspective": False}


# --- целевая переменная ---------------------------------------------------------
def xy_to_cell(x: np.ndarray, y: np.ndarray, cfg: WardConfig) -> np.ndarray:
    """Мировые координаты → класс 0..63 (row-major: row * n + col).

    row растёт по Y снизу вверх, col — по X слева направо.
    """
    n = cfg.n_cells
    w = (cfg.map_max_x - cfg.map_min_x) / n
    h = (cfg.map_max_y - cfg.map_min_y) / n
    col = np.clip(((np.asarray(x) - cfg.map_min_x) // w).astype(np.int64), 0, n - 1)
    row = np.clip(((np.asarray(y) - cfg.map_min_y) // h).astype(np.int64), 0, n - 1)
    return row * n + col


# --- геометрическое скользящее окно на deque ------------------------------------
class LocalActivityWindow:
    """Окно последних WINDOW_SECONDS секунд: (time, x, y, kind).

    kind: 0 — смерть героя, 1 — разрушенное строение.
    """

    __slots__ = ("_dq", "_max_age")

    def __init__(self, max_age: float):
        self._dq: deque = deque()
        self._max_age = max_age

    def push(self, time: float, x: float, y: float, kind: int) -> None:
        self._dq.append((time, x, y, kind))

    def evict(self, now: float) -> None:
        cutoff = now - self._max_age
        dq = self._dq
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def stats(self, x: float, y: float, radius: float) -> Tuple[int, int, float]:
        """Смерти/строения и минимальная дистанция в радиусе от (x, y)."""
        deaths = buildings = 0
        min_dist = math.inf
        r2 = radius * radius
        for _t, ex, ey, kind in self._dq:
            d2 = (ex - x) * (ex - x) + (ey - y) * (ey - y)
            if d2 <= r2:
                min_dist = min(min_dist, math.sqrt(d2))
                if kind == 0:
                    deaths += 1
                else:
                    buildings += 1
        return deaths, buildings, (-1.0 if min_dist == math.inf else min_dist)


# --- вспомогательные функции ------------------------------------------------------
def _dedup_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Убрать дубли obs (стеки/повторы диспенсера): тот же слот, Δt<1с, те же координаты."""
    obs = obs.sort_values(["slot", "time"], kind="stable").reset_index(drop=True)
    dup = (
        obs["slot"].eq(obs["slot"].shift())
        & (obs["time"] - obs["time"].shift() < 1.0)
        & ((obs["x"] - obs["x"].shift()).abs() < 1.0)
        & ((obs["y"] - obs["y"].shift()).abs() < 1.0)
    )
    return obs.loc[~dup.fillna(False)].reset_index(drop=True)


def _prepare_slot_stream(intervals: pd.DataFrame, slot: int) -> Optional[pd.DataFrame]:
    """Поток позиции слота, известный НАМ: только видимые срезы (если есть флаг)."""
    sub = intervals.loc[intervals["slot"] == slot]
    if "visible" in sub.columns:
        # NaN-флаг считаем «не видно» — консервативно против утечки
        sub = sub.loc[sub["visible"].fillna(0).astype(bool)]
    if sub.empty:
        return None
    return sub.sort_values("time", kind="stable")


def _last_known(frame: Optional[pd.DataFrame], times: np.ndarray):
    """Последняя известная позиция слота для каждого времени в `times`.

    Возвращает (x, y, t_last_seen); если слот ещё не виделся — NaN.
    """
    n = len(times)
    lx = np.full(n, np.nan, dtype=np.float64)
    ly = np.full(n, np.nan, dtype=np.float64)
    t_seen = np.full(n, np.nan, dtype=np.float64)
    if frame is None:
        return lx, ly, t_seen
    stream_t = frame["time"].to_numpy(np.float64)
    pos = np.searchsorted(stream_t, times, side="right") - 1
    ok = pos >= 0
    pc = np.clip(pos, 0, None)
    lx = np.where(ok, frame["x"].to_numpy()[pc], np.nan)
    ly = np.where(ok, frame["y"].to_numpy()[pc], np.nan)
    t_seen = np.where(ok, stream_t[pc], np.nan)
    return lx, ly, t_seen


def _team_total(frames: List[Optional[pd.DataFrame]], col: str,
                times: np.ndarray) -> np.ndarray:
    """Сумма последних известных значений `col` по 5 слотам команды."""
    total = np.zeros(len(times), dtype=np.float64)
    for frame in frames:
        if frame is None:
            continue
        stream_t = frame["time"].to_numpy(np.float64)
        values = np.nan_to_num(frame[col].to_numpy(np.float64), nan=0.0)
        pos = np.searchsorted(stream_t, times, side="right") - 1
        ok = pos >= 0
        total += np.where(ok, values[np.clip(pos, 0, None)], 0.0)
    return total


def _fill_death_coords(deaths: pd.DataFrame,
                       slot_frames: Dict[int, Optional[pd.DataFrame]]) -> pd.DataFrame:
    """Добрать координаты смертей по последней позиции жертвы (смерть публична)."""
    if deaths.empty:
        return deaths
    x = deaths["x"].to_numpy(np.float64).copy()
    y = deaths["y"].to_numpy(np.float64).copy()
    need = np.isnan(x) | np.isnan(y)
    if not need.any():
        return deaths
    slots = deaths["slot"].to_numpy()
    times = deaths["time"].to_numpy(np.float64)
    for slot, frame in slot_frames.items():
        idx = np.flatnonzero(need & (slots == slot))
        if idx.size == 0 or frame is None:
            continue
        stream_t = frame["time"].to_numpy(np.float64)
        pos = np.searchsorted(stream_t, times[idx], side="right") - 1
        ok = pos >= 0
        pc = np.clip(pos, 0, None)
        x[idx] = np.where(ok, frame["x"].to_numpy()[pc], x[idx])
        y[idx] = np.where(ok, frame["y"].to_numpy()[pc], y[idx])
    return deaths.assign(x=x.astype(np.float32), y=y.astype(np.float32))


def _towers_timeline(buildings: pd.DataFrame, cfg: WardConfig) -> pd.DataFrame:
    """Статические башни из конфига + первое время разрушения из лога."""
    kill_map: Dict[str, float] = {}
    if not buildings.empty:
        first_kill = buildings.sort_values("time").drop_duplicates("name", keep="first")
        kill_map = dict(zip(first_kill["name"], first_kill["time"].astype(float)))
    rows = []
    for name, (x, y, team) in cfg.tower_positions.items():
        rows.append({"name": name, "x": x, "y": y, "team": team,
                     "destroyed_at": kill_map.get(name, np.inf)})
    return pd.DataFrame(rows)


def _nearest_alive_tower(towers: pd.DataFrame, support_team: np.ndarray,
                         obs_t: np.ndarray, lx: np.ndarray, ly: np.ndarray):
    """Дистанция до ближайшей живой башни своей и вражеской (для саппорта) команды."""
    n = len(obs_t)
    best_ally = np.full(n, np.inf)
    best_enemy = np.full(n, np.inf)
    if not towers.empty:
        tx = towers["x"].to_numpy(np.float32)
        ty = towers["y"].to_numpy(np.float32)
        tteam = towers["team"].to_numpy()
        tdead = towers["destroyed_at"].to_numpy(np.float64)
        for i in range(len(towers)):          # ~20 башен; внутри — векторно
            d = np.hypot(lx - tx[i], ly - ty[i])
            alive = obs_t < tdead[i]
            ally = alive & (tteam[i] == support_team)
            enemy = alive & (tteam[i] != support_team) & (tteam[i] >= 0)
            best_ally = np.where(ally & (d < best_ally), d, best_ally)
            best_enemy = np.where(enemy & (d < best_enemy), d, best_enemy)
    return (np.where(np.isfinite(best_ally), best_ally, -1.0),
            np.where(np.isfinite(best_enemy), best_enemy, -1.0))


def _dist_to_landmarks(lx: np.ndarray, ly: np.ndarray, cfg: WardConfig,
                       names: Tuple[str, ...]) -> np.ndarray:
    best = np.full(len(lx), np.inf)
    for name in names:
        point = cfg.landmarks.get(name)
        if point is not None:
            best = np.minimum(best, np.hypot(lx - point[0], ly - point[1]))
    return np.where(np.isfinite(best), best, -1.0)


def _nan_to_marker(values: np.ndarray) -> np.ndarray:
    """NaN → -1.0 (маркер «нет данных»), остальное во float32."""
    return np.where(np.isnan(values), -1.0, values).astype(np.float32)


# --- главный вход -----------------------------------------------------------------
def build_snapshots(match: MatchEvents,
                    times: np.ndarray,
                    slots: np.ndarray,
                    cfg: WardConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Собрать снэпшоты признаков в произвольные моменты `times` для слотов `slots`.

    Не использует `match.obs` — подходит как для обучения (вызывается из
    `build_match_features`), так и для inference в произвольный момент игры `T`.
    Инвариант анти-утечки: в каждый снэпшот попадают только события с `time <= T`.
    """
    obs_t = np.asarray(times, dtype=np.float64)
    obs_slot = np.asarray(slots, dtype=np.int16)
    if obs_t.shape != obs_slot.shape or obs_t.ndim != 1:
        raise ValueError("times и slots должны быть одномерными массивами одинаковой длины")
    if len(obs_t) == 0:
        return _EMPTY_SNAPSHOTS.copy()
    if (obs_slot < 0).any() or (obs_slot > 9).any():
        raise ValueError("slots должны быть в диапазоне 0..9")

    if "visible" not in match.intervals.columns and not _WARNED["perspective"]:
        _WARNED["perspective"] = True
        warnings.warn(
            "В interval-срезах нет флага 'visible': считаем, что поток уже от лица "
            "наблюдателя. Если это full-parse координаты — last_x/time_unseen утекают!",
            stacklevel=2,
        )

    n = len(obs_t)
    support_team = np.where(obs_slot < 5, 0, 1).astype(np.int8)   # слоты 0..4 — Radiant
    our_team = 1 - support_team

    # --- поток срезов по слотам ----------------------------------------------------
    slot_frames: Dict[int, Optional[pd.DataFrame]] = {
        s: _prepare_slot_stream(match.intervals, s) for s in range(10)
    }
    radiant_frames = [slot_frames[s] for s in range(5)]
    dire_frames = [slot_frames[s] for s in range(5, 10)]

    # --- last known position / time_unseen / hero ---------------------------------
    last_x = np.full(n, np.nan)
    last_y = np.full(n, np.nan)
    t_seen = np.full(n, np.nan)
    hero_id = np.full(n, -1.0)
    for slot in np.unique(obs_slot):
        idx = np.flatnonzero(obs_slot == slot)
        frame = slot_frames[int(slot)]
        lx, ly, tl = _last_known(frame, obs_t[idx])
        last_x[idx], last_y[idx], t_seen[idx] = lx, ly, tl
        if frame is not None and "hero_id" in frame.columns and len(frame):
            # hero_id строго на последний видимый срез <= T, а не последний в файле
            stream_t = frame["time"].to_numpy(np.float64)
            pos = np.searchsorted(stream_t, obs_t[idx], side="right") - 1
            ok = pos >= 0
            hero_vals = frame["hero_id"].to_numpy(np.float64)
            hero_id[idx] = np.where(ok, hero_vals[np.clip(pos, 0, None)], -1.0)
    unseen = obs_t - t_seen
    time_unseen = np.where(np.isnan(unseen), cfg.max_time_unseen,
                           np.clip(unseen, 0.0, cfg.max_time_unseen))

    # --- глобальные: gold/xp преимущество (с нашей стороны) ------------------------
    radiant_nw = _team_total(radiant_frames, "net_worth", obs_t)
    dire_nw = _team_total(dire_frames, "net_worth", obs_t)
    radiant_xp = _team_total(radiant_frames, "xp", obs_t)
    dire_xp = _team_total(dire_frames, "xp", obs_t)
    nw_adv = np.where(our_team == 0, radiant_nw - dire_nw, dire_nw - radiant_nw)
    xp_adv = np.where(our_team == 0, radiant_xp - dire_xp, dire_xp - radiant_xp)

    # --- геометрия -----------------------------------------------------------------
    dist_roshan = _nan_to_marker(
        _dist_to_landmarks(last_x, last_y, cfg, ("roshan_a", "roshan_b")))
    dist_tormentor = _nan_to_marker(
        _dist_to_landmarks(last_x, last_y, cfg, ("tormentor_a", "tormentor_b")))
    towers = _towers_timeline(match.buildings, cfg)
    dist_ally, dist_enemy = _nearest_alive_tower(towers, support_team, obs_t, last_x, last_y)

    # --- локальная активность: смерти + строения в радиусе (deque) ------------------
    deaths = _fill_death_coords(match.deaths, slot_frames)
    local_parts = []
    if not deaths.empty:
        d = deaths.loc[deaths["x"].notna()]
        if len(d):
            local_parts.append(pd.DataFrame({"time": d["time"].to_numpy(np.float64),
                                             "x": d["x"].to_numpy(np.float32),
                                             "y": d["y"].to_numpy(np.float32),
                                             "kind": np.zeros(len(d), np.int8)}))
    if not match.buildings.empty:
        b = match.buildings.loc[match.buildings["x"].notna()]
        if len(b):
            local_parts.append(pd.DataFrame({"time": b["time"].to_numpy(np.float64),
                                             "x": b["x"].to_numpy(np.float32),
                                             "y": b["y"].to_numpy(np.float32),
                                             "kind": np.ones(len(b), np.int8)}))
    if local_parts:
        events = pd.concat(local_parts, ignore_index=True).sort_values("time", kind="stable")
        ev_t = events["time"].to_numpy(np.float64)
        ev_x = events["x"].to_numpy(np.float32)
        ev_y = events["y"].to_numpy(np.float32)
        ev_kind = events["kind"].to_numpy(np.int8)
    else:
        ev_t = np.empty(0, np.float64)
        ev_x = ev_y = np.empty(0, np.float32)
        ev_kind = np.empty(0, np.int8)

    loc_deaths = np.zeros(n)
    loc_buildings = np.zeros(n)
    loc_min_dist = np.full(n, -1.0)
    window = LocalActivityWindow(cfg.window_seconds)
    ptr = 0
    # снэпшоты в глобальном порядке времени — обязательное условие корректности deque
    for i in np.argsort(obs_t, kind="stable"):
        t = obs_t[i]
        while ptr < len(ev_t) and ev_t[ptr] <= t:
            window.push(ev_t[ptr], ev_x[ptr], ev_y[ptr], int(ev_kind[ptr]))
            ptr += 1
        window.evict(t)
        if math.isfinite(last_x[i]):
            deaths_n, buildings_n, min_d = window.stats(last_x[i], last_y[i],
                                                        cfg.local_radius)
            loc_deaths[i], loc_buildings[i], loc_min_dist[i] = deaths_n, buildings_n, min_d

    # --- глобальные смерти за окно (searchsorted-разность, векторно) ----------------
    death_t = np.sort(match.deaths["time"].to_numpy(np.float64)) \
        if not match.deaths.empty else np.empty(0, np.float64)
    hi = np.searchsorted(death_t, obs_t, side="right")
    deaths_global = hi - np.searchsorted(death_t, obs_t - cfg.window_seconds, side="left")

    # --- покупки вардов саппортом (прокси obs_in_inventory) -------------------------
    bought60 = np.zeros(n)
    bought120 = np.zeros(n)
    since_buy = np.full(n, -1.0)
    buy_by_slot: Dict[int, np.ndarray] = {}
    if not match.purchases.empty:
        p = match.purchases.sort_values("time", kind="stable")
        for slot, grp in p.groupby("slot"):
            buy_by_slot[int(slot)] = grp["time"].to_numpy(np.float64)
    for slot in np.unique(obs_slot):
        idx = np.flatnonzero(obs_slot == slot)
        times = obs_t[idx]
        pt = buy_by_slot.get(int(slot))
        if pt is None or pt.size == 0:
            continue
        hi = np.searchsorted(pt, times, side="right")
        bought60[idx] = hi - np.searchsorted(pt, times - cfg.window_seconds, side="left")
        bought120[idx] = hi - np.searchsorted(pt, times - cfg.long_window_seconds, side="left")
        has = hi > 0
        since_buy[idx] = np.where(has, times - pt[np.clip(hi - 1, 0, None)], -1.0)

    # --- сборка ---------------------------------------------------------------------
    out = pd.DataFrame({
        "match_id": match.match_id,
        "game_time": obs_t,
        "is_night": ((obs_t // cfg.day_night_phase) % 2).astype(np.int8),
        "networth_advantage": nw_adv,
        "xp_advantage": xp_adv,
        "last_x": _nan_to_marker(last_x),
        "last_y": _nan_to_marker(last_y),
        "time_unseen": time_unseen.astype(np.float32),
        "dist_roshan": dist_roshan,
        "dist_tormentor": dist_tormentor,
        "dist_nearest_ally_tower": dist_ally.astype(np.float32),
        "dist_nearest_enemy_tower": dist_enemy.astype(np.float32),
        "local_deaths_60s": loc_deaths.astype(np.int16),
        "local_buildings_60s": loc_buildings.astype(np.int16),
        "local_min_death_dist": loc_min_dist.astype(np.float32),
        "deaths_60s_global": deaths_global.astype(np.int16),
        "obs_bought_60s": bought60.astype(np.int8),
        "obs_bought_120s": bought120.astype(np.int8),
        "time_since_obs_buy": since_buy.astype(np.float32),
        "slot": obs_slot.astype(np.int8),
        "team": support_team,
        "hero_id": hero_id.astype(np.int16),
    })
    ordered = [c for c in FEATURE_COLS if c in out.columns] + ["match_id"]
    return out[ordered]


def build_match_features(match: MatchEvents,
                         cfg: WardConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Собрать по одному снэпшоту признаков на каждое obs-событие матча."""
    obs = _dedup_obs(match.obs)
    if obs.empty:
        return _EMPTY_RESULT.copy()
    obs = obs.loc[(obs["slot"] >= 0) & (obs["slot"] <= 9)].reset_index(drop=True)
    if obs.empty:
        return _EMPTY_RESULT.copy()

    obs_t = obs["time"].to_numpy(np.float64)
    obs_slot = obs["slot"].to_numpy(np.int16)
    obs_x = obs["x"].to_numpy(np.float32)
    obs_y = obs["y"].to_numpy(np.float32)

    out = build_snapshots(match, obs_t, obs_slot, cfg)
    out["cell"] = xy_to_cell(obs_x, obs_y, cfg).astype(np.int16)
    out["ward_x"] = obs_x
    out["ward_y"] = obs_y
    # канонический порядок: признаки, таргет, match_id
    ordered = [c for c in FEATURE_COLS if c in out.columns] \
        + ["cell", "ward_x", "ward_y", "match_id"]
    return out[ordered]
