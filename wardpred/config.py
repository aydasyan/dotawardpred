"""Калибровочные константы пайплайна предсказания вардов.

Всё, что зависит от версии парсера и патча, собрано здесь — это точки калибровки,
а не «магические числа» в коде агрегации.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class WardConfig:
    # --- скользящие окна ------------------------------------------------------
    window_seconds: float = 60.0          # основное окно Δt
    long_window_seconds: float = 120.0    # расширенное окно (покупки вардов)
    local_radius: float = 1500.0          # радиус «локальной активности», world units
    max_time_unseen: float = 600.0        # ограничение сверху для time_unseen

    # --- целевая сетка 8x8 (world units, играбельная зона) ---------------------
    n_cells: int = 8
    map_min_x: float = -6800.0
    map_max_x: float = 6800.0
    map_min_y: float = -6800.0
    map_max_y: float = 6800.0

    # --- нормализация координат парсера ----------------------------------------
    # если |coord| <= 200 — считаем, что это сетка 0..127, и переводим в world units
    grid_to_world: float = 128.0
    grid_offset: float = 64.0

    # --- день/ночь --------------------------------------------------------------
    # сверено: цикл 10 мин (5 мин день с 0:00, 5 мин ночь) — gem.to_json-матчи
    day_night_phase: float = 300.0        # длина одной фазы (день или ночь), сек

    # --- предметы ---------------------------------------------------------------
    # item_ward_observer — имя в combatlog gem/OpenDota; item_observer_ward —
    # встречается в других выгрузках, оставлено для совместимости
    obs_items: frozenset = frozenset({
        "item_ward_observer", "item_observer_ward", "item_ward_dispenser",
    })

    # --- ключевые точки карты (world units) --------------------------------------
    # извлечены entity-экстрактором из реплеев patch 7.4x (DreamLeague);
    # рошан ходит между ямами (день/ночь), мучитель в конце матча может исчезать
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "roshan_a": (2760.0, -2740.0),      # SE-яма
        "roshan_b": (-3150.0, 2350.0),      # NW-яма
        "tormentor_a": (7744.0, -6208.0),   # SE угол
        "tormentor_b": (-7680.0, 6336.0),   # NW угол
    })

    # --- башни: name -> (x, y, team) ---------------------------------------------
    # координаты извлечены entity-экстрактором из реплеев patch 7.4x (DreamLeague);
    # смена карты между патчами требует пересборки
    tower_positions: Dict[str, Tuple[float, float, int]] = field(default_factory=lambda: {
        # Radiant (team 0)
        "npc_dota_tower1_top_radiant": (-6336.0, 1856.0, 0),
        "npc_dota_tower1_mid_radiant": (-1234.0, -1232.0, 0),
        "npc_dota_tower1_bot_radiant": (4859.9, -6379.2, 0),
        "npc_dota_tower2_top_radiant": (-6501.0, -872.0, 0),
        "npc_dota_tower2_mid_radiant": (-3008.0, -2744.0, 0),
        "npc_dota_tower2_bot_radiant": (-360.0, -6256.0, 0),
        "npc_dota_tower3_top_radiant": (-6592.0, -3408.0, 0),
        "npc_dota_tower3_mid_radiant": (-4640.0, -4144.0, 0),
        "npc_dota_tower3_bot_radiant": (-3952.0, -6112.0, 0),
        "npc_dota_tower4_top_radiant": (-5712.0, -4864.0, 0),
        "npc_dota_tower4_bot_radiant": (-5392.0, -5192.0, 0),
        # Dire (team 1)
        "npc_dota_tower1_top_dire": (-5274.6, 6036.0, 1),
        "npc_dota_tower1_mid_dire": (832.0, 848.0, 1),
        "npc_dota_tower1_bot_dire": (6269.3, -2240.0, 1),
        "npc_dota_tower2_top_dire": (-128.0, 6016.0, 1),
        "npc_dota_tower2_mid_dire": (2704.0, 2272.0, 1),
        "npc_dota_tower2_bot_dire": (6400.0, 384.0, 1),
        "npc_dota_tower3_top_dire": (3552.0, 5776.0, 1),
        "npc_dota_tower3_mid_dire": (4272.0, 3759.0, 1),
        "npc_dota_tower3_bot_dire": (6336.0, 3032.0, 1),
        "npc_dota_tower4_top_dire": (4944.0, 4776.0, 1),
        "npc_dota_tower4_bot_dire": (5280.0, 4432.0, 1),
    })


# --- типы событий, встречающиеся в логах (стиль OpenDota/Clarity) ---------------
PURCHASE_EVENT_TYPES = frozenset({"DOTA_COMBATLOG_PURCHASE", "purchase"})
DEATH_EVENT_TYPES = frozenset({"DOTA_COMBATLOG_DEATH", "death"})
BUILDING_EVENT_TYPES = frozenset({
    "DOTA_COMBATLOG_TOWERKILL",
    "DOTA_COMBATLOG_BARRACKS_KILL",
    "DOTA_COMBATLOG_BUILDING_DESTRUCTION",
})


DEFAULT_CONFIG = WardConfig()
