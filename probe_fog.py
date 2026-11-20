"""Проверка hero_visibility_at из gem-dota на реальном реплее.

Санити-чеки:
  1) свой герой виден своей команде ~100% времени;
  2) враг виден частично (правдоподобная доля тумана);
  3) ночью вражеская видимость ниже, чем днём;
  4) после смерти/до первого наблюдения состояние не «visible» зря.
"""

from __future__ import annotations

import collections
import sys

sys.stdout.reconfigure(encoding="utf-8")

import os

import gem

REPLAY = "replays/9097129353.dem"
CACHE = "replays/9097129353.json"

if os.path.exists(CACHE):
    match = gem.load_json(CACHE)
    print("loaded from cache")
else:
    match = gem.parse(REPLAY)

print("players:")
players = {}
for p in match.players:
    players[p.player_id] = p
    print(f"  slot={p.player_id} team={p.team} hero={p.hero_name}")

# длительность в тиках
end_tick = match.game_end_tick
start = match.game_start_tick or 0
print("start/end tick:", start, end_tick)
print("hero_visibility_events:", len(match.hero_visibility_events))
print("example event:", match.hero_visibility_events[0] if match.hero_visibility_events else None)
print("entity_visibility_events:", len(match.entity_visibility_events))

# сэмплируем видимость раз в 5 секунд (150 тиков)
STEP = 150
stats = collections.defaultdict(lambda: collections.Counter())
daynight = collections.defaultdict(lambda: collections.Counter())  # (slot, 'day'/'night') -> counter

# начало игры: фазы день/ночь по 10 минут = 18000 тиков от game_start_tick
if end_tick is None:
    sys.exit("net end_tick")

n_unknown = 0
for slot, p in players.items():
    enemy_team = 3 if p.team == 2 else 2
    for t in range(start, end_tick, STEP):
        st_own = gem.hero_visibility_at(match, player_id=slot, observing_team=p.team, tick=t)
        st_enemy = gem.hero_visibility_at(match, player_id=slot, observing_team=enemy_team, tick=t)
        stats[(slot, "own")][str(st_own)] += 1
        stats[(slot, "enemy")][str(st_enemy)] += 1
        phase = "day" if ((t - start) // 18000) % 2 == 0 else "night"
        daynight[(slot, phase)][str(st_enemy)] += 1

print("\n=== видимость: доля состояний ===")
for slot in sorted(players):
    p = players[slot]
    own = stats[(slot, "own")]
    enemy = stats[(slot, "enemy")]
    tot_o = sum(own.values())
    tot_e = sum(enemy.values())
    print(f"slot {slot} ({p.hero_name}):")
    print(f"  own:   " + ", ".join(f"{k}={v / tot_o:.1%}" for k, v in sorted(own.items())))
    print(f"  enemy: " + ", ".join(f"{k}={v / tot_e:.1%}" for k, v in sorted(enemy.items())))

print("\n=== вражеская видимость: день vs ночь ===")
for slot in sorted(players):
    row = []
    for phase in ("day", "night"):
        c = daynight[(slot, phase)]
        tot = sum(c.values())
        vis = c.get("visible", 0) / tot if tot else float("nan")
        row.append(f"{phase}: visible={vis:.1%} (n={tot})")
    print(f"slot {slot} ({players[slot].hero_name}): " + "  ".join(row))

print("\nPROBE OK")
