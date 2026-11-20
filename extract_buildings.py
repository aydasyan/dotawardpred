"""Одноразовый экстрактор позиций башен, рошана, мучителей и фонтанов из .dem.

Запуск: python extract_buildings.py
Выход: JSON с именем сущности -> координаты gem и world units (gem - 16384).
"""

from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from gem.parser import ReplayParser
from gem.schema.sendtable.models import FieldAccessPlan

_POS = FieldAccessPlan((
    "CBodyComponent.m_cellX", "CBodyComponent.m_cellY",
    "CBodyComponent.m_vecX", "CBodyComponent.m_vecY",
))
_NAME = FieldAccessPlan((
    "m_pEntity.m_nameStringTableIndex", "m_pEntity.m_nameStringableIndex",
))

WANTED_CLASSES = (
    "CDOTA_BaseNPC_Tower",
    "CDOTA_Unit_Roshan",
    "CDOTA_Unit_Tormentor",
    "CDOTA_Unit_Miniboss",
    "CDOTA_Unit_Fountain",
    "CDOTA_BaseNPC_Barracks",
    "CDOTA_BaseNPC_Fort",
)

parser = ReplayParser("replays/9097221853.dem")
seen = {}


def _pos(entity):
    fx = entity._resolve_fields(_POS)
    cell_x = entity._get_uint32_resolved(fx[0])
    cell_y = entity._get_uint32_resolved(fx[1])
    vec_x = entity._get_float32_resolved(fx[2])
    vec_y = entity._get_float32_resolved(fx[3])
    if None in (cell_x, cell_y, vec_x, vec_y):
        return None
    return cell_x * 128 + vec_x, cell_y * 128 + vec_y


def _name(entity, entity_names):
    if entity_names is None:
        return None
    fx = entity._resolve_fields(_NAME)
    idx = entity._get_int32_resolved(fx[0])
    if idx is None:
        idx = entity._get_int32_resolved(fx[1])
    if idx is None or idx < 0:
        return None
    item = entity_names.items.get(idx)
    return item[0] if item is not None else None


def on_entity(entity, op):
    cls = entity.get_class_name()
    if cls in WANTED_CLASSES:
        seen[(cls, entity.get_index())] = entity


parser.on_entity(on_entity)
parser.parse()

entity_names = parser.string_tables.get_by_name("EntityNames")

out = []
for (cls, idx), entity in sorted(seen.items()):
    pos = _pos(entity)
    name = _name(entity, entity_names)
    team = entity._get_int32_resolved(entity._resolve_fields(
        FieldAccessPlan(("m_iTeamNum",)))[0])
    out.append({
        "class": cls, "index": idx, "name": name, "team": team,
        "gem_x": round(pos[0], 1) if pos else None,
        "gem_y": round(pos[1], 1) if pos else None,
        "world_x": round(pos[0] - 16384.0, 1) if pos else None,
        "world_y": round(pos[1] - 16384.0, 1) if pos else None,
    })

with open("buildings_dump.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

for row in out:
    print(f"{row['class']:28s} {str(row['name']):38s} team={row['team']} "
          f"world=({row['world_x']}, {row['world_y']})")
print("DUMP OK")
