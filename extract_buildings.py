"""Одноразовый экстрактор позиций башен, рошана, мучителей и фонтанов из .dem.

Запуск:
    python extract_buildings.py replays/9097221853.dem
    python extract_buildings.py replays/9097221853.dem --out buildings_dump.json

Выход: JSON с именем сущности -> координаты gem и world units (gem - 16384).
"""

from __future__ import annotations

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WANTED_CLASSES = (
    "CDOTA_BaseNPC_Tower",
    "CDOTA_Unit_Roshan",
    "CDOTA_Unit_Tormentor",
    "CDOTA_Unit_Miniboss",
    "CDOTA_Unit_Fountain",
    "CDOTA_BaseNPC_Barracks",
    "CDOTA_BaseNPC_Fort",
)


def extract_buildings(dem_path: str, out_path: str = "buildings_dump.json") -> list[dict]:
    from gem.parser import ReplayParser
    from gem.schema.sendtable.models import FieldAccessPlan

    pos_plan = FieldAccessPlan((
        "CBodyComponent.m_cellX", "CBodyComponent.m_cellY",
        "CBodyComponent.m_vecX", "CBodyComponent.m_vecY",
    ))
    name_plan = FieldAccessPlan((
        "m_pEntity.m_nameStringTableIndex", "m_pEntity.m_nameStringableIndex",
    ))
    team_plan = FieldAccessPlan(("m_iTeamNum",))

    def _pos(entity):
        fx = entity._resolve_fields(pos_plan)
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
        fx = entity._resolve_fields(name_plan)
        idx = entity._get_int32_resolved(fx[0])
        if idx is None:
            idx = entity._get_int32_resolved(fx[1])
        if idx is None or idx < 0:
            return None
        item = entity_names.items.get(idx)
        return item[0] if item is not None else None

    parser = ReplayParser(dem_path)
    seen = {}

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
        team = entity._get_int32_resolved(entity._resolve_fields(team_plan)[0])
        out.append({
            "class": cls, "index": idx, "name": name, "team": team,
            "gem_x": round(pos[0], 1) if pos else None,
            "gem_y": round(pos[1], 1) if pos else None,
            "world_x": round(pos[0] - 16384.0, 1) if pos else None,
            "world_y": round(pos[1] - 16384.0, 1) if pos else None,
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Экстрактор позиций строений и landmarks из .dem")
    parser.add_argument("replay", help="путь к .dem файлу реплея")
    parser.add_argument("--out", default="buildings_dump.json",
                        help="выходной JSON-файл (по умолчанию: buildings_dump.json)")
    args = parser.parse_args()

    out = extract_buildings(args.replay, args.out)
    for row in out:
        print(f"{row['class']:28s} {str(row['name']):38s} team={row['team']} "
              f"world=({row['world_x']}, {row['world_y']})")
    print("DUMP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
