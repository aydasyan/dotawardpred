"""CLI inference: тепловая карта вероятностей установки вражеского варда в момент T.

Примеры:
    python predict.py replays/9097129353.json --time 900 --slot 9
    python predict.py match.jsonl --time 600 --slot 7 --model ward_model.cbm --meta ward_model_meta.json
    python predict.py replays/9097129353.json --time 1200 --slot 4 --json-out heatmap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from wardpred.config import DEFAULT_CONFIG, WardConfig
from wardpred.features import build_snapshots
from wardpred.loader import load_match
from wardpred.model import GRID, predict_heatmap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SHADES = " .:-=+*#%@"


def _cell_center(cell: int, cfg: WardConfig = DEFAULT_CONFIG) -> tuple[float, float]:
    """Центр клетки 0..63 в мировых координатах (x, y)."""
    row, col = divmod(int(cell), cfg.n_cells)
    w = (cfg.map_max_x - cfg.map_min_x) / cfg.n_cells
    h = (cfg.map_max_y - cfg.map_min_y) / cfg.n_cells
    return cfg.map_min_x + (col + 0.5) * w, cfg.map_min_y + (row + 0.5) * h


def _load_perspective(path: Path, target_slot: int, cfg: WardConfig):
    """Загрузить матч и выбрать перспективу команды-наблюдателя (врага для target_slot)."""
    if path.suffix.lower() in (".dem", ".json"):
        from wardpred.gem_loader import load_match_gem
        perspectives = load_match_gem(path, cfg)
        if not perspectives:
            raise SystemExit(f"Не удалось загрузить перспективы из {path}")
        # слоты 0..4 — Radiant (team 0), значит наблюдает Dire (перспектива 1); и наоборот
        # в gem_loader: observing=2 (Radiant) идёт первым (target=Dire, слоты 5..9),
        # observing=3 (Dire) идёт вторым (target=Radiant, слоты 0..4).
        # Но если одна перспектива пустая по obs — проверяем по obs.slot или по видимости.
        for m in perspectives:
            if not m.obs.empty and ((m.obs["slot"] < 5).all() == (target_slot < 5)):
                return m
        return perspectives[0]
    return load_match(path, cfg)


def _render_ascii(heatmap: np.ndarray) -> str:
    """ASCII-тепловая карта 8x8 (сверху вниз: row=7..0, как на миникарте Dota 2)."""
    mx = float(heatmap.max())
    lines = []
    for row in range(GRID - 1, -1, -1):
        cells = []
        for col in range(GRID):
            v = float(heatmap[row, col])
            idx = int(round((v / mx) * (len(_SHADES) - 1))) if mx > 0 else 0
            cells.append(_SHADES[idx] * 2)
        lines.append(f"r{row} |" + " ".join(cells) + "|")
    lines.append("    " + " ".join(f"c{c}" for c in range(GRID)))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Предсказание тепловой карты варда в момент T")
    parser.add_argument("match", help="файл матча (.json кэш gem, .dem или .jsonl)")
    parser.add_argument("--time", type=float, required=True,
                        help="игровое время T в секундах")
    parser.add_argument("--slot", type=int, required=True, choices=range(10),
                        help="слот вражеского игрока (0..4 Radiant, 5..9 Dire)")
    parser.add_argument("--model", default="ward_model.cbm",
                        help="путь к модели CatBoost (по умолчанию: ward_model.cbm)")
    parser.add_argument("--meta", default="ward_model_meta.json",
                        help="путь к JSON-метаданным (по умолчанию: ward_model_meta.json)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="сколько горячих клеток вывести (по умолчанию: 3)")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="Gaussian blur sigma для тепловой карты (по умолчанию: 0.7)")
    parser.add_argument("--json-out", default=None,
                        help="опционально сохранить результат (heatmap + top_k) в JSON")
    args = parser.parse_args()

    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    feature_cols = meta["feature_cols"]

    model = CatBoostClassifier()
    model.load_model(args.model)

    match = _load_perspective(Path(args.match), args.slot, DEFAULT_CONFIG)
    snap = build_snapshots(match, np.array([args.time]), np.array([args.slot]), DEFAULT_CONFIG)
    X = snap[feature_cols]

    heatmap = predict_heatmap(model, X, sigma=args.sigma)
    flat = heatmap.ravel()
    k = min(args.top_k, len(flat))
    top_cells = np.argsort(-flat)[:k]

    print(f"Матч: {match.match_id} | T={args.time:.1f} с | слот={args.slot} "
          f"(hero_id={int(snap.iloc[0]['hero_id'])}, time_unseen={float(snap.iloc[0]['time_unseen']):.0f} с)")
    print(f"\nTop-{k} вероятных клеток (сглаженная карта, sigma={args.sigma}):")
    top_list = []
    for rank, cell in enumerate(top_cells, 1):
        row, col = divmod(int(cell), GRID)
        cx, cy = _cell_center(int(cell), DEFAULT_CONFIG)
        prob = float(flat[cell])
        top_list.append({
            "rank": rank, "cell": int(cell), "row": row, "col": col,
            "world_x": round(cx, 1), "world_y": round(cy, 1), "prob": round(prob, 5),
        })
        print(f"  #{rank}: cell={int(cell):2d} (row={row}, col={col}) "
              f"center=({cx:+.0f}, {cy:+.0f})  p={prob:.3f}")

    print("\nТепловая карта 8x8 (row 7 = север, col 7 = восток):")
    print(_render_ascii(heatmap))

    if args.json_out:
        payload = {
            "match_id": match.match_id,
            "time": args.time,
            "slot": args.slot,
            "sigma": args.sigma,
            "top_k": top_list,
            "heatmap": heatmap.tolist(),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nСохранено в {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
