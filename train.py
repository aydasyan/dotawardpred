"""Сборка датасета из матчей и обучение модели.

Входные файлы:
  * .jsonl — логи в стиле OpenDota/Clarity (loader.load_match);
  * .dem   — сырой реплей Source 2, парсится через gem-dota (медленно);
  * .json  — кэш gem.to_json, мгновенная загрузка (gem-dota).

Пути .dem/.json дают честный туман войны (authoritative-видимость из реплея)
и раскрываются в две перспективы на матч — см. wardpred/gem_loader.py.

Пример:
    python train.py data/matches/*.jsonl --model-out ward_model.cbm
    python train.py replays/*.json --workers 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from wardpred.config import DEFAULT_CONFIG
from wardpred.features import build_match_features
from wardpred.loader import load_match
from wardpred.model import evaluate_quality, save_meta, train_ward_classifier


def _load_any(path, cfg) -> list:
    """Один файл -> один или несколько MatchEvents (gem-путь даёт 2 перспективы)."""
    if Path(path).suffix.lower() in (".dem", ".json"):
        from wardpred.gem_loader import load_match_gem  # gem нужен только здесь
        return load_match_gem(path, cfg)
    return [load_match(path, cfg)]


def _process_file(task) -> tuple:
    path, = task
    try:
        frames = [build_match_features(m, DEFAULT_CONFIG)
                  for m in _load_any(path, DEFAULT_CONFIG)]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return path, frame, None
    except Exception as exc:  # один битый матч не роняет сборку
        return path, None, repr(exc)


def expand_paths(patterns) -> list:
    paths = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(Path(pattern))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение предсказателя вардов")
    parser.add_argument("matches", nargs="+", help="jsonl/dem/json-файлы или glob")
    parser.add_argument("--model-out", default="ward_model.cbm")
    parser.add_argument("--meta-out", default="ward_model_meta.json")
    parser.add_argument("--workers", type=int, default=1,
                        help="процессы для сборки датасета (1 = последовательно)")
    args = parser.parse_args()

    paths = expand_paths(args.matches)
    if not paths:
        raise SystemExit("Нет файлов для обработки")

    frames = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for path, frame, error in pool.map(_process_file, [(p,) for p in paths]):
                if error:
                    print(f"[skip] {path}: {error}")
                elif len(frame):
                    frames.append(frame)
                    print(f"[ok]   {path}: {len(frame)} снэпшотов")
    else:
        for path in paths:
            _, frame, error = _process_file((path,))
            if error:
                print(f"[skip] {path}: {error}")
            elif len(frame):
                frames.append(frame)
                print(f"[ok]   {path}: {len(frame)} снэпшотов")

    if not frames:
        raise SystemExit("Ни в одном матче не оказалось obs-событий")

    dataset = pd.concat(frames, ignore_index=True)
    print(f"\nДатасет: {len(dataset)} снэпшотов, "
          f"{dataset['cell'].nunique()} непустых клеток из 64")
    print(dataset["cell"].value_counts().describe().to_string())

    model, meta = train_ward_classifier(dataset)

    metrics = evaluate_quality(model, dataset)
    meta["metrics"] = metrics
    if metrics.get("n_val"):
        print("\nКачество на отложенных матчах (n={n_val}):".format(**metrics))
        for k in (1, 3):
            if f"top{k}" in metrics:
                print(f"  Top-{k}: модель {metrics[f'top{k}']:.3f} | "
                      f"частоты {metrics[f'frequency_top{k}']:.3f} | "
                      f"равномерно {metrics[f'uniform_top{k}']:.3f}")
        if "last_position_top1" in metrics:
            print(f"  Baseline 'клетка последней позиции': "
                  f"{metrics['last_position_top1']:.3f} "
                  f"(покрытие {metrics['last_position_coverage']:.0%})")

    model.save_model(args.model_out)
    save_meta(args.meta_out, meta)
    print(f"\nМодель: {args.model_out} (best_iteration={meta['best_iteration']})")
    print(f"Мета:   {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
