"""Кэширование результата парсинга .dem в JSON, чтобы не парсить повторно.

Примеры:
    python cache_parse.py replays/9097129353.dem
    python cache_parse.py replays/*.dem --delete-dem
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _expand(patterns) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(Path(pattern))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Парсинг .dem -> .json кэш gem-dota")
    parser.add_argument("replays", nargs="+", help=".dem файлы или glob-шаблон")
    parser.add_argument("--overwrite", action="store_true",
                        help="перезаписывать уже существующие .json кэши")
    parser.add_argument("--delete-dem", action="store_true",
                        help="удалять .dem после успешного создания кэша")
    args = parser.parse_args()

    paths = _expand(args.replays)
    if not paths:
        raise SystemExit("Нет файлов для парсинга")

    import gem

    ok = 0
    for dem_path in paths:
        out_path = dem_path.with_suffix(".json")
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {dem_path}: кэш {out_path} уже существует")
            ok += 1
            continue
        t0 = time.monotonic()
        try:
            match = gem.parse(str(dem_path))
            tmp_path = out_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(gem.to_json(match))
            tmp_path.replace(out_path)
            if args.delete_dem:
                dem_path.unlink()
            print(f"[ok]   {dem_path} -> {out_path} ({time.monotonic() - t0:.0f} с)")
            ok += 1
        except Exception as exc:
            print(f"[err]  {dem_path}: {exc!r}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
