"""Автоматический сбор датасета реплеев: OpenDota API -> .dem -> .json кэш.

Поток на каждый матч:
  1. Берём свежие match_id из OpenDota (/api/proMatches или /api/parsedMatches)
     либо из --match-ids.
  2. Пропускаем уже закэшированные (<out_dir>/<match_id>.json).
  3. Скачиваем реплей через gem.fetch_replay, парсим через gem.parse и
     сохраняем кэш gem.to_json.
  4. По умолчанию удаляем сырой .dem после успешного кэширования (экономия
     сотен МБ на матч; флаг --keep-dem оставляет .dem).

Пример:
    python collect_dataset.py --limit 20 --out-dir replays
    python collect_dataset.py --match-ids 9097129353 9097221853 --keep-dem
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_OPENDOTA_ENDPOINTS = {
    "pro": "https://api.opendota.com/api/proMatches",
    "parsed": "https://api.opendota.com/api/parsedMatches",
}


def fetch_recent_match_ids(source: str, limit: int, max_age_days: float) -> List[int]:
    """Получить свежие match_id из OpenDota API (не старше max_age_days)."""
    url = _OPENDOTA_ENDPOINTS[source]
    req = urllib.request.Request(url, headers={"User-Agent": "DotaWardPrediction/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    cutoff = time.time() - max_age_days * 86400.0
    ids: List[int] = []
    seen = set()
    for row in payload:
        mid = row.get("match_id")
        if mid is None or mid in seen:
            continue
        start_time = row.get("start_time")
        if start_time is not None and float(start_time) < cutoff:
            continue
        seen.add(mid)
        ids.append(int(mid))
        if len(ids) >= limit:
            break
    return ids


def process_match(match_id: int, out_dir: Path, keep_dem: bool = False) -> bool:
    """Скачать, распарсить и закэшировать один матч. Возвращает True при успехе."""
    import gem  # локальный импорт: gem нужен только при фактическом запуске

    cache_path = out_dir / f"{match_id}.json"
    if cache_path.exists():
        print(f"[cached] {match_id}: уже есть {cache_path}")
        return True

    dem_path = out_dir / f"{match_id}.dem"
    t0 = time.monotonic()
    try:
        if not dem_path.exists():
            fetched = gem.fetch_replay(match_id, str(out_dir))
            if fetched:
                dem_path = Path(fetched)
        if not dem_path.exists():
            print(f"[skip]   {match_id}: реплей не скачался")
            return False

        match = gem.parse(str(dem_path))
        tmp_path = cache_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(gem.to_json(match))
        tmp_path.replace(cache_path)

        if not keep_dem and dem_path.exists():
            dem_path.unlink()

        dt = time.monotonic() - t0
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"[ok]     {match_id}: {cache_path.name} ({size_mb:.1f} МБ, {dt:.0f} с)")
        return True
    except Exception as exc:
        print(f"[skip]   {match_id}: {exc!r}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор и кэширование реплеев Dota 2")
    parser.add_argument("--source", choices=tuple(_OPENDOTA_ENDPOINTS), default="pro",
                        help="эндпоинт OpenDota (по умолчанию: pro)")
    parser.add_argument("--limit", type=int, default=20,
                        help="сколько матчей собрать (по умолчанию: 20)")
    parser.add_argument("--match-ids", type=int, nargs="+",
                        help="явный список match_id вместо запроса к OpenDota")
    parser.add_argument("--out-dir", default="replays",
                        help="каталог для кэшей .json (по умолчанию: replays)")
    parser.add_argument("--max-age-days", type=float, default=12.0,
                        help="максимальный возраст матча в днях (реплеи живут ~14 дней)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="пауза между матчами в секундах")
    parser.add_argument("--keep-dem", action="store_true",
                        help="не удалять сырой .dem после создания .json кэша")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match_ids:
        match_ids = list(args.match_ids)
    else:
        print(f"Запрос свежих матчей из OpenDota ({args.source})...")
        match_ids = fetch_recent_match_ids(args.source, args.limit, args.max_age_days)
        print(f"Получено {len(match_ids)} match_id")

    ok = 0
    for idx, mid in enumerate(match_ids):
        if process_match(mid, out_dir, keep_dem=args.keep_dem):
            ok += 1
        if idx + 1 < len(match_ids) and args.delay > 0:
            time.sleep(args.delay)

    print(f"\nГотово: {ok}/{len(match_ids)} матчей в {out_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
