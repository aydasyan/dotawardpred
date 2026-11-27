"""Автоматический сбор датасета реплеев: OpenDota API -> .dem -> .json кэш.

Поток на каждый матч:
  1. Берём свежие match_id из OpenDota (/api/proMatches или /api/parsedMatches
     с пагинацией less_than_match_id), либо из --match-ids / --ids-file.
  2. При --shard i/N берём только i-ю долю списка (для распределения по ПК).
  3. Пропускаем уже закэшированные (<out_dir>/<match_id>.json).
  4. Скачиваем реплей через gem.fetch_replay, парсим через gem.parse и
     сохраняем кэш gem.to_json (параллельно при --workers > 1).
  5. По умолчанию удаляем сырой .dem после успешного кэширования (экономия
     сотен МБ на матч; флаг --keep-dem оставляет .dem).

Примеры:
    python collect_dataset.py --limit 100 --workers 4 --out-dir replays
    python collect_dataset.py --limit 300 --dump-ids match_ids.txt
    python collect_dataset.py --ids-file match_ids.txt --shard 0/2 --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_OPENDOTA_ENDPOINTS = {
    "pro": "https://api.opendota.com/api/proMatches",
    "parsed": "https://api.opendota.com/api/parsedMatches",
}


def fetch_recent_match_ids(source: str, limit: int, max_age_days: float,
                           page_delay: float = 1.0) -> List[int]:
    """Получить до `limit` свежих match_id из OpenDota API (с пагинацией)."""
    base_url = _OPENDOTA_ENDPOINTS[source]
    cutoff = time.time() - max_age_days * 86400.0
    ids: List[int] = []
    seen = set()
    less_than: Optional[int] = None

    for page in range(20):  # защита от бесконечной пагинации
        url = f"{base_url}?less_than_match_id={less_than}" if less_than else base_url
        req = urllib.request.Request(url, headers={"User-Agent": "DotaWardPrediction/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload:
            break

        min_mid: Optional[int] = None
        reached_cutoff = False
        for row in payload:
            mid = row.get("match_id")
            if mid is None:
                continue
            mid = int(mid)
            min_mid = mid if min_mid is None else min(min_mid, mid)
            start_time = row.get("start_time")
            if start_time is not None and float(start_time) < cutoff:
                reached_cutoff = True
                continue
            if mid in seen:
                continue
            seen.add(mid)
            ids.append(mid)
            if len(ids) >= limit:
                return ids

        if reached_cutoff or min_mid is None or min_mid == less_than:
            break
        less_than = min_mid
        if page_delay > 0:
            time.sleep(page_delay)

    return ids


def _parse_shard(spec: str) -> Tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Формат --shard: i/N (например, 0/2)")
    idx, total = int(parts[0]), int(parts[1])
    if total < 1 or not (0 <= idx < total):
        raise argparse.ArgumentTypeError("Требуется 0 <= i < N и N >= 1")
    return idx, total


def process_match(match_id: int, out_dir_str: str, keep_dem: bool = False) -> Tuple[int, bool, str]:
    """Скачать, распарсить и закэшировать один матч. Возвращает (id, ok, status_line)."""
    import gem  # локальный импорт: gem нужен только при фактическом запуске

    out_dir = Path(out_dir_str)
    cache_path = out_dir / f"{match_id}.json"
    if cache_path.exists():
        return match_id, True, f"[cached] {match_id}: уже есть {cache_path.name}"

    dem_path = out_dir / f"{match_id}.dem"
    t0 = time.monotonic()
    try:
        if not dem_path.exists():
            fetched = gem.fetch_replay(match_id, str(out_dir))
            if fetched:
                dem_path = Path(fetched)
        if not dem_path.exists():
            return match_id, False, f"[skip]   {match_id}: реплей не скачался"

        match = gem.parse(str(dem_path))
        tmp_path = cache_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(gem.to_json(match))
        tmp_path.replace(cache_path)

        if not keep_dem and dem_path.exists():
            dem_path.unlink()

        dt = time.monotonic() - t0
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        return match_id, True, f"[ok]     {match_id}: {cache_path.name} ({size_mb:.1f} МБ, {dt:.0f} с)"
    except Exception as exc:
        if not keep_dem and dem_path.exists():
            try:
                dem_path.unlink()
            except OSError:
                pass
        return match_id, False, f"[skip]   {match_id}: {exc!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор и кэширование реплеев Dota 2")
    parser.add_argument("--source", choices=tuple(_OPENDOTA_ENDPOINTS), default="pro",
                        help="эндпоинт OpenDota (по умолчанию: pro)")
    parser.add_argument("--limit", type=int, default=20,
                        help="сколько матчей собрать (по умолчанию: 20)")
    parser.add_argument("--match-ids", type=int, nargs="+",
                        help="явный список match_id вместо запроса к OpenDota")
    parser.add_argument("--ids-file", default=None,
                        help="прочитать список match_id из файла (по одному в строке)")
    parser.add_argument("--dump-ids", default=None,
                        help="сохранить полученный список match_id в файл и выйти")
    parser.add_argument("--shard", type=_parse_shard, default=None,
                        help="обработать только долю i/N списка (например, 0/2 и 1/2 на двух ПК)")
    parser.add_argument("--out-dir", default="replays",
                        help="каталог для кэшей .json (по умолчанию: replays)")
    parser.add_argument("--max-age-days", type=float, default=12.0,
                        help="максимальный возраст матча в днях (реплеи живут ~14 дней)")
    parser.add_argument("--workers", type=int, default=1,
                        help="число параллельных процессов парсинга (по умолчанию: 1)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="пауза между запуском задач в секундах (по умолчанию: 1.0)")
    parser.add_argument("--keep-dem", action="store_true",
                        help="не удалять сырой .dem после создания .json кэша")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match_ids:
        match_ids = list(args.match_ids)
    elif args.ids_file:
        with open(args.ids_file, encoding="utf-8") as fh:
            match_ids = [int(line.strip()) for line in fh if line.strip() and not line.startswith("#")]
    else:
        print(f"Запрос до {args.limit} свежих матчей из OpenDota ({args.source})...", flush=True)
        match_ids = fetch_recent_match_ids(args.source, args.limit, args.max_age_days)
        print(f"Получено {len(match_ids)} match_id", flush=True)

    if args.dump_ids:
        Path(args.dump_ids).write_text("\n".join(map(str, match_ids)) + "\n", encoding="utf-8")
        print(f"Сохранено {len(match_ids)} id в {args.dump_ids}")
        return 0

    if args.shard is not None:
        idx, total = args.shard
        match_ids = [mid for i, mid in enumerate(match_ids) if i % total == idx]
        print(f"Шард {idx}/{total}: {len(match_ids)} матчей", flush=True)

    if not match_ids:
        print("Нет матчей для обработки")
        return 1

    t_start = time.monotonic()
    ok = 0
    done = 0
    total_n = len(match_ids)

    if args.workers > 1:
        print(f"Запуск в {args.workers} воркеров (всего {total_n} матчей)...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for i, mid in enumerate(match_ids):
                futures.append(pool.submit(process_match, mid, str(out_dir), args.keep_dem))
                # Небольшой сдвиг старта первых воркеров, чтобы не бить одновременно в API
                if i < args.workers and args.delay > 0:
                    time.sleep(args.delay)
            for fut in as_completed(futures):
                _mid, success, msg = fut.result()
                done += 1
                if success:
                    ok += 1
                elapsed = (time.monotonic() - t_start) / 60.0
                print(f"[{done}/{total_n} | ok={ok} | {elapsed:.1f}м] {msg}", flush=True)
    else:
        for i, mid in enumerate(match_ids):
            _mid, success, msg = process_match(mid, str(out_dir), args.keep_dem)
            done += 1
            if success:
                ok += 1
            elapsed = (time.monotonic() - t_start) / 60.0
            print(f"[{done}/{total_n} | ok={ok} | {elapsed:.1f}м] {msg}", flush=True)
            if i + 1 < total_n and args.delay > 0:
                time.sleep(args.delay)

    total_min = (time.monotonic() - t_start) / 60.0
    print(f"\nГотово: {ok}/{total_n} матчей в {out_dir} за {total_min:.1f} мин", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
