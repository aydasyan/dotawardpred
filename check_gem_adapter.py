"""Сквозная проверка gem-пути на реальном реплее: MatchEvents -> признаки."""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from wardpred import FEATURE_COLS
from wardpred.features import build_match_features
from wardpred.gem_loader import load_match_gem

matches = load_match_gem("replays/9097129353.json")
print("перспектив:", len(matches))
for m in matches:
    print(f"  match_id={m.match_id} intervals={len(m.intervals)} obs={len(m.obs)} "
          f"purchases={len(m.purchases)} deaths={len(m.deaths)} buildings={len(m.buildings)}")
    assert m.intervals["visible"].isin([0, 1]).all()

    frame = build_match_features(m)
    print(f"  снэпшотов: {len(frame)}, непустых клеток: {frame['cell'].nunique()}")
    if len(frame):
        feat = frame[FEATURE_COLS]
        assert not feat.isna().any().any(), "NaN в признаках!"
        print("  NaN в признаках: нет")
        tu = frame["time_unseen"]
        print(f"  time_unseen: min={tu.min():.0f} median={tu.median():.0f} max={tu.max():.0f}")
        seen = (frame["last_x"] != -1.0).mean()
        print(f"  доля снэпшотов с известной позицией вардера: {seen:.1%}")
        print(f"  game_time: {frame['game_time'].min():.0f}..{frame['game_time'].max():.0f}")
        print(f"  ward_x range: {frame['ward_x'].min():.0f}..{frame['ward_x'].max():.0f}")
        print(f"  ward_y range: {frame['ward_y'].min():.0f}..{frame['ward_y'].max():.0f}")
        bought = frame["obs_bought_120s"]
        print(f"  obs_bought_120s>0: {(bought > 0).mean():.1%}")
        assert frame["cell"].between(0, 63).all()

print("\nGEM ADAPTER OK")
