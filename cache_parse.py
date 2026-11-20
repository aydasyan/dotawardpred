"""Кэширование результата парсинга в JSON, чтобы не парсить .dem повторно.

Запускать один раз: python cache_parse.py
Дальше во всех пробах: match = gem.load_json("replays/9097129353.json")
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import gem

match = gem.parse("replays/9097129353.dem")
with open("replays/9097129353.json", "w", encoding="utf-8") as fh:
    fh.write(gem.to_json(match))
print("cached OK")
