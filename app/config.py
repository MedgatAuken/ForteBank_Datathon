# -*- coding: utf-8 -*-
"""Настройки сервиса. Всё через переменные окружения — чтобы в Docker ничего не править."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Ключ для жюри. В проде ОБЯЗАТЕЛЬНО задать через env.
API_KEY = os.getenv("API_KEY", "dev-key")

# Лимит ТЗ — 5 минут на видео. Берём запас на загрузку и сериализацию ответа:
# пайплайн обязан уложиться в 240с и отдать то, что успел найти.
MAX_PROCESSING_SEC = float(os.getenv("MAX_PROCESSING_SEC", "240"))

# Одновременных анализов. На одной GPU больше 1 обычно вредно.
WORKERS = int(os.getenv("WORKERS", "1"))

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_DIR / "var" / "uploads"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "200"))
ALLOWED_SUFFIXES = {".mov", ".mp4", ".m4v"}

# Сколько кадров-доказательств отдавать на одно нарушение (ТЗ: 1–3).
EVIDENCE_FRAMES = int(os.getenv("EVIDENCE_FRAMES", "3"))
EVIDENCE_JPEG_QUALITY = int(os.getenv("EVIDENCE_JPEG_QUALITY", "85"))

RULES_PATH = Path(os.getenv("RULES_PATH", BASE_DIR / "rules.yaml"))

# Общий множитель порогов семантических правил. Сдвигает рабочую точку между
# "не пропустить" и "не ложно сработать" БЕЗ правки rules.yaml.
#   0.2  -> ни одного пропуска (FN 0), но вердикт "нарушение" почти всегда
#   1.0  -> лучшая accuracy 0.675, точность 0.75, но 9 пропусков
# Кривая целиком — в README, раздел "Рабочая точка".
CLIP_THRESHOLD_MULT = float(os.getenv("CLIP_THRESHOLD_MULT", "1.0"))

# Максимум нарушений на видео. Отбор — по ОТНОШЕНИЮ p/порог: сырые p между
# правилами несравнимы (у R4 порог 0.0004, у R9 — 0.995).
#
# ПРОВЕРЕНО: ограничение НЕ улучшает метрики. При лимите 2 и 3 точность
# события остаётся 0.39/0.38, а полнота падает с 13/22 до 12/22. Поэтому
# значение по умолчанию заведомо выше числа правил — отсечки фактически нет.
# Ручка оставлена на случай расширения набора правил.
MAX_VIOLATIONS = int(os.getenv("MAX_VIOLATIONS", "8"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
