# -*- coding: utf-8 -*-
"""
Контракт пайплайна + заглушка.

Шаг 2 заменит StubAnalyzer на реальный пайплайн, НЕ ТРОГАЯ app/main.py и app/jobs.py.
Для этого достаточно реализовать один метод:

    def analyze(self, video_path, deadline, sink) -> str   # вернуть verdict

Два требования к реальной реализации:
  1. Регулярно проверять deadline.expired() и корректно выходить. Лимит ТЗ — 5 минут,
     превышение = видео не засчитано.
  2. Складывать найденные нарушения в sink ПО МЕРЕ ОБНАРУЖЕНИЯ, а не в конце.
     Тогда при срабатывании watchdog мы отдадим частичный результат вместо failed.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Protocol

from app import config
from app.schemas import EvidenceFrame, Violation


class Deadline:
    """Кооперативный таймер. Пайплайн обязан его опрашивать."""

    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds

    @property
    def remaining(self) -> float:
        return self._end - time.monotonic()

    def expired(self) -> bool:
        return self.remaining <= 0

    def check(self) -> None:
        if self.expired():
            raise TimeoutError("превышен бюджет обработки")


# --------------------------------------------------------------- кадры
def extract_frames_b64(
    video_path: Path,
    timestamps: list[float],
    quality: int = config.EVIDENCE_JPEG_QUALITY,
) -> list[EvidenceFrame]:
    """Достаёт кадры по таймкодам и кодирует в JPEG base64 (формат из ТЗ)."""
    import cv2  # локальный импорт: сервис должен подниматься даже без cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    out: list[EvidenceFrame] = []
    for ts in timestamps:
        idx = int(round(ts * fps))
        if n_frames and not (0 <= idx < n_frames):
            idx = max(0, min(int(n_frames) - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        out.append(
            EvidenceFrame(
                timestamp_sec=round(idx / fps, 2),
                image_base64=base64.b64encode(buf.tobytes()).decode("ascii"),
            )
        )
    cap.release()
    return out


def video_duration(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n) / fps if fps else 0.0


# --------------------------------------------------------------- контракт
class Analyzer(Protocol):
    name: str

    def analyze(self, video_path: Path, deadline: Deadline, sink: list[Violation]) -> str:
        ...


class StubAnalyzer:
    """
    Заглушка для Шага 1.

    ANALYZER=none (по умолчанию) — всегда no_violation. Нужна, чтобы поднять
        публичный URL и отдать его жюри задолго до готовности модели.
    ANALYZER=demo — выдаёт ОДНО синтетическое событие в середине ролика с
        настоящими кадрами. Нужна только чтобы проверить сквозную проводку
        API -> UI -> кадры. Это НЕ детекция, на eval её не гоняйте.
    """

    def __init__(self, mode: str = "none") -> None:
        self.mode = mode
        self.name = f"stub:{mode}"

    def analyze(self, video_path: Path, deadline: Deadline, sink: list[Violation]) -> str:
        deadline.check()
        if self.mode != "demo":
            return "no_violation"

        dur = video_duration(video_path)
        ts = round(dur / 2, 1) if dur else 1.0
        frames = extract_frames_b64(video_path, [max(0.0, ts - 1), ts, ts + 1][: config.EVIDENCE_FRAMES])
        sink.append(
            Violation(
                type="ЗАГЛУШКА — не результат детекции",
                timecode_sec=ts,
                description=(
                    "Синтетическое событие для проверки сквозной проводки "
                    "API -> UI -> кадры-доказательства. Реальный анализ не выполнялся."
                ),
                evidence_frames=frames,
                confidence=0.0,
            )
        )
        return "violation"


def load_analyzer() -> Analyzer:
    """
    Точка подмены анализатора.

    ANALYZER=real (по умолчанию) — детекция + правила.
    ANALYZER=none / demo        — заглушки Шага 1 (см. StubAnalyzer).
    """
    mode = os.getenv("ANALYZER", "real")
    if mode == "real":
        from app.analyzer import RealAnalyzer  # импорт здесь: torch тяжёлый

        return RealAnalyzer()
    return StubAnalyzer(mode)
