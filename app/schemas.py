# -*- coding: utf-8 -*-
"""
Схемы ответа. Поля verdict / violations[].type / timecode_sec / description /
evidence_frames[].timestamp_sec / image_base64 заданы ТЗ ДОСЛОВНО — не переименовывать.

Поля rule и confidence — наши дополнительные. Лишние ключи в JSON стандартные
парсеры игнорируют, а нам они нужны для UI (п.7 сценария ТЗ: показать правило
и нормативный источник).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["violation", "no_violation"]
JobStatus = Literal["processing", "done", "failed"]


class RuleRef(BaseModel):
    """Нормативный источник. Пустой verified=False — значит пункт ещё не сверен на adilet."""
    id: str
    title: str
    document: str
    clause: str
    quote: str = ""          # дословная цитата пункта
    url: str = ""
    verified: bool = False


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class InspectionFrame(BaseModel):
    """
    Увеличенный вырез зоны для ручной проверки. Не из схемы ТЗ — наше
    дополнение: признаки мельче разрешения видео, поэтому решение остаётся
    за человеком, а сервис показывает, КУДА смотреть (см. LIMITATIONS.md).
    """
    label: str
    region: str
    timestamp_sec: float
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)
    # --- дополнительные поля (не из ТЗ) ---
    rule: Optional[RuleRef] = None
    confidence: Optional[float] = None
    inspection_frames: list[InspectionFrame] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Optional[Verdict] = None
    violations: list[Violation] = Field(default_factory=list)
    error: Optional[str] = None
    # диагностика, полезна на отладке
    processing_sec: Optional[float] = None
    truncated: bool = False          # True, если сработал watchdog
