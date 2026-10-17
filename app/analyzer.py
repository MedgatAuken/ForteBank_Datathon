# -*- coding: utf-8 -*-
"""
Рабочий анализатор: детекция -> зона кассы -> семантический скоринг -> правила.

Построен вокруг того, что по замерам РАБОТАЕТ. Классификация «нарушение или
нет» по кадру на этих данных недостижима (см. LIMITATIONS.md), а вот
ЛОКАЛИЗАЦИЯ момента работает: пик семантического скоринга попадает в ±3 с от
размеченного таймкода в 10–13 случаях из 21. Поэтому сервис построен как
подсказка проверяющему — «посмотрите этот фрагмент и вот почему», что и
требует п. 7 ограничений ТЗ.

Описания формируются по шаблону, а не генеративной моделью: шаблон не может
придумать того, чего нет на кадре. Критерий 4 штрафует неподтверждённые
утверждения.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from app import config
from app.clipscore import score_video, square_crop_around
from app.pipeline import Deadline, extract_frames_b64
from app.inspect import inspection_crops, scene_regions
from app.scene import estimate_cash_zone
from app.schemas import InspectionFrame, RuleRef, Violation
from app.vision import detect_video


def _load_rules() -> dict:
    import yaml

    with config.RULES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _softmax_first(xs: list, temp: float = 100.0) -> float:
    m = max(xs)
    e = [math.exp((x - m) * temp) for x in xs]
    return e[0] / sum(e)


def _evidence_timestamps(t: float, duration: float, n: int) -> list:
    """1–3 кадра вокруг события, всё внутри ролика."""
    out = []
    for cand in (t, t - 1.0, t + 1.0):
        v = max(0.0, min(duration - 0.05 if duration else cand, cand))
        if all(abs(v - u) > 0.3 for u in out):
            out.append(round(v, 2))
        if len(out) >= n:
            break
    return out


def _describe(rule: dict, t: float, p: float) -> str:
    """
    Шаблон, а не генеративная модель: шаблон не может придумать того, чего нет
    на кадре. Оговорка caveat из rules.yaml добавляется дословно — она несёт
    юридические ограничения трактовки (напр. п.12 допускает средства связи
    без камеры), и без неё вывод был бы неподтверждённым утверждением.
    """
    parts = [
        f"{rule['type']}. Признак наиболее выражен на {t:.1f} с записи "
        f"(уверенность {p:.2f}). Вывод сделан по видимому на кадре; "
        "требуется проверка человеком."
    ]
    caveat = " ".join((rule.get("caveat") or "").split())
    if caveat:
        parts.append(caveat)
    elif rule.get("weak"):
        # отдельная пометка нужна только если оговорка не задана явно
        parts.append("Трактовка нормы расширительная, требует проверки.")
    return " ".join(parts)


class RealAnalyzer:
    name = "rcnn+siglip+rules"

    def analyze(self, video_path: Path, deadline: Deadline, sink: list) -> str:
        deadline.check()
        spec = _load_rules()
        docs = spec.get("documents", {})
        neutral = spec.get("clip_neutral", [])
        clip_rules = [
            r for r in spec.get("rules", []) if r.get("enabled") and "clip" in r
        ]
        if not clip_rules or not neutral:
            return "no_violation"

        det = detect_video(video_path, deadline=deadline)
        deadline.check()
        duration = det.get("duration") or 0.0
        zone = estimate_cash_zone(det)
        box = square_crop_around(zone, det["width"], det["height"])

        # Зоны ручной проверки. Рабочее место берём из плотности детекций
        # человека, стол и хранение — из геометрии сцены.
        regions = scene_regions(video_path)
        regions["workplace"] = [int(v) for v in zone]

        prompts = [r["clip"]["prompt"] for r in clip_rules] + neutral
        scored = score_video(video_path, prompts, crop_box=box, deadline=deadline)
        sims, ts = scored["sims"], scored["t"]
        if not sims:
            return "no_violation"

        # Сначала считаем пики по всем правилам, потом отбираем лучшие —
        # иначе кадры-доказательства извлекались бы и для отброшенных.
        n_rules = len(clip_rules)
        hits = []
        for i, rule in enumerate(clip_rules):
            if deadline.expired():
                break  # отдаём то, что уже в sink
            best_p, best_t = 0.0, 0.0
            for row, t in zip(sims, ts):
                p = _softmax_first([row[i]] + row[n_rules:n_rules + len(neutral)])
                if p > best_p:
                    best_p, best_t = p, t
            thr = float(rule["clip"].get("threshold", 1.0)) * config.CLIP_THRESHOLD_MULT
            if best_p < thr:
                continue
            hits.append((best_p / thr if thr else 0.0, best_p, best_t, rule))

        hits.sort(key=lambda h: -h[0])
        for _, best_p, best_t, rule in hits[: config.MAX_VIOLATIONS]:
            if deadline.expired():
                break
            doc = docs.get(rule.get("document", ""), {})
            sink.append(
                Violation(
                    type=rule["type"],
                    timecode_sec=round(best_t, 1),
                    description=_describe(rule, best_t, best_p),
                    evidence_frames=extract_frames_b64(
                        video_path,
                        _evidence_timestamps(best_t, duration, config.EVIDENCE_FRAMES),
                    ),
                    confidence=round(best_p, 3),
                    inspection_frames=[
                        InspectionFrame(**c)
                        for c in inspection_crops(
                            video_path, best_t, regions,
                            deadline=deadline, detections=det,
                        )
                    ],
                    rule=RuleRef(
                        id=rule["id"],
                        title=(doc.get("title") or "").strip(),
                        document=doc.get("act", ""),
                        clause=rule.get("clause", ""),
                        quote=" ".join((rule.get("quote") or "").split()),
                        url=doc.get("url", ""),
                        verified=bool(rule.get("verified", False)),
                    ),
                )
            )

        sink.sort(key=lambda v: v.timecode_sec)
        return "violation" if sink else "no_violation"
