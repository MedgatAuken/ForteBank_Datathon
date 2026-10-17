# -*- coding: utf-8 -*-
"""
Зоны ручной проверки — увеличенные кадры тех участков, где решается вопрос.

ЗАЧЕМ. Замеры показали потолок: признаки, по которым размечен датасет, мельче
разрешающей способности видео (ключи в замке — 3x10 px, пачка купюр — 25x12 px,
см. LIMITATIONS.md). Поэтому сервис не притворяется, что решает за человека.
Он показывает проверяющему ровно ту зону и тот момент, которые надо посмотреть,
в максимальном увеличении — и решение занимает секунды вместо просмотра ролика.

Это прямо соответствует п.7 ограничений ТЗ: результат — подсказка для проверки
человеком, а не заключение.

ГЕОМЕТРИЯ, А НЕ КООРДИНАТЫ. Финальное видео снято под другим углом (FAQ №4),
поэтому зоны выводятся из сцены, а не зашиты:
  * стол      — крупнейшая связная область цвета дерева на фоновом кадре;
  * хранение  — полоса НАД столом, выведенная из его размеров. Отдельный
                детектор шкафа пробовался и врал (ловил стену), геометрия от
                стола попала в шкаф на всех проверенных ракурсах;
  * рабочее место — зона плотности детекций человека (app.scene).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

from app.desk import wood_mask
from app.vision import CACHE_DIR, file_md5

UPSCALE = 3.0
JPEG_QUALITY = 90


def _background(video_path: Path, n: int = 12):
    """Медианный кадр: камера статична, значит медиана = сцена без людей."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * k / n))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def scene_regions(video_path: Path, use_cache: bool = True) -> dict:
    """
    Возвращает {"desk": [x1,y1,x2,y2], "storage": [...], "workplace": [...]|None}
    в координатах исходного кадра.
    """
    cache = CACHE_DIR / f"{file_md5(video_path)}_regions.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import cv2

    bg = _background(video_path)
    if bg is None:
        return {}
    H, W = bg.shape[:2]

    mask = wood_mask(bg)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out: dict = {}
    if n >= 2:
        q = max(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_AREA])
        x1 = int(stats[q, cv2.CC_STAT_LEFT])
        y1 = int(stats[q, cv2.CC_STAT_TOP])
        x2 = x1 + int(stats[q, cv2.CC_STAT_WIDTH])
        y2 = y1 + int(stats[q, cv2.CC_STAT_HEIGHT])
        dh = max(1, y2 - y1)
        out["desk"] = [x1, y1, x2, y2]
        out["storage"] = [
            max(0, int(x1 - 0.10 * W)),
            max(0, int(y1 - 0.95 * dh)),
            min(W, int(x2 + 0.10 * W)),
            min(H, int(y1 + 0.25 * dh)),
        ]

    if use_cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


LABELS = {
    "desk": "Поверхность стола — не оставлены ли ценности "
            "(найденные купюры обведены)",
    "storage": "Зона хранения — закрыт ли шкаф, не забыты ли ключи в замке",
    "workplace": "Рабочее место кассира — руки, телефон, посторонние предметы",
}


def inspection_crops(
    video_path: Path,
    timestamp: float,
    regions: dict,
    upscale: float = UPSCALE,
    deadline=None,
    detections: Optional[dict] = None,
) -> list:
    """
    Увеличенные вырезы зон на момент события. Режем из НАТИВНОГО кадра.

    Если передать detections, на вырезе стола подсвечиваются найденные пачки
    купюр (app/banknote.py, точность 0.95). Сам по себе факт денег на столе
    нарушением НЕ является — п.20 прямо допускает наличные клиента на рабочем
    месте, — но проверяющему важно сразу видеть, где они лежат.
    """
    import cv2

    if not regions:
        return []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    idx = int(round(timestamp * fps))
    if total:
        idx = max(0, min(int(total) - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return []

    H, W = frame.shape[:2]

    notes = []
    if detections is not None:
        try:
            from app.banknote import boxes_at

            notes = boxes_at(video_path, detections, timestamp)
        except Exception:
            notes = []

    out = []
    for key in ("desk", "storage", "workplace"):
        box = regions.get(key)
        if not box:
            continue
        if deadline is not None and deadline.expired():
            break
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = frame[y1:y2, x1:x2].copy()
        if key == "desk" and notes:
            for nb in notes:
                cv2.rectangle(
                    crop,
                    (int(nb[0]) - x1 - 2, int(nb[1]) - y1 - 2),
                    (int(nb[2]) - x1 + 2, int(nb[3]) - y1 + 2),
                    (0, 215, 255), 1,
                )
        crop = cv2.resize(
            crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
        ok2, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok2:
            continue
        out.append(
            {
                "label": LABELS[key],
                "region": key,
                "timestamp_sec": round(idx / fps, 2),
                "image_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
            }
        )
    return out
