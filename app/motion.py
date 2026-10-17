# -*- coding: utf-8 -*-
"""
Признаки движения.

Единственное семейство, не покрытое детектором и CLIP: те отвечают на вопрос
"что в кадре", а разметка ForteBank отмечает ДЕЙСТВИЯ — передачу купюр, глоток,
отход от стола. У действия есть временная сигнатура, которой нет у статики.

Считается разностью соседних кадров на уменьшенной картинке — дёшево
(единицы секунд на ролик) и не требует весов. Оптический поток Фарнебака
доступен как более точный, но более медленный вариант.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from app.vision import CACHE_DIR, file_md5

MOTION_FPS = float(os.getenv("MOTION_FPS", "10"))
SMALL_W = int(os.getenv("MOTION_WIDTH", "128"))


def cache_path(video_path: Path, fps: float, zone_tag: str) -> Path:
    return CACHE_DIR / f"{file_md5(video_path)}_motion_{fps:g}_{zone_tag}.json"


def motion_series(
    video_path: Path,
    zone: Optional[list] = None,
    fps: float = MOTION_FPS,
    deadline=None,
    use_cache: bool = True,
) -> dict:
    """
    Возвращает {"t":[...], "zone":[...], "full":[...]}

    zone — средняя величина межкадровой разности внутри зоны кассы,
    full — то же по всему кадру (нужен как фон: дрожание камеры и общий
           трафик людей поднимают оба канала разом, и их можно вычесть).
    """
    import hashlib

    zone_tag = (
        hashlib.md5(str([round(v) for v in zone]).encode()).hexdigest()[:8]
        if zone
        else "full"
    )
    cache = cache_path(video_path, fps, zone_tag)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    step = max(1, int(round(src_fps / fps)))
    scale = SMALL_W / float(W)
    sh = max(1, int(H * scale))

    if zone:
        zx1 = max(0, int(zone[0] * scale))
        zy1 = max(0, int(zone[1] * scale))
        zx2 = min(SMALL_W, int(zone[2] * scale))
        zy2 = min(sh, int(zone[3] * scale))
        if zx2 <= zx1 or zy2 <= zy1:
            zone = None

    ts: list[float] = []
    m_zone: list[float] = []
    m_full: list[float] = []
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if deadline is not None and deadline.expired():
                break
            small = cv2.cvtColor(
                cv2.resize(frame, (SMALL_W, sh)), cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            if prev is not None:
                d = np.abs(small - prev)
                ts.append(round(idx / src_fps, 3))
                m_full.append(round(float(d.mean()) / 255.0, 5))
                m_zone.append(
                    round(float(d[zy1:zy2, zx1:zx2].mean()) / 255.0, 5)
                    if zone
                    else round(float(d.mean()) / 255.0, 5)
                )
            prev = small
        idx += 1
    cap.release()

    out = {"fps": fps, "t": ts, "zone": m_zone, "full": m_full}
    if use_cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out
