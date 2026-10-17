# -*- coding: utf-8 -*-
"""
Детектор предметов, оставленных на поверхности стола.

Зачем отдельный модуль: в COCO нет класса "банкноты", а именно пачка купюр на
столе отличает X-26 (нарушение) от X-29 (без нарушения) — см. README, Шаг 2г
и evidence/money_on_desk_X26_vs_X29.jpg. Это прямая конъюнкция из п.20
Правил НБРК №231: ценности должны лежать в запираемом устройстве.

Почему не по цвету: замеры показали, что купюры и столешница практически
неразличимы по цвету (H 15.2 / S 69 / V 203 против H 15.7 / S 77 / V 170).
Поэтому работаем вычитанием фона: камера статична, стол статичен, а пачка
появляется и лежит.

Три вещи, без которых не работает:
  1. НАТИВНОЕ разрешение. Пачка — примерно 25x12 px в кадре 464x848;
     на уменьшенной картинке она растворяется.
  2. Маска людей боксами детектора: движущийся человек даёт огромный
     фон-форграунд и забивает мелкие блобы (разрыв между классами вырос
     с +0.03 до +0.13 только за счёт этого).
  3. Маска столешницы по дереву: стена и пол дают ложные блобы.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Дерево столешницы в HSV (OpenCV: H 0..179). Замерено на кадрах датасета.
WOOD_H = (8, 24)
WOOD_S = (40, 130)
WOOD_V_MIN = 110

DIFF_THRESH = 30          # порог отличия от фона, уровни яркости
BLOB_MIN, BLOB_MAX = 40, 1800     # площадь в px при нативном разрешении
PERSON_PAD = 8            # расширение бокса человека, px


def wood_mask(frame_bgr):
    """Маска столешницы по цвету дерева. Заменяет зашитый y-диапазон."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (
        (h >= WOOD_H[0]) & (h <= WOOD_H[1])
        & (s >= WOOD_S[0]) & (s <= WOOD_S[1])
        & (v >= WOOD_V_MIN)
    ).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def analyze(
    video_path: Path,
    detections: dict,
    fps: float = 3.0,
    deadline=None,
) -> list:
    """
    Возвращает [(t, площадь крупнейшего оставленного предмета), ...].

    detections — результат app.vision.detect_video (нужны боксы людей).
    """
    import cv2
    import numpy as np

    person_by_t = {
        round(f["t"], 2): [d["b"] for d in f["dets"] if d["g"] == "person"]
        for f in detections["frames"]
    }
    keys = sorted(person_by_t)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))

    frames, ts = [], []
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if deadline is not None and deadline.expired():
                break
            frames.append(fr)
            ts.append(idx / src_fps)
        idx += 1
    cap.release()
    if len(frames) < 6:
        return []

    stack = np.stack(frames).astype(np.float32)
    bg = np.median(stack, axis=0)
    desk = wood_mask(bg.astype(np.uint8))

    out = []
    for fr, t in zip(frames, ts):
        diff = np.abs(fr.astype(np.float32) - bg).max(axis=2)
        m = ((diff > DIFF_THRESH) & (desk > 0)).astype(np.uint8)

        if keys:
            k = min(keys, key=lambda v: abs(v - t))
            for b in person_by_t[k]:
                x1 = max(0, int(b[0]) - PERSON_PAD)
                y1 = max(0, int(b[1]) - PERSON_PAD)
                x2 = min(m.shape[1], int(b[2]) + PERSON_PAD)
                y2 = min(m.shape[0], int(b[3]) + PERSON_PAD)
                m[y1:y2, x1:x2] = 0

        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stt, _ = cv2.connectedComponentsWithStats(m, 8)
        best = 0
        for q in range(1, n):
            a = int(stt[q, cv2.CC_STAT_AREA])
            if BLOB_MIN <= a <= BLOB_MAX:
                w = int(stt[q, cv2.CC_STAT_WIDTH])
                h = int(stt[q, cv2.CC_STAT_HEIGHT])
                if h > 0 and 0.8 <= w / h <= 5.0:     # плоский предмет на столе
                    best = max(best, a)
        out.append((round(t, 2), best))
    return out
