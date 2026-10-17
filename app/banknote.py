# -*- coding: utf-8 -*-
"""
Детектор банкнот на столе: кандидаты от app/desk.py + обученный классификатор.

В COCO класса «банкноты» нет, а попиксельно пачка купюр неотличима от
столешницы (замер: H 15.2/S 69/V 203 против H 15.7/S 77/V 170). Но на уровне
ПАТЧА купюры различимы — форма, бандероль, характерный цвет. Поэтому связка:
вычитание фона даёт кандидатов, маленькая CNN решает, купюры это или нет.

Качество классификатора (leave-one-video-out, evalkit/train_banknote.py):
точность 0.95, полнота 0.72.

Модель: var/banknote_cnn.pt. Если файла нет — детектор молча отключается,
сервис продолжает работать без этого сигнала.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from app.desk import DIFF_THRESH, PERSON_PAD, wood_mask
from app.vision import CACHE_DIR, file_md5

MODEL_PATH = Path(
    os.getenv("BANKNOTE_MODEL", Path(__file__).resolve().parent.parent / "var" / "banknote_cnn.pt")
)
FPS = float(os.getenv("BANKNOTE_FPS", "2"))
PROB_MIN = float(os.getenv("BANKNOTE_PROB", "0.5"))
AREA_MIN, AREA_MAX = 40, 1800

_model = None
_size = 48
_dev = None
_loaded = False


def _load():
    global _model, _size, _dev, _loaded
    if _loaded:
        return _model
    _loaded = True
    if not MODEL_PATH.exists():
        return None
    import torch

    from evalkit.train_banknote import build_model

    ck = torch.load(MODEL_PATH, map_location="cpu")
    _size = int(ck.get("size", 48))
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model()
    m.load_state_dict(ck["state"])
    m.eval().to(_dev)
    _model = m
    return _model


def available() -> bool:
    return _load() is not None


def boxes_at(video_path: Path, detections: dict, t: float) -> list:
    """Боксы купюр на ОДНОМ кадре. Нужно, чтобы подсветить находку проверяющему."""
    model = _load()
    if model is None:
        return []

    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    frames = []
    for k in range(12):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * k / 12))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    idx = int(round(t * src_fps))
    if total:
        idx = max(0, min(int(total) - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or len(frames) < 6:
        return []

    bg = np.median(np.stack(frames).astype(np.float32), axis=0)
    desk = wood_mask(bg.astype(np.uint8))
    H, W = desk.shape
    diff = np.abs(frame.astype(np.float32) - bg).max(axis=2)
    m = ((diff > DIFF_THRESH) & (desk > 0)).astype(np.uint8)
    for f in detections["frames"]:
        if abs(f["t"] - t) <= 0.5:
            for d in f["dets"]:
                if d["g"] != "person":
                    continue
                b = d["b"]
                m[max(0, int(b[1]) - PERSON_PAD):min(H, int(b[3]) + PERSON_PAD),
                  max(0, int(b[0]) - PERSON_PAD):min(W, int(b[2]) + PERSON_PAD)] = 0
            break
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stt, _ = cv2.connectedComponentsWithStats(m, 8)

    cand, crops = [], []
    for q in range(1, n):
        a = int(stt[q, cv2.CC_STAT_AREA])
        if not (AREA_MIN <= a <= AREA_MAX):
            continue
        x, y = int(stt[q, cv2.CC_STAT_LEFT]), int(stt[q, cv2.CC_STAT_TOP])
        w, h = int(stt[q, cv2.CC_STAT_WIDTH]), int(stt[q, cv2.CC_STAT_HEIGHT])
        if h <= 0 or not (0.8 <= w / h <= 5.0):
            continue
        pad = 4
        c = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if not c.size:
            continue
        cand.append([x, y, x + w, y + h])
        crops.append(cv2.resize(c, (_size, _size), interpolation=cv2.INTER_AREA))
    if not crops:
        return []
    arr = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops])
    xt = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(_dev)
    with torch.inference_mode():
        probs = torch.softmax(model(xt), dim=1)[:, 1].cpu().numpy()
    return [b for b, p in zip(cand, probs) if p >= PROB_MIN]


def detect(
    video_path: Path,
    detections: dict,
    fps: float = FPS,
    deadline=None,
    use_cache: bool = True,
) -> list:
    """
    Возвращает [(t, max_prob, n_notes), ...] — по кадрам с шагом 1/fps.

    n_notes считает только кандидатов с вероятностью выше PROB_MIN и вне
    боксов людей: пачка в руках кассира — не «оставленные ценности».
    """
    model = _load()
    if model is None:
        return []

    cache = CACHE_DIR / f"{file_md5(video_path)}_banknote_{fps:g}.json"
    if use_cache and cache.exists():
        return [tuple(r) for r in json.loads(cache.read_text(encoding="utf-8"))]

    import cv2
    import numpy as np
    import torch

    person_by_t = {
        round(f["t"], 2): [d["b"] for d in f["dets"] if d["g"] == "person"]
        for f in detections["frames"]
    }
    keys = sorted(person_by_t)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    frames, ts, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            if deadline is not None and deadline.expired():
                break
            frames.append(fr)
            ts.append(i / src_fps)
        i += 1
    cap.release()
    if len(frames) < 6:
        return []

    bg = np.median(np.stack(frames).astype(np.float32), axis=0)
    desk = wood_mask(bg.astype(np.uint8))
    H, W = desk.shape

    out = []
    for fr, t in zip(frames, ts):
        diff = np.abs(fr.astype(np.float32) - bg).max(axis=2)
        m = ((diff > DIFF_THRESH) & (desk > 0)).astype(np.uint8)
        if keys:
            for b in person_by_t[min(keys, key=lambda v: abs(v - t))]:
                m[max(0, int(b[1]) - PERSON_PAD):min(H, int(b[3]) + PERSON_PAD),
                  max(0, int(b[0]) - PERSON_PAD):min(W, int(b[2]) + PERSON_PAD)] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stt, _ = cv2.connectedComponentsWithStats(m, 8)

        crops = []
        for q in range(1, n):
            a = int(stt[q, cv2.CC_STAT_AREA])
            if not (AREA_MIN <= a <= AREA_MAX):
                continue
            x, y = int(stt[q, cv2.CC_STAT_LEFT]), int(stt[q, cv2.CC_STAT_TOP])
            w, h = int(stt[q, cv2.CC_STAT_WIDTH]), int(stt[q, cv2.CC_STAT_HEIGHT])
            if h <= 0 or not (0.8 <= w / h <= 5.0):
                continue
            pad = 4
            c = fr[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            if c.size:
                crops.append(cv2.resize(c, (_size, _size), interpolation=cv2.INTER_AREA))

        if not crops:
            out.append((round(t, 2), 0.0, 0))
            continue
        arr = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops])
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(_dev)
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        out.append((round(t, 2), round(float(probs.max()), 3),
                    int((probs >= PROB_MIN).sum())))

    if use_cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out
