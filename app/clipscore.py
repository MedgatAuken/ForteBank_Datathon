# -*- coding: utf-8 -*-
"""
Zero-shot скоринг кадров текстовыми промптами (SigLIP).

Зачем, если уже есть детектор: COCO отвечает на вопрос "что лежит в кадре",
а разметка ForteBank отмечает "что произошло" — передачу купюр, глоток из
стакана, взгляд в телефон. Детектор объектов такие действия не различает
(замеры в README, Шаг 2).

Плюс это единственный способ закрыть FAQ №9 ТЗ: в тесте будут типы нарушений,
которых нет в обучающих данных. Классификатор на 21 примере их не возьмёт,
а zero-shot по текстовому промпту — возьмёт.

Лицензия SigLIP — Apache-2.0, модель работает локально. После первой загрузки
весов интернет не нужен (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from app import config
from app.vision import CACHE_DIR, file_md5

MODEL_ID = os.getenv("CLIP_MODEL", "google/siglip-base-patch16-224")
CLIP_FPS = float(os.getenv("CLIP_FPS", "2"))
CLIP_BATCH = int(os.getenv("CLIP_BATCH", "32"))

_model = None
_proc = None
_device = None


def _load():
    global _model, _proc, _device
    if _model is not None:
        return _model, _proc, _device
    import torch
    from transformers import AutoModel, AutoProcessor

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _proc = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModel.from_pretrained(MODEL_ID)
    _model.eval().to(_device)
    if _device.type == "cuda":
        _model.half()
    return _model, _proc, _device


def cache_path(video_path: Path, tag: str) -> Path:
    return CACHE_DIR / f"{file_md5(video_path)}_clip_{tag}.json"


def score_video(
    video_path: Path,
    prompts: list[str],
    fps: float = CLIP_FPS,
    deadline=None,
    crop: Optional[str] = None,
    crop_box: Optional[list] = None,
    use_cache: bool = True,
) -> dict:
    """
    Прогоняет видео и возвращает вероятности по каждому промпту на каждый кадр.

    crop=None       — весь кадр
    crop="desk"     — нижние 2/3 кадра
    crop_box=[...]  — явный прямоугольник. Главный режим: кадр 464x848
                      SigLIP ужимает в квадрат 224x224 с сильным искажением,
                      поэтому подаём КВАДРАТНЫЙ вырез вокруг зоны кассы.

    Возвращает {"fps":..., "prompts":[...], "t":[...], "sims":[[c1..cn], ...]}
    где sims — косинусная близость кадра к каждому промпту.
    """
    import hashlib

    tag = hashlib.md5(
        ("|".join(prompts) + f"|{fps}|{crop}|{crop_box}|{MODEL_ID}|v2cos").encode("utf-8")
    ).hexdigest()[:10]
    cache = cache_path(video_path, tag)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    model, proc, device = _load()

    with torch.inference_mode():
        txt = proc(text=prompts, padding="max_length", truncation=True, return_tensors="pt")
        txt = {k: v.to(device) for k, v in txt.items()}
        temb = model.get_text_features(**txt)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))

    ts: list[float] = []
    scores: list[list[float]] = []
    batch: list = []
    batch_t: list[float] = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode():
            px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            if device.type == "cuda":
                px = px.half()
            iemb = model.get_image_features(pixel_values=px)
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            # Храним СЫРЫЕ косинусы, а не вероятности. Абсолютная сигмоида
            # SigLIP на наших кадрах даёт ~0.001 и бесполезна как порог;
            # значение имеет динамика во времени и контраст между промптами.
            # Что с ними делать (softmax по группе / z-оценка) решает вызывающий.
            sims = (iemb @ temb.T).float()
        for t, row in zip(batch_t, sims.cpu().numpy()):
            ts.append(round(float(t), 3))
            scores.append([round(float(v), 4) for v in row])
        batch.clear()
        batch_t.clear()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if deadline is not None and deadline.expired():
                break
            img = frame
            if crop_box is not None:
                x1, y1, x2, y2 = (int(v) for v in crop_box)
                img = img[max(0, y1):y2, max(0, x1):x2]
                if img.size == 0:
                    img = frame
            elif crop == "desk":
                h = img.shape[0]
                img = img[h // 3:, :]
            batch.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            batch_t.append(idx / src_fps)
            if len(batch) >= CLIP_BATCH:
                flush()
        idx += 1
    flush()
    cap.release()

    out = {"fps": fps, "crop": crop, "prompts": prompts, "t": ts,
           "sims": scores, "kind": "cosine"}
    if use_cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def square_crop_around(zone: list, width: int, height: int, pad: float = 1.6) -> list:
    """
    Квадрат вокруг зоны кассы — чтобы SigLIP не искажал пропорции при ресайзе.
    Сторона берётся с запасом (pad), квадрат прижимается к границам кадра.
    """
    cx = (zone[0] + zone[2]) / 2.0
    cy = (zone[1] + zone[3]) / 2.0
    side = max(zone[2] - zone[0], zone[3] - zone[1]) * pad
    side = min(side, float(min(width, height)))
    half = side / 2.0
    cx = min(max(cx, half), width - half)
    cy = min(max(cy, half), height - half)
    return [cx - half, cy - half, cx + half, cy + half]
