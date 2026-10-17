# -*- coding: utf-8 -*-
"""
Детекция объектов. Faster R-CNN из torchvision.

Почему torchvision, а не YOLOv8/v11:
  Ultralytics распространяется под AGPL-3.0 — копилефт, несовместимый с
  требованием ТЗ "open source с лицензией, разрешающей коммерческое
  использование" (FAQ №24) и с п.34 о правах ForteBank.
  torchvision — BSD-3, весы COCO тоже свободные. Плюс никаких новых
  зависимостей: torch уже стоит.

Результат детекции КЭШИРУЕТСЯ на диск по md5 видео. Это критично для Шага 3:
подбор порогов не должен каждый раз гонять нейросеть заново.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from app import config

# --- COCO-классы torchvision, которые нам нужны ---
CLS_PERSON = 1
CLS_BAG = {27, 31, 33}           # backpack, handbag, suitcase
CLS_PHONE = {77}                 # cell phone
CLS_CUP_FOOD = {44, 46, 47, 51, 52, 53, 54, 59, 60}  # bottle/glass/cup/bowl/еда
CLS_LAPTOP = {73}                # якорь стола
CLS_KEEP = {CLS_PERSON} | CLS_BAG | CLS_PHONE | CLS_CUP_FOOD | CLS_LAPTOP

GROUP_OF_CLASS: dict[int, str] = {}
for _c in [CLS_PERSON]:
    GROUP_OF_CLASS[_c] = "person"
for _c in CLS_BAG:
    GROUP_OF_CLASS[_c] = "bag"
for _c in CLS_PHONE:
    GROUP_OF_CLASS[_c] = "phone"
for _c in CLS_CUP_FOOD:
    GROUP_OF_CLASS[_c] = "cup_or_food"
for _c in CLS_LAPTOP:
    GROUP_OF_CLASS[_c] = "laptop"

CACHE_DIR = Path(os.getenv("CACHE_DIR", config.PROJECT_DIR / "var" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_FPS = float(os.getenv("ANALYSIS_FPS", "3"))
BATCH = int(os.getenv("DETECT_BATCH", "8"))
SCORE_MIN = float(os.getenv("SCORE_MIN", "0.30"))

_model = None
_device = None


def _load_model():
    """Ленивая загрузка. Весы берутся из кэша torch (offline после первого раза)."""
    global _model, _device
    if _model is not None:
        return _model, _device
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    _model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_MIN)
    _model.eval().to(_device)
    if _device.type == "cuda":
        _model.half()
    return _model, _device


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(video_path: Path, fps: float = ANALYSIS_FPS) -> Path:
    return CACHE_DIR / f"{file_md5(video_path)}_{fps:g}.json"


def detect_video(
    video_path: Path,
    fps: float = ANALYSIS_FPS,
    deadline=None,
    use_cache: bool = True,
) -> dict:
    """
    Прогоняет видео через детектор с частотой fps.

    Возвращает {"fps":..., "duration":..., "width":..., "height":...,
                "frames":[{"t":float, "dets":[{"g":group,"s":score,"b":[x1,y1,x2,y2]}]}]}
    """
    cache = cache_path(video_path, fps)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import cv2
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = n_frames / src_fps if src_fps else 0.0

    step = max(1, int(round(src_fps / fps)))
    model, device = _load_model()

    frames_out: list[dict] = []
    batch_imgs: list = []
    batch_ts: list[float] = []

    def flush() -> None:
        if not batch_imgs:
            return
        with torch.inference_mode():
            outs = model(batch_imgs)
        for ts, out in zip(batch_ts, outs):
            dets = []
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                li = int(l)
                if li not in CLS_KEEP or s < SCORE_MIN:
                    continue
                dets.append({
                    "g": GROUP_OF_CLASS[li],
                    "s": round(float(s), 3),
                    "b": [round(float(v), 1) for v in b],
                })
            frames_out.append({"t": round(ts, 3), "dets": dets})
        batch_imgs.clear()
        batch_ts.clear()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if deadline is not None and deadline.expired():
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            t = (t.half() if device.type == "cuda" else t.float()) / 255.0
            batch_imgs.append(t.to(device))
            batch_ts.append(idx / src_fps)
            if len(batch_imgs) >= BATCH:
                flush()
        idx += 1
    flush()
    cap.release()

    result = {
        "fps": fps,
        "src_fps": src_fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frames": frames_out,
    }
    if use_cache:
        cache.write_text(json.dumps(result), encoding="utf-8")
    return result
