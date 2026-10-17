# -*- coding: utf-8 -*-
"""
Пакетная детекция по всему датасету с кэшированием на диск.

Запускается ОДИН раз. После этого подбор порогов в Шаге 3 идёт мгновенно:
rule engine читает кэш, а не гоняет нейросеть.

    python evalkit/extract_states.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vision import cache_path, detect_video  # noqa: E402
from evalkit.labels import load_clips  # noqa: E402


def main() -> None:
    clips = load_clips()
    t_start = time.time()
    done = cached = 0
    for i, clip in enumerate(clips, 1):
        if cache_path(clip.path).exists():
            cached += 1
            print(f"[{i:2d}/{len(clips)}] {clip.video_id:<8} кэш")
            continue
        t0 = time.time()
        r = detect_video(clip.path)
        done += 1
        n_det = sum(len(f["dets"]) for f in r["frames"])
        print(
            f"[{i:2d}/{len(clips)}] {clip.video_id:<8} "
            f"{time.time() - t0:5.1f}s  кадров {len(r['frames']):4d}  детекций {n_det:5d}"
        )
    print(f"\nГотово за {time.time() - t_start:.0f}s — обработано {done}, из кэша {cached}")


if __name__ == "__main__":
    main()
