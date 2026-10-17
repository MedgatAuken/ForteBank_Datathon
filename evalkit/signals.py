# -*- coding: utf-8 -*-
"""
Диагностика сигналов: что вообще различает "нарушение" и "без нарушения".

Печатает по каждому видео агрегаты вектора состояния рядом с GT-меткой и
считает разделяющую способность каждого сигнала. Пороги подбирать ПОСЛЕ
этого отчёта, а не до: правило, у которого распределения классов совпадают,
бесполезно и даст только ложные срабатывания.

    python evalkit/signals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scene import absence_intervals, build_scene, smooth  # noqa: E402
from app.vision import cache_path, detect_video  # noqa: E402
from evalkit.labels import load_clips  # noqa: E402


def aggregates(track) -> dict:
    gaps = absence_intervals(track)
    longest = max((b - a for a, b in gaps), default=0.0)
    n_frames = max(1, len(track.frames))
    return {
        "absent_max": longest,
        "absent_frac": sum(not f.cashier_present for f in track.frames) / n_frames,
        "bag_max": max((f.objects.get("bag", 0.0) for f in track.frames), default=0.0),
        "bag_frames": sum(f.objects.get("bag", 0.0) > 0.5 for f in track.frames),
        "food_max": max((f.objects.get("cup_or_food", 0.0) for f in track.frames), default=0.0),
        "food_frames": sum(f.objects.get("cup_or_food", 0.0) > 0.5 for f in track.frames),
        "phone_frames": sum(f.phone_in_hand for f in track.frames),
        "persons_max": max((f.persons_in_cash_zone for f in track.frames), default=0),
    }


def main() -> None:
    clips = load_clips()
    rows = []
    for c in clips:
        if not cache_path(c.path).exists():
            print(f"{c.video_id}: нет кэша, пропуск")
            continue
        track = smooth(build_scene(detect_video(c.path)))
        rows.append((c, aggregates(track)))

    keys = ["absent_max", "absent_frac", "bag_frames", "food_frames",
            "phone_frames", "persons_max"]
    print(f"{'ID':<7}{'GT':<6}" + "".join(f"{k:>13}" for k in keys))
    print("-" * (13 + 13 * len(keys)))
    for c, a in sorted(rows, key=lambda r: (not r[0].is_violation, r[0].video_id)):
        gt = "VIOL" if c.is_violation else "ok"
        vals = "".join(f"{a[k]:>13.2f}" for k in keys)
        print(f"{c.video_id:<7}{gt:<6}{vals}")

    print("\nРАЗДЕЛЯЮЩАЯ СПОСОБНОСТЬ (среднее по классам)")
    print(f"{'сигнал':<16}{'нарушение':>12}{'без нар.':>12}{'разрыв':>12}")
    print("-" * 52)
    for k in keys:
        v = [a[k] for c, a in rows if c.is_violation]
        o = [a[k] for c, a in rows if not c.is_violation]
        mv = sum(v) / len(v) if v else 0.0
        mo = sum(o) / len(o) if o else 0.0
        flag = "  <-- разделяет" if mo > 0 and mv / max(mo, 1e-6) > 1.6 else ""
        print(f"{k:<16}{mv:>12.2f}{mo:>12.2f}{mv - mo:>12.2f}{flag}")


if __name__ == "__main__":
    main()
