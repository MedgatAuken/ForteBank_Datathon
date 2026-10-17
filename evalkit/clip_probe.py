# -*- coding: utf-8 -*-
"""
Проверка, разделяет ли zero-shot SigLIP классы — ДО того, как строить на нём правила.

Для каждой гипотезы задаётся КОНТРАСТНАЯ группа: один "нарушающий" промпт и
несколько нейтральных. По группе берётся softmax — это даёт откалиброванную
вероятность вместо бесполезных абсолютных косинусов.

Печатает по каждой гипотезе:
  * среднее max-p у нарушений и у чистых роликов,
  * лучший порог и COST,
  * попал ли пик в +-3 с от GT-таймкода (это уже про критерий 2 ТЗ).

    python evalkit/clip_probe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clipscore import score_video, square_crop_around  # noqa: E402
from app.scene import estimate_cash_zone  # noqa: E402
from app.vision import detect_video  # noqa: E402
from evalkit.labels import load_clips  # noqa: E402

NEUTRAL = [
    "a bank teller sitting and working at the desk",
    "a bank teller talking with a customer at the counter",
    "an ordinary office room",
]

GROUPS = {
    "phone": "a bank teller looking at a mobile phone in their hands",
    "drink": "a bank teller drinking from a cup at the workplace",
    "food": "food on the desk at the workplace",
    "bag": "a personal handbag lying on the desk",
    "empty": "an empty workplace, the chair is empty and nobody is there",
    "standing": "the bank teller is standing up and walking away from the desk",
    "handover": "two people passing money over the counter",
    "counting": "a person counting paper banknotes in their hands",
    "bigbag": "a large collection bag being handed over the counter",
}

PROMPTS = list(GROUPS.values()) + NEUTRAL


def softmax(xs: list[float], temp: float = 100.0) -> list[float]:
    m = max(xs)
    e = [math.exp((x - m) * temp) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def main() -> None:
    clips = load_clips()
    names = list(GROUPS)
    n_neg = len(NEUTRAL)

    # video_id -> {group: (max_p, t_at_max)}
    per_video: dict = {}
    for i, c in enumerate(clips, 1):
        det = detect_video(c.path)
        box = square_crop_around(
            estimate_cash_zone(det), det["width"], det["height"]
        )
        r = score_video(c.path, PROMPTS, crop_box=box)
        sims, ts = r["sims"], r["t"]
        res = {}
        for gi, g in enumerate(names):
            best_p, best_t = 0.0, 0.0
            for row, t in zip(sims, ts):
                group_vals = [row[gi]] + row[len(names):len(names) + n_neg]
                p = softmax(group_vals)[0]
                if p > best_p:
                    best_p, best_t = p, t
            res[g] = (best_p, best_t)
        per_video[c.video_id] = res
        print(f"[{i:2d}/{len(clips)}] {c.video_id}", end="\r")

    print(" " * 40)
    print(f"{'гипотеза':<12}{'нарушение':>11}{'без нар.':>11}{'разрыв':>9}"
          f"{'лучший COST':>13}{'порог':>8}{'пик в ±3с':>11}")
    print("-" * 75)

    for g in names:
        v = [per_video[c.video_id][g][0] for c in clips if c.is_violation]
        o = [per_video[c.video_id][g][0] for c in clips if not c.is_violation]
        mv, mo = sum(v) / len(v), sum(o) / len(o)

        best = (999, 0.0)
        for th in [i / 40 for i in range(1, 40)]:
            tp = sum(1 for c in clips if c.is_violation and per_video[c.video_id][g][0] >= th)
            fp = sum(1 for c in clips if not c.is_violation and per_video[c.video_id][g][0] >= th)
            fn = sum(1 for c in clips if c.is_violation and per_video[c.video_id][g][0] < th)
            best = min(best, (3 * fn + fp, th))

        hits = sum(
            1
            for c in clips
            if c.is_violation
            and any(abs(per_video[c.video_id][g][1] - gt) <= 3.0 for gt in c.timestamps)
        )
        n_viol = sum(1 for c in clips if c.is_violation)
        flag = "  <--" if mo > 0 and mv / mo > 1.25 else ""
        print(f"{g:<12}{mv:>11.3f}{mo:>11.3f}{mv - mo:>+9.3f}"
              f"{best[0]:>13}{best[1]:>8.2f}{hits:>7}/{n_viol}{flag}")

    print("\nбаза: 'всегда нарушение' COST 19 | 'всегда чисто' COST 63")


if __name__ == "__main__":
    main()
