# -*- coding: utf-8 -*-
"""
Подбор текстовых промптов под типы нарушений.

Все кандидаты прогоняются ОДНИМ проходом SigLIP (эмбеддинги кадров считаются
один раз, промпты сравниваются с ними), поэтому перебор десятков формулировок
стоит секунды, а не минуты.

Ранжирование — по ЛОКАЛИЗАЦИИ (попадание пика в ±3 с от GT), а не по
разделению классов. Причина в LIMITATIONS.md: вердикт по кадру на этих данных
недостижим, а локализация работает и даёт критерий 2 ТЗ (15 баллов).

    python evalkit/prompt_search.py
    python evalkit/prompt_search.py --top 15
"""
from __future__ import annotations

import argparse
import math
import statistics as st
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

# Кандидаты сгруппированы по типам нарушений, которые видны в датасете.
CANDIDATES = {
    # --- уже используются, нужны как опорные точки ---
    "bag:handbag on desk": "a personal handbag lying on the desk",
    "phone:looking at phone": "a bank teller looking at a mobile phone in their hands",
    "drink:drinking from cup": "a bank teller drinking from a cup at the workplace",
    # --- отход от рабочего места ---
    "leave:empty chair": "an empty chair at the desk, nobody is sitting there",
    "leave:nobody at desk": "the workplace is empty, nobody is at the desk",
    "leave:walking away": "a bank teller standing up and walking away from the desk",
    "leave:leaving room": "a person walking out of the room through the door",
    "leave:back turned": "a person standing with their back to the desk",
    # --- посторонние ---
    "stranger:entering": "a person entering the room through the door",
    "stranger:behind counter": "a stranger standing behind the counter next to the teller",
    "stranger:two staff": "two employees together at one cash workplace",
    # --- деньги ---
    "money:counting": "a person counting paper banknotes in their hands",
    "money:on counter": "banknotes lying on the counter between two people",
    "money:handover": "two people passing money across the counter",
    "money:machine": "a person using a banknote counting machine",
    # --- инкассация ---
    "collect:bag handover": "a large collection bag being handed over the counter",
    "collect:green bag": "a person carrying a large sealed bag of valuables",
    # --- еда и напитки ---
    "food:on desk": "food or snack lying on the working desk",
    "food:cup on desk": "a paper coffee cup standing on the working desk",
    "food:eating": "a person eating at the working desk",
    # --- прочее ---
    "safe:open cabinet": "an open metal cabinet or safe with the door ajar",
    "safe:at cabinet": "a person opening a metal cabinet next to the desk",
    "doc:signing": "a person signing documents at the counter",
}


def softmax_first(xs: list, temp: float = 100.0) -> float:
    m = max(xs)
    e = [math.exp((x - m) * temp) for x in xs]
    return e[0] / sum(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()

    names = list(CANDIDATES)
    prompts = [CANDIDATES[k] for k in names] + NEUTRAL
    n = len(names)
    clips = load_clips()

    # video_id -> list[(peak_p, peak_t)] по каждому кандидату
    peaks: dict = {}
    for i, c in enumerate(clips, 1):
        det = detect_video(c.path)
        box = square_crop_around(estimate_cash_zone(det), det["width"], det["height"])
        r = score_video(c.path, prompts, crop_box=box)
        row_out = []
        for gi in range(n):
            bp, bt = 0.0, 0.0
            for row, t in zip(r["sims"], r["t"]):
                p = softmax_first([row[gi]] + row[n:n + len(NEUTRAL)])
                if p > bp:
                    bp, bt = p, t
            row_out.append((bp, bt))
        peaks[c.video_id] = row_out
        print(f"[{i:2d}/{len(clips)}] {c.video_id}", end="\r")
    print(" " * 40)

    n_viol = sum(1 for c in clips if c.is_violation)
    n_ev = sum(len(c.timestamps) for c in clips)
    rows = []
    for gi, key in enumerate(names):
        hits = sum(
            1
            for c in clips
            if c.is_violation
            and any(abs(peaks[c.video_id][gi][1] - g) <= args.tol for g in c.timestamps)
        )
        v = [peaks[c.video_id][gi][0] for c in clips if c.is_violation]
        o = [peaks[c.video_id][gi][0] for c in clips if not c.is_violation]
        mv, mo = sum(v) / len(v), sum(o) / len(o)
        allp = sorted(v + o)
        p75 = allp[int(len(allp) * 0.75)]
        best = (999, 0.0)
        for i in range(1, 60):
            th = allp[0] + (allp[-1] - allp[0]) * i / 60
            tp = sum(1 for c in clips if c.is_violation and peaks[c.video_id][gi][0] >= th)
            fp = sum(1 for c in clips if not c.is_violation and peaks[c.video_id][gi][0] >= th)
            best = min(best, (3 * (n_viol - tp) + fp, th))
        rows.append((hits, mv - mo, best[0], p75, key))

    rows.sort(key=lambda r: (-r[0], -r[1]))
    print(f"{'попад.±3с':>10}{'разрыв':>9}{'COST':>6}{'порог p75':>11}  кандидат")
    print("-" * 78)
    for hits, gap, cost, p75, key in rows[: args.top]:
        print(f"{hits:>6}/{n_viol}{gap:>+9.3f}{cost:>6}{p75:>11.4f}  {key}")
    print(f"\nвсего событий в разметке: {n_ev} | база 'всегда нарушение' COST 19")
    print("Порог p75 — готовое значение для поля clip.threshold в rules.yaml.")


if __name__ == "__main__":
    main()
