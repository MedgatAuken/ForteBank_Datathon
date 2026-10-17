# -*- coding: utf-8 -*-
"""
Закрывающий эксперимент: есть ли сигнал в КОМБИНАЦИИ признаков.

Поодиночке ни детектор, ни CLIP, ни движение классы не разделяют. Остаётся
проверить, не размазана ли информация по признакам так, что видна только
вместе. Если и здесь выйдет уровень случайного угадывания — значит визуально
доступной информации в этих данных действительно нет, и это надо писать в
"перечень ограничений" (раздел 8 ТЗ), а не продолжать перебор моделей.

Оценка — leave-one-out: на 40 роликах любой train/test split слишком шумный,
а LOO даёт честную оценку без подглядывания в тестовый объект.

    python evalkit/combine.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clipscore import score_video, square_crop_around  # noqa: E402
from app.motion import motion_series  # noqa: E402
from app.scene import build_scene, estimate_cash_zone, smooth  # noqa: E402
from app.vision import detect_video  # noqa: E402
from evalkit.clip_probe import GROUPS, NEUTRAL, PROMPTS, softmax  # noqa: E402
from evalkit.labels import load_clips  # noqa: E402

W_FN, W_FP = 3, 1


def features(clip) -> dict:
    det = detect_video(clip.path)
    zone = estimate_cash_zone(det)
    track = smooth(build_scene(det))
    n = max(1, len(track.frames))

    f = {
        "absent_frac": sum(not s.cashier_present for s in track.frames) / n,
        "persons_max": max((s.persons_in_cash_zone for s in track.frames), default=0),
        "bag_frames": sum(s.objects.get("bag", 0.0) > 0.5 for s in track.frames) / n,
        "food_frames": sum(s.objects.get("cup_or_food", 0.0) > 0.5 for s in track.frames) / n,
        "phone_frames": sum(s.phone_in_hand for s in track.frames) / n,
        "duration": det.get("duration", 0.0),
    }

    box = square_crop_around(zone, det["width"], det["height"])
    r = score_video(clip.path, PROMPTS, crop_box=box)
    names = list(GROUPS)
    nn = len(NEUTRAL)
    for gi, g in enumerate(names):
        ser = [softmax([row[gi]] + row[len(names):len(names) + nn])[0] for row in r["sims"]]
        med = st.median(ser)
        sd = st.pstdev(ser) or 1e-6
        f[f"clip_{g}_max"] = max(ser)
        f[f"clip_{g}_z"] = (max(ser) - med) / sd

    ms = motion_series(clip.path, zone=zone)
    z = ms["zone"] or [0.0]
    med = st.median(z)
    sd = st.pstdev(z) or 1e-6
    f["mot_mean"] = sum(z) / len(z)
    f["mot_zmax"] = (max(z) - med) / sd
    f["mot_spikes"] = sum(1 for v in z if (v - med) / sd > 2.5) / max(1, len(z))
    return f


def main() -> None:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clips = load_clips()
    rows, ys, ids = [], [], []
    for i, c in enumerate(clips, 1):
        f = features(c)
        rows.append(f)
        ys.append(1 if c.is_violation else 0)
        ids.append(c.video_id)
        print(f"[{i:2d}/{len(clips)}] {c.video_id}", end="\r")
    print(" " * 40)

    keys = sorted(rows[0])
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    y = np.array(ys)
    print(f"матрица признаков: {X.shape[0]} роликов x {X.shape[1]} признаков\n")

    models = {
        "логистическая регрессия (L2, C=0.3)":
            make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "случайный лес (200 деревьев, глубина 4)":
            RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
    }

    for name, model in models.items():
        preds = np.zeros(len(y), dtype=int)
        for tr, te in LeaveOneOut().split(X):
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        print(f"{name}")
        print(f"  LOO accuracy {(tp + tn) / len(y):.3f}   TP {tp} TN {tn} FP {fp} FN {fn}"
              f"   COST {W_FN * fn + W_FP * fp}")

    print("\nориентиры: случайное угадывание ~0.525 | 'всегда нарушение' COST 19")

    rf = RandomForestClassifier(n_estimators=400, max_depth=4, random_state=0).fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1][:8]
    print("\nсамые важные признаки (на ВСЕЙ выборке, только для интуиции):")
    for i in order:
        print(f"  {keys[i]:<20}{rf.feature_importances_[i]:.3f}")


if __name__ == "__main__":
    main()
