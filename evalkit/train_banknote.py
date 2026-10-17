# -*- coding: utf-8 -*-
"""
Обучение классификатора патчей «банкноты / не банкноты».

ЗАЧЕМ. В COCO нет класса «банкноты», поэтому правила R1/R8 («ранее принятые
наличные не убраны в запираемое устройство») были выключены — нечем увидеть
сам объект. Детектор предметов на столе (app/desk.py) предлагает кандидатов,
но не знает, что лежит: купюры, ноутбук, стакан, сумка. Классификатор
закрывает ровно этот пробел.

ДАННЫЕ. 29 размеченных вручную положительных патчей — мало, поэтому к ним
добавляется синтетика: настоящие пиксели купюр вклеиваются на настоящие
столешницы из роликов со случайным масштабом, поворотом, яркостью, размытием.
Ключевое — всё в НАТИВНОМ разрешении: модель учится видеть пятно 25x12 px,
а не чёткое фото банкноты.

ОЦЕНКА — leave-one-video-out. Патчи одного ролика это почти соседние кадры,
случайный сплит завысил бы точность.

    python evalkit/train_banknote.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PATCH_DIR = Path(__file__).resolve().parent.parent.parent / "var_look" / "patches"
MODEL_OUT = Path(__file__).resolve().parent.parent / "var" / "banknote_cnn.pt"
SIZE = 48
SEED = 0
EPOCHS = 12


def desk_backgrounds(n_per_video: int = 6) -> list:
    """Чистые куски столешницы из всех роликов — фон для вклейки и негативы."""
    import cv2
    import numpy as np

    from app.desk import wood_mask
    from evalkit.labels import load_clips

    out = []
    for clip in load_clips():
        cap = cv2.VideoCapture(str(clip.path))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        frames = []
        for k in range(n_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * k / max(1, n_per_video)))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
        if not frames:
            continue
        bg = np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
        mask = wood_mask(bg)
        ys, xs = np.where(mask > 0)
        if len(xs) < 500:
            continue
        for _ in range(n_per_video):
            i = random.randrange(len(xs))
            cx, cy = int(xs[i]), int(ys[i])
            h = random.randint(14, 34)
            w = int(h * random.uniform(1.4, 3.0))
            x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
            crop = bg[y1:y1 + h, x1:x1 + w]
            if crop.shape[0] >= 8 and crop.shape[1] >= 8:
                out.append(crop)
    return out


def synth_positives(positives: list, backgrounds: list, n: int) -> list:
    """Вклейка реальных купюр на реальные столешницы со случайными вариациями."""
    import cv2
    import numpy as np

    out = []
    if not positives or not backgrounds:
        return out
    for _ in range(n):
        note = random.choice(positives).copy()
        bg = random.choice(backgrounds).copy()

        ang = random.uniform(-18, 18)
        scale = random.uniform(0.7, 1.35)
        h, w = note.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        note = cv2.warpAffine(note, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        note = np.clip(
            note.astype(np.float32) * random.uniform(0.75, 1.25)
            + random.uniform(-18, 18),
            0, 255,
        ).astype(np.uint8)
        if random.random() < 0.5:
            k = random.choice([3, 5])
            note = cv2.GaussianBlur(note, (k, k), 0)

        bh = max(h + 8, int(h * random.uniform(1.3, 2.0)))
        bw = max(w + 8, int(w * random.uniform(1.3, 2.0)))
        bg = cv2.resize(bg, (bw, bh))
        oy = random.randint(0, bh - h)
        ox = random.randint(0, bw - w)
        # мягкий край, чтобы модель не училась на шве вклейки
        alpha = cv2.GaussianBlur(np.ones((h, w), np.float32), (5, 5), 0)[..., None]
        region = bg[oy:oy + h, ox:ox + w].astype(np.float32)
        bg[oy:oy + h, ox:ox + w] = (
            note.astype(np.float32) * alpha + region * (1 - alpha)
        ).astype(np.uint8)
        out.append(bg)
    return out


def to_tensor(imgs: list):
    import cv2
    import numpy as np
    import torch

    arr = []
    for im in imgs:
        r = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        arr.append(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    x = np.stack(arr).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def build_model():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        blk(3, 16), blk(16, 32), blk(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(64, 2),
    )


def fit(pos_tr: list, neg_tr: list, bgs: list, dev, epochs: int = EPOCHS, seed: int = SEED):
    import torch
    import torch.nn as nn

    random.seed(seed)
    torch.manual_seed(seed)
    synth = synth_positives(pos_tr, bgs, n=900)
    X = to_tensor(pos_tr * 8 + synth + neg_tr * 2 + bgs).to(dev)
    y = torch.tensor(
        [1] * (len(pos_tr) * 8 + len(synth)) + [0] * (len(neg_tr) * 2 + len(bgs))
    ).to(dev)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 64):
            idx = perm[k:k + 64]
            xb = X[idx]
            if random.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            lossf(model(xb), y[idx]).backward()
            opt.step()
    model.eval()
    return model


def main() -> None:
    import cv2
    import torch

    from evalkit.banknote_labels import split

    random.seed(SEED)
    index = json.loads((PATCH_DIR / "index.json").read_text(encoding="utf-8"))
    pos_ids, neg_ids = split(index)

    def load_many(ids):
        out = []
        for i in ids:
            im = cv2.imread(str(PATCH_DIR / f"{i}.png"))
            if im is not None and im.size:
                out.append(im)
        return out

    print(f"размечено вручную: {len(pos_ids)} купюр, {len(neg_ids)} не-купюр")
    print("собираю фоны столешниц...")
    bgs = desk_backgrounds()
    print(f"фонов: {len(bgs)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_vids = sorted({i.split("_")[0] for i in pos_ids})
    print(f"\nкросс-валидация leave-one-video-out, роликов с купюрами: {len(pos_vids)}")

    TP = FP = FN = TN = 0
    for held in pos_vids:
        tr_p = load_many([i for i in pos_ids if i.split("_")[0] != held])
        tr_n = load_many([i for i in neg_ids if i.split("_")[0] != held])
        te_p = load_many([i for i in pos_ids if i.split("_")[0] == held])
        te_n = load_many([i for i in neg_ids if i.split("_")[0] == held])
        if not te_p:
            continue
        model = fit(tr_p, tr_n, bgs, dev)
        with torch.inference_mode():
            pp = model(to_tensor(te_p).to(dev)).argmax(1)
            tp, fn = int((pp == 1).sum()), int((pp == 0).sum())
            fp = tn = 0
            if te_n:
                pn = model(to_tensor(te_n).to(dev)).argmax(1)
                fp, tn = int((pn == 1).sum()), int((pn == 0).sum())
        TP, FN, FP, TN = TP + tp, FN + fn, FP + fp, TN + tn
        print(f"  {held:<7} купюр {len(te_p):>2}: найдено {tp:>2}, пропущено {fn:>2}"
              f"  |  не-купюр {len(te_n):>2}: ложно {fp}")

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    print(f"\nИТОГ LOVO: точность {prec:.2f}, полнота {rec:.2f}"
          f"   TP {TP}  FP {FP}  FN {FN}  TN {TN}")

    print("\nобучаю финальную модель на всех размеченных данных...")
    model = fit(load_many(pos_ids), load_many(neg_ids), bgs, dev)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "size": SIZE}, MODEL_OUT)
    print(f"модель сохранена: {MODEL_OUT}")


if __name__ == "__main__":
    main()
