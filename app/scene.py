# -*- coding: utf-8 -*-
"""
Сборка вектора состояния сцены из сырых детекций.

КЛЮЧЕВОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ — зона кассира определяется АВТОМАТИЧЕСКИ,
никаких зашитых пиксельных ROI. ТЗ предупреждает: финальное видео снято под
другим углом (FAQ №4), а критерий 5 (10 баллов) — именно про устойчивость.
Хардкод координат там развалится.

Как ищем зону кассира: камера статична, кассир — самый "долгоживущий" человек
в кадре. Берём все детекции person за весь ролик, строим гистограмму позиций
и находим самый плотный кластер. Он и есть рабочее место.
Все пороги — в ДОЛЯХ кадра, не в пикселях.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Box = list  # [x1, y1, x2, y2]


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _overlap_frac(inner: Box, outer: Box) -> float:
    """Какая доля inner попадает в outer."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = _area(inner)
    return inter / a if a > 0 else 0.0


@dataclass
class FrameState:
    t: float
    cashier_present: bool = False
    cashier_box: Optional[Box] = None
    persons_in_cash_zone: int = 0
    persons_total: int = 0
    phone_in_hand: bool = False
    objects: dict = field(default_factory=dict)   # group -> max score в зоне кассы

    def signal(self, name: str) -> float | bool | int:
        if name.startswith("objects_on_desk."):
            return self.objects.get(name.split(".", 1)[1], 0.0)
        return getattr(self, name, 0.0)


@dataclass
class SceneTrack:
    frames: list[FrameState]
    cash_zone: Box
    width: int
    height: int
    duration: float
    fps: float

    def at(self, t: float) -> Optional[FrameState]:
        best, bd = None, 1e9
        for f in self.frames:
            d = abs(f.t - t)
            if d < bd:
                best, bd = f, d
        return best


def estimate_cash_zone(detections: dict, grid: int = 12) -> Box:
    """
    Самый плотный кластер детекций person за весь ролик = рабочее место кассира.

    Работает при любом ракурсе, потому что опирается не на геометрию помещения,
    а на то, что кассир проводит в кадре больше времени, чем кто-либо ещё.
    """
    w, h = detections["width"], detections["height"]
    cell_w, cell_h = w / grid, h / grid
    heat = [[0.0] * grid for _ in range(grid)]

    for fr in detections["frames"]:
        for d in fr["dets"]:
            if d["g"] != "person":
                continue
            cx, cy = _center(d["b"])
            gx = min(grid - 1, max(0, int(cx / cell_w)))
            gy = min(grid - 1, max(0, int(cy / cell_h)))
            heat[gy][gx] += d["s"]

    # лучшая ячейка 2x2 — устойчивее одиночной
    best, bx, by = -1.0, 0, 0
    for gy in range(grid - 1):
        for gx in range(grid - 1):
            s = heat[gy][gx] + heat[gy][gx + 1] + heat[gy + 1][gx] + heat[gy + 1][gx + 1]
            if s > best:
                best, bx, by = s, gx, gy

    # расширяем на одну ячейку в каждую сторону — кассир двигается на стуле
    x1 = max(0.0, (bx - 1) * cell_w)
    y1 = max(0.0, (by - 1) * cell_h)
    x2 = min(float(w), (bx + 3) * cell_w)
    y2 = min(float(h), (by + 3) * cell_h)
    return [x1, y1, x2, y2]


def build_scene(detections: dict, zone_overlap_min: float = 0.25) -> SceneTrack:
    """Превращает сырые детекции в последовательность состояний сцены."""
    zone = estimate_cash_zone(detections)
    w, h = detections["width"], detections["height"]
    frames: list[FrameState] = []

    for fr in detections["frames"]:
        st = FrameState(t=fr["t"])
        persons = [d for d in fr["dets"] if d["g"] == "person"]
        st.persons_total = len(persons)

        in_zone = [d for d in persons if _overlap_frac(d["b"], zone) >= zone_overlap_min]
        st.persons_in_cash_zone = len(in_zone)
        if in_zone:
            # кассир = самая крупная фигура в зоне (ближе к камере / сидит за столом)
            cashier = max(in_zone, key=lambda d: _area(d["b"]))
            st.cashier_present = True
            st.cashier_box = cashier["b"]

        for d in fr["dets"]:
            g = d["g"]
            if g in ("person", "laptop"):
                continue
            # объект считаем "на рабочем месте", если он пересекается с зоной кассы
            if _overlap_frac(d["b"], zone) >= 0.2:
                st.objects[g] = max(st.objects.get(g, 0.0), d["s"])

        if st.cashier_box is not None:
            for d in fr["dets"]:
                if d["g"] == "phone" and _overlap_frac(d["b"], st.cashier_box) >= 0.5:
                    st.phone_in_hand = True

        frames.append(st)

    return SceneTrack(
        frames=frames,
        cash_zone=zone,
        width=w,
        height=h,
        duration=detections["duration"],
        fps=detections["fps"],
    )


def smooth(track: SceneTrack, window: int = 3, fill_gap_s: float = 1.5) -> SceneTrack:
    """
    Два шага подавления шума детектора.

    1. Медианное сглаживание булевых сигналов — гасит одиночные пропуски,
       из-за которых cashier_present мигает и правила ловят шум.
    2. Заполнение коротких дыр в сигналах объектов. Сумка на столе не исчезает
       на полсекунды — это детектор её потерял. Без этого шага непрерывный
       интервал никогда не дотягивает до min_duration_s, и правило молчит,
       хотя объект на кадре есть (проверено на X-40: сумка со score 0.96
       не поднимала R2).
    """
    n = len(track.frames)
    if n == 0:
        return track

    if n >= window >= 3:
        half = window // 2
        raw = [(f.cashier_present, f.persons_in_cash_zone, f.phone_in_hand)
               for f in track.frames]
        for i, f in enumerate(track.frames):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            seg = raw[lo:hi]
            f.cashier_present = sum(s[0] for s in seg) * 2 > len(seg)
            f.persons_in_cash_zone = sorted(s[1] for s in seg)[len(seg) // 2]
            f.phone_in_hand = sum(s[2] for s in seg) * 2 > len(seg)

    groups = {g for f in track.frames for g in f.objects}
    for g in groups:
        last_i: Optional[int] = None
        for i, f in enumerate(track.frames):
            if f.objects.get(g, 0.0) > 0:
                if last_i is not None and 0 < i - last_i:
                    if track.frames[i].t - track.frames[last_i].t <= fill_gap_s:
                        lvl = min(f.objects[g], track.frames[last_i].objects[g])
                        for j in range(last_i + 1, i):
                            track.frames[j].objects[g] = max(
                                track.frames[j].objects.get(g, 0.0), lvl
                            )
                last_i = i
    return track


def absence_intervals(track: SceneTrack, min_sec: float = 0.0) -> list[tuple]:
    """Интервалы, где кассира нет в зоне. Удобно для диагностики и правил."""
    out, start, prev = [], None, None
    for f in track.frames:
        if not f.cashier_present and start is None:
            start = f.t
        elif f.cashier_present and start is not None:
            if prev is not None and prev - start >= min_sec:
                out.append((start, prev))
            start = None
        prev = f.t
    if start is not None and prev is not None and prev - start >= min_sec:
        out.append((start, prev))
    return out
