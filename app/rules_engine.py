# -*- coding: utf-8 -*-
"""
Движок правил. Декларативные конъюнкции из rules.yaml поверх вектора состояния.

Почему конъюнкции, а не одиночные признаки — проверено на данных:
  X-29 размечено "без нарушения", хотя кассир встаёт и отходит от стола;
  X-16 размечено "без нарушения", хотя содержит передачу инкассаторской сумки.
Правило "ушёл = нарушение" на них даёт ложные срабатывания.

Пороги живут в YAML, а не в коде: их крутит аналитик в Шаге 3, не трогая пайплайн.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app import config
from app.scene import FrameState, SceneTrack


@dataclass
class Event:
    rule_id: str
    type: str
    t_start: float
    t_end: float
    timecode: float
    confidence: float
    document: dict
    clause: str
    quote: str
    verified: bool

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cmp(value: Any, cond: dict) -> bool:
    """Одно атомарное условие над сигналом."""
    if "equals" in cond:
        return bool(value) == bool(cond["equals"])
    if "gt" in cond:
        return float(value) > float(cond["gt"])
    if "gte" in cond:
        return float(value) >= float(cond["gte"])
    if "lt" in cond:
        return float(value) < float(cond["lt"])
    return False


def _holds(st: FrameState, node: dict) -> bool:
    """Рекурсивно вычисляет узел условия на ОДНОМ кадре (без временных окон)."""
    if "all" in node:
        return all(_holds(st, c) for c in node["all"])
    if "any" in node:
        return any(_holds(st, c) for c in node["any"])
    if "signal" in node:
        return _cmp(st.signal(node["signal"]), node)
    return False


def _min_duration(node: dict) -> float:
    """Максимальное min_duration_s по всему дереву условия."""
    d = float(node.get("min_duration_s", 0.0))
    for key in ("all", "any"):
        for c in node.get(key, []):
            d = max(d, _min_duration(c))
    return d


def evaluate(track: SceneTrack, rules_path: Path = config.RULES_PATH) -> list[Event]:
    spec = _load_yaml(rules_path)
    docs = spec.get("documents", {})
    events: list[Event] = []

    for rule in spec.get("rules", []):
        if not rule.get("enabled", False):
            continue
        need = _min_duration(rule)

        # находим непрерывные интервалы, где условие держится
        run_start: Optional[float] = None
        prev_t = None
        for st in track.frames:
            ok = _holds(st, rule)
            if ok and run_start is None:
                run_start = st.t
            elif not ok and run_start is not None:
                _emit(events, rule, docs, run_start, prev_t, need, track)
                run_start = None
            prev_t = st.t
        if run_start is not None and prev_t is not None:
            _emit(events, rule, docs, run_start, prev_t, need, track)

    events.sort(key=lambda e: e.timecode)
    return events


def _emit(
    events: list[Event],
    rule: dict,
    docs: dict,
    t0: float,
    t1: float,
    need: float,
    track: SceneTrack,
) -> None:
    dur = t1 - t0
    if dur < need:
        return
    mode = rule.get("timecode", "at_start")
    if mode == "at_end":
        tc = t1
    elif mode == "at_peak":
        tc = (t0 + t1) / 2.0
    else:
        tc = t0
    # уверенность растёт с длительностью: мгновенная вспышка детектора слабее
    conf = min(1.0, 0.45 + 0.25 * (dur / max(need, 1.0)))
    doc = docs.get(rule.get("document", ""), {})
    events.append(
        Event(
            rule_id=rule["id"],
            type=rule["type"],
            t_start=t0,
            t_end=t1,
            timecode=round(tc, 1),
            confidence=round(conf, 2),
            document=doc,
            clause=rule.get("clause", ""),
            quote=(rule.get("quote") or "").strip(),
            verified=bool(rule.get("verified", False)),
        )
    )


def merge_events(events: list[Event], gap: float = 4.0) -> list[Event]:
    """
    Схлопывает события одного правила, стоящие вплотную.

    Без этого мигание детектора порождает десяток "нарушений" вместо одного,
    и event-precision в хартнессе проваливается.
    """
    out: list[Event] = []
    for e in sorted(events, key=lambda x: (x.rule_id, x.t_start)):
        if out and out[-1].rule_id == e.rule_id and e.t_start - out[-1].t_end <= gap:
            prev = out[-1]
            prev.t_end = max(prev.t_end, e.t_end)
            prev.confidence = max(prev.confidence, e.confidence)
        else:
            out.append(e)
    out.sort(key=lambda x: x.timecode)
    return out
