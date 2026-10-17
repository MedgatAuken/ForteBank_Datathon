# -*- coding: utf-8 -*-
"""
Загрузка и нормализация разметки ForteBank Datathon 26.

Решает три проблемы исходного open_labels.xlsx:
  1. Имена файлов смешивают кириллическую 'Х' и латинскую 'X'  -> нормализуем ID.
  2. Таймкоды записаны прозой ("22 секунда", "1минута 8 секунда") -> парсим в секунды.
  3. Есть побайтовые дубли файлов (проверено по md5):
        Х-4 == Х-47  -> ОДНО видео с ДВУМЯ нарушениями (57с и 68с). Сливаем.
        Х-5 == Х-48  -> полный дубль с одинаковой меткой. Оставляем один.
     Наивный dedup по имени потеряет метку 68с - единственный в датасете
     пример мульти-нарушения в одном ролике.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "open"
VIDEO_DIR = DATA_DIR / "videos"
LABELS_XLSX = DATA_DIR / "open_labels.xlsx"


@dataclass
class Clip:
    """Одно уникальное видео (после слияния дублей)."""
    video_id: str                    # нормализованный ID, напр. "X-4"
    path: Path                       # путь к файлу на диске
    md5: str
    timestamps: list[float] = field(default_factory=list)   # секунды, отсортированы
    aliases: list[str] = field(default_factory=list)        # ID дублей, напр. ["X-47"]

    @property
    def is_violation(self) -> bool:
        return len(self.timestamps) > 0

    @property
    def verdict(self) -> str:
        return "violation" if self.is_violation else "no_violation"


def _norm_id(raw: str) -> str:
    """Кириллическая 'Х' (U+0425) -> латинская 'X'. Иначе X-4 и Х-4 будут разными ключами."""
    return raw.strip().replace("\u0425", "X").replace("\u0445", "x")


def parse_timestamp(raw: str | None) -> float | None:
    """'22 секунда' -> 22.0 ; '1минута 8 секунда' -> 68.0 ; '' -> None"""
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    total = 0.0
    m = re.search(r"(\d+)\s*(?:минут|мин|m)", s)
    if m:
        total += int(m.group(1)) * 60
        s = s[m.end():]
    m = re.search(r"(\d+)", s)
    if m:
        total += int(m.group(1))
    return total if total > 0 else None


def _read_rows(xlsx: Path) -> list[list[str]]:
    """Читает лист xlsx без pandas/openpyxl - только stdlib."""
    with zipfile.ZipFile(xlsx) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.iter(f"{{{NS['m']}}}row"):
            cells: dict[str, str] = {}
            for c in row:
                ref = c.get("r", "")
                col = re.sub(r"\d", "", ref)
                v = c.find("m:v", NS)
                inline = c.find("m:is/m:t", NS)
                if inline is not None:
                    val = inline.text or ""
                elif v is not None:
                    val = v.text or ""
                    if c.get("t") == "s":          # индекс в sharedStrings
                        val = shared[int(val)]
                else:
                    val = ""
                cells[col] = val
            rows.append([cells.get("A", ""), cells.get("B", ""), cells.get("C", "")])
        return rows


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clips(video_dir: Path = VIDEO_DIR, xlsx: Path = LABELS_XLSX) -> list[Clip]:
    """Возвращает список УНИКАЛЬНЫХ видео со слитыми метками."""
    # 1) сырые строки разметки
    raw: dict[str, list[float]] = {}
    for a, b, c in _read_rows(xlsx)[1:]:
        if not a.strip():
            continue
        vid = _norm_id(a)
        ts = parse_timestamp(b)
        is_viol = "без" not in c.lower() and "наруш" in c.lower()
        raw.setdefault(vid, [])
        if is_viol and ts is not None:
            raw[vid].append(ts)

    # 2) файлы на диске -> тот же нормализованный ID
    files: dict[str, Path] = {}
    for p in sorted(video_dir.glob("*.mov")):
        files[_norm_id(p.stem)] = p

    missing = set(raw) - set(files)
    if missing:
        raise FileNotFoundError(f"В разметке есть ID без файлов: {sorted(missing)}")

    # 3) группировка по md5 - слияние побайтовых дублей
    by_hash: dict[str, Clip] = {}
    for vid in sorted(raw, key=lambda s: (len(s), s)):
        p = files[vid]
        h = _md5(p)
        if h in by_hash:
            clip = by_hash[h]
            clip.aliases.append(vid)
            # КЛЮЧЕВОЕ: объединяем таймкоды, а не отбрасываем дубль целиком
            for t in raw[vid]:
                if all(abs(t - e) > 1.0 for e in clip.timestamps):
                    clip.timestamps.append(t)
            clip.timestamps.sort()
        else:
            by_hash[h] = Clip(video_id=vid, path=p, md5=h,
                              timestamps=sorted(raw[vid]))
    return sorted(by_hash.values(), key=lambda c: c.video_id)


if __name__ == "__main__":
    clips = load_clips()
    n_viol = sum(c.is_violation for c in clips)
    n_events = sum(len(c.timestamps) for c in clips)
    print(f"Уникальных видео : {len(clips)}")
    print(f"  с нарушением   : {n_viol}")
    print(f"  без нарушения  : {len(clips) - n_viol}")
    print(f"Событий-нарушений: {n_events}")
    print()
    for c in clips:
        if c.aliases:
            print(f"  СЛИТО {c.video_id} + {c.aliases}  md5={c.md5[:8]}  ts={c.timestamps}")
