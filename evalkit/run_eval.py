# -*- coding: utf-8 -*-
"""
Оценочный хартнесс. Единственный источник правды о том, стало лучше или хуже.

Запуск:
    python evalkit/run_eval.py --baseline none      # вердикт всегда no_violation
    python evalkit/run_eval.py --baseline always    # вердикт всегда violation
    python evalkit/run_eval.py --api http://localhost:8000 --key dev-key

Метрики считаются на ДВУХ уровнях:
  * вердикт по видео    -> критерий 1 ТЗ (25 баллов)
  * локализация события -> критерий 2 ТЗ (15 баллов)

Стоимость ошибки асимметрична (FAQ №16 ТЗ: "ложных срабатываний как можно
меньше, пропуск нарушения недопустим"), поэтому COST = FN*3 + FP*1.
Именно COST минимизируем при подборе порогов, а не accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalkit.labels import Clip, load_clips  # noqa: E402

W_FN, W_FP = 3, 1
TOLERANCES = (2.0, 3.0, 5.0)


# ---------------------------------------------------------------- предикторы
def predict_baseline(clip: Clip, mode: str) -> dict:
    if mode == "always":
        return {
            "verdict": "violation",
            "violations": [{"type": "baseline", "timecode_sec": 0.0}],
        }
    return {"verdict": "no_violation", "violations": []}


def predict_offline(clip: Clip) -> dict:
    """
    Прогон правил напрямую по КЭШУ детекций, без HTTP и без кадров-доказательств.

    Это рабочий режим Шага 3: подбор порогов в rules.yaml идёт за секунды,
    потому что нейросеть уже отработала и её результат лежит на диске.
    """
    from app.analyzer import RealAnalyzer
    from app.pipeline import Deadline

    sink: list = []
    verdict = RealAnalyzer().analyze(clip.path, Deadline(240), sink)
    return {
        "verdict": verdict,
        "violations": [
            {
                "type": v.type,
                "timecode_sec": v.timecode_sec,
                "rule_id": v.rule.id if v.rule else None,
                "confidence": v.confidence,
            }
            for v in sink
        ],
    }


def predict_api(clip: Clip, base_url: str, api_key: str, timeout: float = 300.0) -> dict:
    """Повторяет ровно то, что сделает жюри: POST /analyze -> поллинг GET /jobs/{id}."""
    boundary = "----evalkit"
    head = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + clip.path.name + '"\r\n'
        "Content-Type: video/quicktime\r\n\r\n"
    )
    tail = "\r\n--" + boundary + "--\r\n"
    body = head.encode("utf-8") + clip.path.read_bytes() + tail.encode("utf-8")

    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/v1/analyze",
        data=body,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        job_id = json.loads(r.read())["job_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/api/v1/jobs/" + job_id,
            headers={"X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
        status = payload.get("status")
        if status == "done":
            return payload
        if status == "failed":
            return {"verdict": "no_violation", "violations": [], "_failed": True}
        time.sleep(2.0)
    return {"verdict": "no_violation", "violations": [], "_timeout": True}


# ------------------------------------------------------------------ подсчёт
def evaluate(clips: list[Clip], predict, tol: float) -> dict:
    tp = tn = fp = fn = 0
    fp_ids: list[str] = []
    fn_ids: list[str] = []
    rows = []
    hits = {t: 0 for t in TOLERANCES}
    total_events = 0
    pred_events = 0          # сколько событий мы предсказали ВСЕГО
    pred_events_ok = 0       # из них попавших хотя бы в одно GT-событие (при tol)
    elapsed_total = 0.0

    for clip in clips:
        t0 = time.time()
        pred = predict(clip)
        dt = time.time() - t0
        elapsed_total += dt

        pv = pred.get("verdict") == "violation"
        gv = clip.is_violation
        if pv and gv:
            tp += 1
        elif not pv and not gv:
            tn += 1
        elif pv and not gv:
            fp += 1
            fp_ids.append(clip.video_id)
        else:
            fn += 1
            fn_ids.append(clip.video_id)

        pred_ts = [
            v.get("timecode_sec")
            for v in pred.get("violations", [])
            if v.get("timecode_sec") is not None
        ]
        total_events += len(clip.timestamps)
        pred_events += len(pred_ts)
        pred_events_ok += sum(
            1 for p in pred_ts if any(abs(p - gt) <= tol for gt in clip.timestamps)
        )
        matched_at_tol = 0
        for gt in clip.timestamps:
            for t in TOLERANCES:
                if any(abs(p - gt) <= t for p in pred_ts):
                    hits[t] += 1
                    if t == tol:
                        matched_at_tol += 1

        rows.append(
            {
                "id": clip.video_id,
                "gt": "VIOL" if gv else "ok",
                "pred": "VIOL" if pv else "ok",
                "ok": pv == gv,
                "gt_ts": clip.timestamps,
                "pred_ts": sorted(pred_ts)[:4],
                "loc": f"{matched_at_tol}/{len(clip.timestamps)}" if clip.timestamps else "-",
                "sec": dt,
            }
        )

    n = len(clips)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "cost": W_FN * fn + W_FP * fp,
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "loc_hits": hits,
        "total_events": total_events,
        "pred_events": pred_events,
        "pred_events_ok": pred_events_ok,
        "event_precision": pred_events_ok / pred_events if pred_events else 0.0,
        "rows": rows,
        "elapsed_total": elapsed_total,
    }


def report(m: dict, tol: float) -> None:
    print("=" * 76)
    header = f"{'ID':<8}{'GT':<6}{'PRED':<6}{'':<3}{'GT таймкоды':<20}{'предсказано':<20}лок."
    print(header)
    print("-" * 76)
    for r in m["rows"]:
        mark = "  " if r["ok"] else "<-"
        gt = ",".join(f"{t:g}" for t in r["gt_ts"]) or "-"
        pr = ",".join(f"{t:g}" for t in r["pred_ts"]) or "-"
        print(f"{r['id']:<8}{r['gt']:<6}{r['pred']:<6}{mark:<3}{gt:<20}{pr:<20}{r['loc']}")
    print("=" * 76)

    print("\nВЕРДИКТ (критерий 1, 25 баллов)")
    print(f"  accuracy   {m['accuracy']:.3f}   ({m['tp'] + m['tn']}/{m['n']})")
    print(f"  precision  {m['precision']:.3f}   recall {m['recall']:.3f}   f1 {m['f1']:.3f}")
    print(f"  TP {m['tp']}  TN {m['tn']}  FP {m['fp']}  FN {m['fn']}")
    print(f"  COST = {W_FN}*FN + {W_FP}*FP = {m['cost']}   <- это минимизируем")
    if m["fp_ids"]:
        print("  ЛОЖНЫЕ СРАБАТЫВАНИЯ: " + ", ".join(m["fp_ids"]))
    if m["fn_ids"]:
        print("  ПРОПУСКИ (критично!): " + ", ".join(m["fn_ids"]))

    print(f"\nЛОКАЛИЗАЦИЯ (критерий 2, 15 баллов)   событий всего: {m['total_events']}")
    for t in TOLERANCES:
        h = m["loc_hits"][t]
        frac = h / m["total_events"] if m["total_events"] else 0.0
        bar = "#" * int(30 * frac)
        print(f"  recall @ +-{t:g}s   {h}/{m['total_events']}  {frac:.3f}  {bar}")
    print(
        f"  precision @ +-{tol:g}s  {m['pred_events_ok']}/{m['pred_events']}"
        f"  {m['event_precision']:.3f}"
        "   <- падает, если спамить таймкодами"
    )

    print(f"\nВРЕМЯ   всего {m['elapsed_total']:.1f}s", end="")
    if m["rows"]:
        slowest = max(m["rows"], key=lambda r: r["sec"])
        warn = "  ПРЕВЫШЕН ЛИМИТ 300s!" if slowest["sec"] > 300 else ""
        print(f"   медленнее всего {slowest['id']}: {slowest['sec']:.1f}s{warn}")
    else:
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["none", "always", "offline"], default="none")
    ap.add_argument("--api", help="базовый URL сервиса, напр. http://localhost:8000")
    ap.add_argument("--key", default="dev-key")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--limit", type=int, help="взять только первые N видео")
    ap.add_argument("--json", help="сохранить метрики в файл")
    args = ap.parse_args()

    clips = load_clips()
    if args.limit:
        clips = clips[: args.limit]

    if args.api:
        def predict(c):
            return predict_api(c, args.api, args.key)
        print(f"Режим: API {args.api}")
    elif args.baseline == "offline":
        predict = predict_offline
        print("Режим: offline (правила по кэшу детекций)")
    else:
        def predict(c):
            return predict_baseline(c, args.baseline)
        print(f"Режим: baseline '{args.baseline}'")
    print(f"Видео: {len(clips)}  (уникальных, дубли слиты)\n")

    m = evaluate(clips, predict, args.tol)
    report(m, args.tol)

    if args.json:
        out = {k: v for k, v in m.items() if k != "rows"}
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nМетрики сохранены: {args.json}")


if __name__ == "__main__":
    main()
