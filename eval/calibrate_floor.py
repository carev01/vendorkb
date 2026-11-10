"""Enumeration rerank-floor sweep over one ``cross_vendor.py score --rerank`` run.

    uv run python eval/calibrate_floor.py eval/runs/RUN.json eval/questions.yaml

Reads each enumeration row's ``vendor_best_scores`` (every scored vendor, those below the
run's --min-score included) and prints, per floor, mean vendor recall, noise vendors and
precision on the calibration and holdout splits. It prints two rules: the plan's (highest
floor with calibration recall >= 0.8) and the adopted one (highest floor with calibration
recall >= 0.95). The floor is recall-first (ruling of 2026-11-06, spec §9.2: the answer's line
check owns precision). It was 0.46 (FTS weight 1.0), then 0.54 (FTS weight 0.25, 2026-11-06),
and is 0.56 since 2026-11-10, when the reranker began to judge six candidates per vendor
(``modes.ENUM_RERANK_DEPTH``): calibration recall 0.964, holdout 0.977 there.
"""
from __future__ import annotations

import json
import statistics
import sys

import yaml

ADOPTED_FLOOR = 0.56


def at(floor: float, rows: list[dict], questions: dict[str, dict]) -> tuple[float, float, float]:
    """(mean recall, mean noise vendors, mean precision) at ``floor``; unclear vendors count
    on neither side, questions with nothing to find are skipped."""
    rec, noise, prec = [], [], []
    for r in rows:
        q = questions[r["id"]]
        unclear = set(q.get("unclear_vendors") or [])
        expected = {e["vendor"] for e in q.get("expected_vendors") or []} - unclear
        if not expected:
            continue
        kept = {v for v, s in (r.get("vendor_best_scores") or {}).items() if s >= floor}
        rec.append(len(kept & expected) / len(expected))
        noise.append(len(kept - expected - unclear))
        prec.append(len(kept & expected) / max(1, len(kept - unclear)))
    if not rec:
        return 0.0, 0.0, 0.0
    return statistics.fmean(rec), statistics.fmean(noise), statistics.fmean(prec)


def main(argv: list[str] | None = None) -> dict[str, float | None]:
    run, qfile = (argv if argv is not None else sys.argv[1:])[:2]
    with open(qfile, encoding="utf-8") as f:
        questions = {q["id"]: q for q in yaml.safe_load(f)}
    with open(run, encoding="utf-8") as f:
        rows = [r for r in json.load(f)["per_question"] if r.get("shape") == "enumeration"]
    cal = [r for r in rows if not questions[r["id"]].get("holdout")]
    hold = [r for r in rows if questions[r["id"]].get("holdout")]
    print("floor | cal recall noise prec | holdout recall noise prec")
    best80: float | None = None
    best95: float | None = None
    for floor in [x / 100 for x in range(0, 80, 2)]:
        c, h = at(floor, cal, questions), at(floor, hold, questions)
        print(f"{floor:.2f} | {c[0]:.3f} {c[1]:5.1f} {c[2]:.3f} | {h[0]:.3f} {h[1]:5.1f} {h[2]:.3f}")
        if c[0] >= 0.8:
            best80 = floor
        if c[0] >= 0.95:
            best95 = floor
    print("plan rule (highest floor with calibration recall >= 0.8):", best80)
    print("adopted rule (highest floor with calibration recall >= 0.95):", best95)
    c, h = at(ADOPTED_FLOOR, cal, questions), at(ADOPTED_FLOOR, hold, questions)
    print(f"shipped ENUMERATION_MIN_SCORE = {ADOPTED_FLOOR} (recall-first floor, spec §9.2): "
          f"calibration recall {c[0]:.3f}, holdout {h[0]:.3f} in this run")
    return {"recall_0.8": best80, "recall_0.95": best95}


if __name__ == "__main__":
    main()
