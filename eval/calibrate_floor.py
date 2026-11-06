import json, sys, statistics, yaml
run, qfile = sys.argv[1], sys.argv[2]
Q = {q["id"]: q for q in yaml.safe_load(open(qfile))}
rows = [r for r in json.load(open(run))["per_question"] if r.get("shape") == "enumeration"]
def at(floor, rs):
    rec, noise, prec = [], [], []
    for r in rs:
        q = Q[r["id"]]; exp = {e["vendor"] for e in q["expected_vendors"]}; unc = set(q["unclear_vendors"])
        kept = {v for v, s in (r.get("vendor_best_scores") or {}).items() if s >= floor}
        e = exp - unc
        if not e: continue
        rec.append(len(kept & e) / len(e)); n = kept - e - unc; noise.append(len(n))
        prec.append(len(kept & e) / max(1, len(kept - unc)))
    return statistics.fmean(rec), statistics.fmean(noise), statistics.fmean(prec)
cal = [r for r in rows if not Q[r["id"]].get("holdout")]; hold = [r for r in rows if Q[r["id"]].get("holdout")]
print("floor | cal recall noise prec | holdout recall noise prec")
best = None
for f in [x / 100 for x in range(0, 80, 2)]:
    c = at(f, cal); h = at(f, hold)
    print(f"{f:.2f} | {c[0]:.3f} {c[1]:5.1f} {c[2]:.3f} | {h[0]:.3f} {h[1]:5.1f} {h[2]:.3f}")
    if c[0] >= 0.8: best = f
print("chosen floor (highest with calibration recall >= 0.8):", best)
