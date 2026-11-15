"""
Verifier 04: the fold models' twist on held-out sheets (saved predictions only, no model is run).
Size ratio, slopes, sign; sums over 10, 20, 40 steps; step-to-step noise of the demonstrated twist and what a mean over
plus or minus 2 steps removes.
"""
import numpy
import pandas
from vcommon import demos, TRAINED

FOLD_RUNS = {"A": "2026-11-09_resnet18_final_candidate", "A seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
             "chunk": "2026-11-10_resnet18_final_candidate_chunk", "E": "2026-11-10_resnet18_chunk_stop_weight_25",
             "E seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "twist focus": "2026-11-10_resnet18_chunk_twist_focus",
             "twist focus seed 2": "2026-11-10_resnet18_chunk_twist_focus_seed_2"}
SHEETS = ["path_0%d" % k for k in range(1, 7)]
frames = demos()


def joined(run_folder):
    parts = []
    for sheet in SHEETS:
        saved = numpy.load(TRAINED / run_folder / f"predictions_without_{sheet}.npz")
        parts.append(pandas.DataFrame({"frame_index": saved["frame_index"], "pred": saved["twist_deg"], "fold_sheet": sheet}))
    table = frames.merge(pandas.concat(parts), on = "frame_index", how = "inner")
    assert (table["sheet"] == table["fold_sheet"]).all()
    return table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def origin_slope(y, x):
    return float(numpy.sum(x * y) / numpy.sum(x * x))


print("1. step by step, non-stop rows of the held-out sheets")
tables = {}
for label, folder in FOLD_RUNS.items():
    if not (TRAINED / folder).exists():
        print("   missing", folder); continue
    t = joined(folder)
    t = t[t["action"] != "stop"].reset_index(drop = True)
    tables[label] = t
    d, p = t["twist"].to_numpy(), t["pred"].to_numpy()
    big = numpy.abs(d) >= 2.0
    print(f"   {label:18s} rows {len(t)}  mean|pred|/mean|demo| {numpy.abs(p).mean() / numpy.abs(d).mean():.2f}  demo on pred {origin_slope(d, p):.2f} (least squares with offset {numpy.polyfit(p, d, 1)[0]:.2f})"
          f"  pred on demo {origin_slope(p, d):.2f}  correlation {numpy.corrcoef(d, p)[0, 1]:.2f}  sign right (|demo| >= 2) {numpy.mean(numpy.sign(p[big]) == numpy.sign(d[big])):.2f}")
for label in ("E", "A"):
    t = tables[label]
    line = []
    for sheet, rows in t.groupby("sheet"):
        d, p = rows["twist"].to_numpy(), rows["pred"].to_numpy()
        big = numpy.abs(d) >= 2.0
        line.append(f"{sheet[-2:]}: size {numpy.abs(p).mean() / numpy.abs(d).mean():.2f} gain {origin_slope(d, p):.2f} corr {numpy.corrcoef(d, p)[0, 1]:.2f} sign {numpy.mean(numpy.sign(p[big]) == numpy.sign(d[big])):.2f}")
    print(f"   {label} per sheet: " + "; ".join(line))

print("\n2. summed over n consecutive demonstrated steps (every start row whose next n - 1 rows are consecutive rows of the same episode)")
for label in ("E", "A"):
    t = tables[label]
    for n in (10, 20, 40):
        D, P, S = [], [], []
        for episode, e in t.groupby("episode_id"):
            rows = e["row_in_episode"].to_numpy(); d = e["twist"].to_numpy(); p = e["pred"].to_numpy()
            cd, cp = numpy.r_[0, numpy.cumsum(d)], numpy.r_[0, numpy.cumsum(p)]
            for i in range(0, len(e) - n + 1):
                if rows[i + n - 1] - rows[i] == n - 1:
                    D.append(cd[i + n] - cd[i]); P.append(cp[i + n] - cp[i]); S.append(e["sheet"].iloc[0])
        D, P, S = numpy.array(D), numpy.array(P), numpy.array(S)
        drift = P - D
        gain = origin_slope(D, P)
        print(f"   {label} n {n:2d}: windows {len(D)}  median |demo turn| {numpy.median(numpy.abs(D)):.1f}  median |drift| {numpy.median(numpy.abs(drift)):.1f}  90th {numpy.quantile(numpy.abs(drift), 0.9):.1f}"
              f"  pred on demo {origin_slope(P, D):.2f}  best gain {gain:.2f}  correlation {numpy.corrcoef(D, P)[0, 1]:.2f}  median |drift| with best gain {numpy.median(numpy.abs(gain * P - D)):.1f}"
              f"  with gain 1.5: {numpy.median(numpy.abs(1.5 * P - D)):.1f}  with gain 2: {numpy.median(numpy.abs(2 * P - D)):.1f}")
        if n == 10:
            print("        per sheet (pred on demo / best gain / correlation): " + "; ".join(
                f"{sheet[-2:]}: {origin_slope(P[S == sheet], D[S == sheet]):.2f} / {origin_slope(D[S == sheet], P[S == sheet]):.2f} / {numpy.corrcoef(D[S == sheet], P[S == sheet])[0, 1]:.2f}" for sheet in SHEETS))
        if n == 40:
            print("        per sheet (pred on demo / best gain / correlation): " + "; ".join(
                f"{sheet[-2:]}: {origin_slope(P[S == sheet], D[S == sheet]):.2f} / {origin_slope(D[S == sheet], P[S == sheet]):.2f} / {numpy.corrcoef(D[S == sheet], P[S == sheet])[0, 1]:.2f}" for sheet in SHEETS))

print("\n3. how noisy is the demonstrated twist step to step (non-stop rows, consecutive rows within an episode)")
t = tables["E"]
now, nxt, raw, smooth, sheets = [], [], [], [], []
for episode, e in t.groupby("episode_id"):
    rows = e["row_in_episode"].to_numpy(); d = e["twist"].to_numpy()
    for i in range(len(e) - 1):
        if rows[i + 1] == rows[i] + 1:
            now.append(d[i]); nxt.append(d[i + 1])
    for i in range(2, len(e) - 2):
        if rows[i + 2] - rows[i - 2] == 4:
            raw.append(d[i]); smooth.append(d[i - 2:i + 3].mean()); sheets.append(e["sheet"].iloc[0])
now, nxt, raw, smooth, sheets = map(numpy.array, (now, nxt, raw, smooth, sheets))
both = (numpy.abs(now) >= 2) & (numpy.abs(nxt) >= 2)
print(f"   next-step correlation {numpy.corrcoef(now, nxt)[0, 1]:.2f}; two twisting steps (>= 2 degrees) in a row that reverse sign: {numpy.mean(numpy.sign(now[both]) != numpy.sign(nxt[both])):.3f}")
removed = raw - smooth
print(f"   mean over +-2 steps: variance kept {smooth.var() / raw.var():.2f}; removed part sd {removed.std():.2f} = {removed.var() / raw.var():.2f} of the variance; raw sd {raw.std():.2f}")
print(f"   step-to-step change sd: raw {numpy.std(nxt - now):.2f}")
for sheet in SHEETS:
    m = sheets == sheet
    print(f"      {sheet}: removed share of variance {removed[m].var() / raw[m].var():.2f}")
# the saved predictions against the raw and the smoothed twist on the same rows
pred_rows, raw_rows, smooth_rows = [], [], []
for episode, e in t.groupby("episode_id"):
    rows = e["row_in_episode"].to_numpy(); d = e["twist"].to_numpy(); p = e["pred"].to_numpy()
    for i in range(2, len(e) - 2):
        if rows[i + 2] - rows[i - 2] == 4:
            pred_rows.append(p[i]); raw_rows.append(d[i]); smooth_rows.append(d[i - 2:i + 3].mean())
pred_rows, raw_rows, smooth_rows = map(numpy.array, (pred_rows, raw_rows, smooth_rows))
print(f"   E predictions against the raw twist: correlation {numpy.corrcoef(pred_rows, raw_rows)[0, 1]:.2f}, size ratio {numpy.abs(pred_rows).mean() / numpy.abs(raw_rows).mean():.2f};"
      f" against the smoothed twist: correlation {numpy.corrcoef(pred_rows, smooth_rows)[0, 1]:.2f}, size ratio {numpy.abs(pred_rows).mean() / numpy.abs(smooth_rows).mean():.2f}")
