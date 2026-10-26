"""Per-frame brightness over the push runs (exposure adaptation) and paper-edge sag (barrel distortion) on far-end frames."""
import sys, glob, os
import cv2, numpy as np
folder = sys.argv[1]
for run in ("1419", "1430"):
    files = sorted(glob.glob(os.path.join(folder, f"2026-10-25_{run}_step_*.jpg")))
    print(f"run {run}: {len(files)} frames   step: mean / paper p90 / line p2 / top-strip mean (rows 0-40) / bottom-strip mean (rows 440-480)")
    for f in files:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        print(f"   {os.path.basename(f)[-12:-4]}: {g.mean():5.1f} / {np.percentile(g,90):5.1f} / {np.percentile(g,2):5.1f} / {g[:40].mean():5.1f} / {g[440:].mean():5.1f}")
# paper far-edge sag on the last frames of 1430
for f in sorted(glob.glob(os.path.join(folder, "2026-10-25_1430_step_*.jpg")))[-5:]:
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).astype(float)
    g = cv2.GaussianBlur(g, (5,5), 0)
    rows = {}
    for col in (40, 120, 200, 320, 440, 520, 600):
        prof = g[:200, col-5:col+5].mean(axis=1)
        # the paper is bright; find the first row from the top where brightness exceeds 0.5*(max+min) of the column strip and stays
        thr = 0.5*(prof.max()+prof.min())
        idx = np.where(prof > thr)[0]
        rows[col] = int(idx[0]) if len(idx) else None
    print(os.path.basename(f), "paper far edge row at columns", rows)
