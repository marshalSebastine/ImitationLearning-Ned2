"""Measure line-edge sharpness (10-90 % rise width in px) and Laplacian variance on saved frames."""
import sys, glob, os
import cv2, numpy as np

def edge_widths(gray, rows):
    out = []
    for r in rows:
        prof = gray[r].astype(float)
        prof = cv2.GaussianBlur(prof.reshape(1,-1), (1,1), 0).ravel()
        # find darkest run (the black line): threshold halfway between min and median
        lo, hi = np.percentile(prof, 2), np.percentile(prof, 80)
        if hi - lo < 40: 
            out.append((r, None)); continue
        mid = (lo + hi) / 2
        dark = prof < mid
        # largest dark run
        best, cur, start = (0,0,0), 0, 0
        for i, d in enumerate(dark):
            if d:
                if cur == 0: start = i
                cur += 1
                if cur > best[0]: best = (cur, start, i)
            else: cur = 0
        n, a, b = best
        if n < 5 or a < 15 or b > len(prof) - 15:
            out.append((r, None)); continue
        # 10-90% rise width at left edge
        def rise(seg, descending):
            seg = seg if not descending else seg[::-1]
            l10 = lo + 0.1*(hi-lo); l90 = lo + 0.9*(hi-lo)
            # seg goes from bright to dark when read in 'descending' orientation; we read from dark outwards
            idx10 = np.argmax(seg > l10) if (seg > l10).any() else None
            idx90 = np.argmax(seg > l90) if (seg > l90).any() else None
            if idx10 is None or idx90 is None: return None
            return abs(idx90 - idx10)
        left = rise(prof[max(0,a-15):a+1], descending=True)
        right = rise(prof[b:b+16], descending=False)
        out.append((r, n, left, right, int(lo), int(hi)))
    return out

for path in sys.argv[1:]:
    img = cv2.imread(path)
    if img is None: print(path, "unreadable"); continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lv = cv2.Laplacian(g, cv2.CV_64F).var()
    print(os.path.basename(path), f"{img.shape[1]}x{img.shape[0]}", f"lapvar {lv:.0f}", f"mean {g.mean():.0f}")
    for item in edge_widths(g, [20, 60, 120, 180, 240, 300, 360, 420, 460]):
        if len(item) == 2: print("   row", item[0], "no line found")
        else: print(f"   row {item[0]:3d}: line {item[1]:3d} px wide, edge rise 10-90% left {item[2]} px right {item[3]} px (dark {item[4]} bright {item[5]})")
