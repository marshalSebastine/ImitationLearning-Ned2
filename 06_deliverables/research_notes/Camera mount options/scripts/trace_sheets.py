"""Trace the black line on each path-sheet photo and compute the travel-direction span.

For each sheet: threshold the dark line, thin it (Zhang-Suen), walk the skeleton from an endpoint,
smooth the tangent direction, and find the smallest half-span of directions, i.e. the maximum turn
away from the best fixed direction, over both traversal directions (same answer by symmetry)."""
import sys, math
import numpy as np, cv2
from scipy import ndimage

def zhang_suen(img):
    img = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]; P3 = p[:-2, 2:]; P4 = p[1:-1, 2:]; P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]; P7 = p[2:, :-2]; P8 = p[1:-1, :-2]; P9 = p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                img[m] = 0; changed = True
    return img

def trace(skel):
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in pts]
    ends = [p for p in pts if len(nbrs(p)) == 1]
    if not ends:
        return None, ends
    # start from the endpoint with the longest walk
    best = None
    for e in ends:
        path = [e]; prev = None; cur = e; seen = {e}
        while True:
            nx = [n for n in nbrs(cur) if n not in seen]
            if not nx: break
            if len(nx) > 1 and prev is not None:
                # junction: keep going straight
                d = np.array(cur) - np.array(prev)
                nx.sort(key=lambda n: -np.dot(np.array(n) - np.array(cur), d))
            cur2 = nx[0]
            seen.add(cur2); path.append(cur2); prev, cur = cur, cur2
        if best is None or len(path) > len(best): best = path
    return np.array(best, float), ends

def direction_span(path, step=40):
    pts = path[::step]
    d = np.diff(pts, axis=0)
    ang = np.degrees(np.arctan2(-d[:, 0], d[:, 1]))  # image y down -> flip so angle is math-style
    # smallest arc containing all angles
    a = np.sort(np.mod(ang, 360))
    gaps = np.diff(np.concatenate([a, [a[0] + 360]]))
    span = 360 - gaps.max()
    return ang, span

for path in sys.argv[1:]:
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (g < 90).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    lab, n = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    keep = (lab == (1 + int(np.argmax(sizes)))).astype(np.uint8)
    skel = zhang_suen(keep)
    pth, ends = trace(skel)
    name = path.split("\\")[-1]
    if pth is None:
        print(name, "no endpoints (closed loop) -> all directions, span 360"); continue
    ang, span = direction_span(pth)
    print(f"{name}: line pixels {int(keep.sum())}, skeleton {int(skel.sum())}, traced {len(pth)} px, endpoints {len(ends)}")
    print(f"   direction span {span:.0f} deg -> max turn from best fixed direction = {span/2:.0f} deg")
    print("   tangent angles along path (deg, every ~40px):", np.round(ang).astype(int).tolist())
