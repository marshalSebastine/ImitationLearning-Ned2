"""Barrel-distortion check: centre column of the (straight) line per row, quadratic fit, compared with Niryo's k1=-0.41 model."""
import sys, os
import cv2, numpy as np
fx, fy, cx, cy = 508.069, 506.445, 332.821, 242.984
k1, k2, k3 = -0.4146402, 0.2407384, -0.1040042
def distort(xn, yn):
    r2 = xn*xn + yn*yn
    f = 1 + k1*r2 + k2*r2*r2 + k3*r2**3
    return xn*f, yn*f
for path in sys.argv[1:]:
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(float)
    rows, cols = [], []
    for r in range(5, 330, 5):
        prof = cv2.GaussianBlur(g[r:r+1], (1,1), 0).ravel()
        lo, hi = np.percentile(prof, 2), np.percentile(prof, 80)
        if hi - lo < 40: continue
        dark = prof < (lo+hi)/2
        best, cur, start = (0,0,0), 0, 0
        for i, d in enumerate(dark):
            if d:
                if cur == 0: start = i
                cur += 1
                if cur > best[0]: best = (cur, start, i)
            else: cur = 0
        n, a, b = best
        if 20 < n < 160 and a > 10 and b < 630:
            rows.append(r); cols.append((a+b)/2)
    rows, cols = np.array(rows), np.array(cols)
    if len(rows) < 8: print(os.path.basename(path), "line not tracked"); continue
    lin = np.polyfit(rows, cols, 1); quad = np.polyfit(rows, cols, 2)
    resid_lin = cols - np.polyval(lin, rows)
    # predicted bow from Niryo's model: take the undistorted straight line through the fitted end points, distort it
    # undistort the two end observations approximately by inverting numerically
    def undistort(xd, yd):
        xn, yn = xd, yd
        for _ in range(30):
            dx, dy = distort(xn, yn)
            xn += xd - dx; yn += yd - dy
        return xn, yn
    r0, r1 = rows[0], rows[-1]
    c0, c1 = np.polyval(quad, r0), np.polyval(quad, r1)
    p0 = undistort((c0-cx)/fx, (r0-cy)/fy); p1 = undistort((c1-cx)/fx, (r1-cy)/fy)
    pred = []
    for r in rows:
        t = (r - r0)/(r1 - r0)
        xn = p0[0] + t*(p1[0]-p0[0]); yn = p0[1] + t*(p1[1]-p0[1])
        xd, yd = distort(xn, yn)
        pred.append(xd*fx + cx)
    pred = np.array(pred)
    pred_lin = np.polyfit(rows, pred, 1); pred_resid = pred - np.polyval(pred_lin, rows)
    print(os.path.basename(path), f"rows {r0}-{r1}, line centre col {cols[0]:.0f} -> {cols[-1]:.0f}")
    print(f"   measured bow (max deviation from a straight fit): {resid_lin.max()-resid_lin.min():.1f} px, quad coeff {quad[0]*1e4:.2f}e-4 px/row^2")
    print(f"   Niryo k1=-0.41 model predicts bow: {pred_resid.max()-pred_resid.min():.1f} px for the same end points")
