"""Field of view from the ned_ros cam_intrinsics.yaml (pinhole + Brown radial distortion), 640x480."""
import numpy as np
fx, fy, cx, cy = 508.069, 506.445, 332.821, 242.984
k1, k2, k3 = -0.4146402, 0.2407384, -0.1040042
W, H = 640, 480
def undistorted_r(rd):
    # solve rd = ru*(1+k1 ru^2+k2 ru^4+k3 ru^6) for ru by bisection
    lo, hi = 0.0, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        f = mid * (1 + k1 * mid**2 + k2 * mid**4 + k3 * mid**6)
        if f < rd: lo = mid
        else: hi = mid
    return (lo + hi) / 2
print('pinhole (undistorted model) FOV:')
print(f'  horizontal = {np.degrees(np.arctan(cx/fx)+np.arctan((W-cx)/fx)):.1f} deg   vertical = {np.degrees(np.arctan(cy/fy)+np.arctan((H-cy)/fy)):.1f} deg   diagonal ~ {np.degrees(2*np.arctan(np.hypot(W/2,H/2)/np.mean([fx,fy]))):.1f} deg')
print('raw (distorted) image edges, i.e. the angle actually seen at the picture border:')
for name, pix, f in (('left', cx, fx), ('right', W - cx, fx), ('top', cy, fy), ('bottom', H - cy, fy)):
    rd = pix / f
    ru = undistorted_r(rd)
    print(f'  {name:6s}: distorted r={rd:.3f} -> undistorted r={ru:.3f} -> half-angle {np.degrees(np.arctan(ru)):.1f} deg')
l, r = undistorted_r(cx/fx), undistorted_r((W-cx)/fx)
t, b = undistorted_r(cy/fy), undistorted_r((H-cy)/fy)
print(f'  raw horizontal FOV ~ {np.degrees(np.arctan(l)+np.arctan(r)):.1f} deg, raw vertical FOV ~ {np.degrees(np.arctan(t)+np.arctan(b)):.1f} deg')
print(f'  note: distortion polynomial is only trustworthy inside the calibrated field; corner r_d = {np.hypot(cx, cy)/fx:.3f}')
# ground footprint for a camera at height h looking straight down: width = 2 h tan(hfov/2)
for h in (0.10, 0.15, 0.20):
    print(f'  straight-down at {h*100:.0f} cm: pinhole footprint {2*h*np.tan(np.radians(64.4/2))*100:.1f} x {2*h*np.tan(np.radians(50.7/2))*100:.1f} cm')
