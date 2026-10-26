"""2D (vertical plane along the push direction) check of the visible stretch of the line ahead of the puck for a lens at
(ahead_mm, height_mm) tilted tilt_deg from straight down, with the puck as a 24 x 29 mm rectangle centred 29 mm ahead of
the rod axis. Line points are visible if inside the vertical FOV and the sight line misses the puck rectangle.
Also: camera-body top height for a box of depth d along the optical axis and face half-height f/2 (worst corner)."""
import math
import numpy as np
def visible(ahead, height, tilt, vfov, puck_c=29.0, puck_r=12.0, puck_h=29.0, far=200.0):
    lo, hi = math.radians(tilt - vfov / 2), math.radians(tilt + vfov / 2)
    vis = []
    for x in np.arange(puck_c + puck_r, far, 1.0):
        ang = math.atan2(x - ahead, height)   # angle from straight down
        if not (lo <= ang <= hi):
            continue
        # sight line from (ahead, height) to (x, 0): check against puck box x in [c-r, c+r], z in [0, h]
        blocked = False
        for px in (puck_c - puck_r, puck_c + puck_r):
            if (px - ahead) * (x - ahead) > 0 and abs(px - ahead) < abs(x - ahead):
                t = (px - ahead) / (x - ahead); z = height * (1 - t)
                if 0 <= z <= puck_h: blocked = True
        # top face
        t_top = 1 - puck_h / height
        xt = ahead + t_top * (x - ahead)
        if puck_c - puck_r <= xt <= puck_c + puck_r: blocked = True
        if not blocked: vis.append(x)
    if not vis: return None
    return (min(vis) - puck_c, max(vis) - puck_c)   # from the puck centre
print('lens ahead / height / tilt / vFOV -> visible line from the puck centre (near, far) in mm; None = nothing')
for vfov, label in ((40.2, '52deg lens'), (50.7, '64deg lens')):
    print('==', label)
    for ahead, height, tilt in ((30, 59, 32), (30, 50, 32), (30, 45, 32), (30, 40, 32), (30, 40, 45), (30, 40, 55),
                                (35, 47, 30), (40, 59, 14), (40, 45, 30), (45, 45, 30), (45, 40, 40), (50, 45, 35)):
        print(f'   ahead {ahead:3d} height {height:3d} tilt {tilt:3d}: {visible(ahead, height, tilt, vfov)}')
print()
print('camera body top above its lens for a box (depth d along the axis, face f across) tilted t from vertical, lens mid-face:')
for d, f, t in ((20, 40, 32), (25, 40, 32), (20, 40, 14), (15, 38, 32), (25, 45, 32)):
    top = d * math.cos(math.radians(t)) + (f / 2) * math.sin(math.radians(t))
    print(f'   d {d} f {f} tilt {t}: body top {top:4.1f} mm above the lens; lowest face edge {(f/2)*math.sin(math.radians(t)):4.1f} mm below the lens')
