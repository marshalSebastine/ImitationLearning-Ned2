"""E1 optics numbers: depth of field / blur, pixels per cm, line width, puck rim row, for the lens heights and tilts in play."""
import math
F_MM = 2.1            # datasheet focal length (assumption: the quoted 2.1 mm is the true focal length)
N_LIST = (2.0, 2.8)   # f-number: ELP 2.1 mm board lenses are F2.0 (assumption), F2.8 as the pessimistic-for-FOV / optimistic-for-DoF bound
FX = 508.0; FY = 506.4           # Niryo intrinsics at 640x480
PIX_UM = F_MM / FX * 1000        # output-pixel pitch on the sensor implied by f and fx (4.13 um)
HALF_V = math.degrees(math.atan(240/FY))   # 25.3 deg half vertical field (pinhole)

def blur_px(u_mm, s_mm, n):
    """Defocus disc diameter in output pixels for an object at u when focused at s (thin lens)."""
    c_mm = (F_MM**2 / n) * abs(u_mm - s_mm) / (u_mm * (s_mm - F_MM))
    return c_mm * 1000 / PIX_UM

print(f"pixel pitch implied by f={F_MM} mm and fx={FX}: {PIX_UM:.2f} um; vertical half field {HALF_V:.1f} deg")
print("\nA) Depth of field with the ring as it is (focus s), CoC = 1 / 2 / 3 px, N = 2.0:")
for s in (50, 61, 70, 90):
    H = F_MM**2 / (2.0 * PIX_UM/1000) + F_MM
    for c_px in (1, 2, 3):
        Hc = F_MM**2 / (2.0 * c_px * PIX_UM/1000) + F_MM
        near = s*(Hc - F_MM)/(Hc + s - 2*F_MM); far = s*(Hc - F_MM)/(Hc - s) if Hc > s else float('inf')
        print(f"   focus {s/10:4.1f} cm, CoC {c_px} px: sharp from {near/10:4.1f} to {far/10:5.1f} cm")

print("\nB) Blur disc (px) at object distance u for focus s = 5.0 / 6.1 / 7.0 cm, N = 2.0 (and N = 2.8 in brackets):")
for u in (20, 30, 45, 60, 70, 84, 93, 96, 105, 119):
    row = f"   u = {u/10:5.1f} cm:"
    for s in (50, 61, 70):
        row += f"  s{s/10:.1f}: {blur_px(u, s, 2.0):4.1f} ({blur_px(u, s, 2.8):4.1f})"
    print(row)

print("\nC) Window geometry per variant (lens height h, tilt t from straight down, lens 3.0 cm from the rod axis along the pan):")
variants = [("E1 as proposed, 64 deg lens", 5.9, 35.0, 3.0), ("E1 proposed, 52 deg assumption (tilt 32)", 5.9, 32.0, 3.0),
            ("E1 with the body under the housing (lens 4.8 cm)", 4.8, 30.0, 3.0), ("E1 tilt 25 (safer for the puck)", 5.9, 25.0, 3.0),
            ("Now: stock bracket, 64 deg fit", 7.0, 15.6, 4.06), ("Now: stock bracket, 52 deg fit", 8.9, 19.4, 3.01)]
for name, h, tilt, ahead in variants:
    print(f" {name}: h {h} cm, tilt {tilt} deg")
    for label, ang in (("bottom row", tilt - HALF_V), ("centre", tilt), ("top row", tilt + HALF_V)):
        if ang <= -85 or ang >= 85: continue
        slant = h / math.cos(math.radians(ang))
        ground = h * math.tan(math.radians(ang))
        # lateral scale (px per cm across the line) ~ fx / slant ; along-track scale: ground cm per px = slant^2/(h*fy) -> px per cm = h*fy/slant^2
        lat = FX / slant; along = h * FY / slant**2
        # distortion factor at that row (centre column): radial position yn = tan(ang - tilt)
        yn = math.tan(math.radians(ang - tilt)); r2 = yn*yn
        fac = 1 - 0.4146*r2 + 0.2407*r2*r2 - 0.1040*r2**3
        dfac = 1 - 3*0.4146*r2 + 5*0.2407*r2*r2 - 7*0.1040*r2**3
        b61 = blur_px(slant*10, 61, 2.0); b50 = blur_px(slant*10, 50, 2.0)
        print(f"    {label:10s}: ray {ang:5.1f} deg, slant {slant:5.1f} cm, paper {ground + ahead:5.1f} cm from rod axis; "
              f"1 cm line {lat:5.0f} px across (x{fac:.2f} with distortion), {along:4.0f} px/cm along track (x{dfac:.2f}); "
              f"blur {b61:3.1f} px (focus 6.1) / {b50:3.1f} px (focus 5.0)")
    # puck front top rim row for puck-centre offsets 2.1 and 2.9 cm from the rod axis, puck top 2.9 cm above paper
    for c in (2.1, 2.9):
        dx = (c + 1.2) - ahead; dz = h - 2.9
        ang_rim = math.degrees(math.atan2(dx, dz))
        below_axis = ang_rim - tilt   # negative = below the optical axis (towards the bottom row)
        row = 240 - FY * math.tan(math.radians(below_axis))
        # hidden paper: ray through the rim reaches the paper at
        hid = ahead + dx * (h / dz)
        print(f"    puck centre {c} cm from rod axis: front top rim at {ang_rim:5.1f} deg from vertical -> row {row:5.0f} "
              f"({'IN frame' if 0 <= row <= 468 else 'OUT of frame'}); paper hidden by the puck out to {hid:4.1f} cm from the rod axis; "
              f"puck top {dz:.1f} cm below the lens -> blur {blur_px(dz*10, 61, 2.0):4.1f} px at focus 6.1")

print("\nD) Puck size in the frame at E1 (lens 3 cm above the puck top): 2.4 cm at 3.0-3.3 cm slant ->", f"{FX*2.4/3.0:.0f}-{FX*2.4/3.3:.0f} px wide")
print("E) Motion: image shift per 1 cm of lens travel at 6 cm = ", f"{FX/6:.0f} px; pan of 20 deg = {math.radians(20)*FX:.0f} px at the centre; at 18 deg/s and 1/30 s exposure: {math.radians(18/30)*FX:.1f} px blur")

print("\nF) Alternatives: E1-far (collar arm, lens 5.5 cm out, body clear of the housing at all pans) and F (wrist stalk):")
alts = [("E1-far: lens 5.5 out, 7.0 up, tilt 0", 7.0, 0.0, 5.5), ("E1-far: lens 5.5 out, 9.0 up, tilt -8 (towards puck)", 9.0, -8.0, 5.5),
        ("F (64 deg): lens 5.1 ahead, 13.9 up, tilt -10", 13.9, -10.0, 5.1), ("F (52 deg): lens 5.1 ahead, 15.9 up, tilt -10", 15.9, -10.0, 5.1)]
for name, h, tilt, ahead in alts:
    print(f" {name}")
    for label, ang in (("bottom row", tilt - HALF_V), ("centre", tilt), ("top row", tilt + HALF_V)):
        slant = h / math.cos(math.radians(ang)); ground = h * math.tan(math.radians(ang))
        lat = FX / slant; along = h * FY / slant**2
        print(f"    {label:10s}: ray {ang:5.1f} deg, slant {slant:5.1f} cm, paper {ground + ahead:5.1f} cm from rod axis; 1 cm line {lat:4.0f} px; "
              f"{along:4.0f} px/cm along; blur {blur_px(slant*10, 61, 2.0):3.1f} px (focus 6.1) / {blur_px(slant*10, 50, 2.0):3.1f} (focus 5.0) / {blur_px(slant*10, 90, 2.0):3.1f} (if refocused to 9)")
    for c in (2.1, 2.9):
        dx = (c + 1.2) - ahead; dz = h - 2.9
        ang_rim = math.degrees(math.atan2(dx, dz)); row = 240 - FY * math.tan(math.radians(ang_rim - tilt))
        print(f"    puck centre {c}: front rim row {row:5.0f} ({'IN' if 0 <= row <= 468 else 'OUT'}); puck top {dz:.1f} cm from lens -> blur {blur_px(dz*10, 61, 2.0):4.1f} px")

print("\nG) Camera body envelope (assumed 37 x 37 mm face, 14 mm deep behind the lens plane, lens at the face centre; measure it!):")
for tilt in (0, 25, 35):
    t = math.radians(tilt)
    up_face = 18.5 * math.sin(t); back_face = 18.5 * math.cos(t)      # top edge of the face
    up_depth = 14 * math.cos(t); back_depth = 14 * math.sin(t)       # rear of the body
    top = up_face + up_depth; back = back_face + back_depth
    print(f"   tilt {tilt:2d}: body top {top:4.1f} mm above the lens centre, rear-top corner {back:4.1f} mm towards the rod from the lens -> "
          f"with the lens 30 mm out the corner is {30 - back:5.1f} mm from the rod axis (rod radius 10.5); to keep the whole body below the housing bottom (5.9 cm) the lens must be at {5.9 - top/10:4.2f} cm")
