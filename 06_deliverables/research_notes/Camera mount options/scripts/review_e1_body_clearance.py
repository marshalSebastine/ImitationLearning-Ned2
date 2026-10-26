"""Sceptical check: does the CAMERA BODY (not the lens point) of the E1 collar designs clear the wrist housing's
bottom plate, the rod and the block over the joint-6 pan range?  Housing outline from Niryo's wrist_link.stl.
Camera body dimensions are ASSUMPTIONS (to be measured): case 38 x 38 x 15 mm, lens barrel d14 x 12 mm in front."""
import numpy, struct, math
from scipy.spatial import ConvexHull, Delaunay

def read_stl(path):
    with open(path,'rb') as f:
        f.read(80); n=struct.unpack('<I',f.read(4))[0]
        d=numpy.frombuffer(f.read(n*50),dtype=numpy.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')]))
    return d['v'].reshape(-1,3)
base="D:/AI_Project/06_deliverables/research_notes/Camera mount options/niryo_files/"
w=read_stl(base+"ned2_wrist_link.stl")*1000.0
# mesh frame: hand face at x_mesh=19.7 (extreme), hand axis at (y=9.25, z=0); "height above hand face" h = 19.7 - x_mesh
h_all = 19.7 - w[:,0]
r_all = numpy.hypot(w[:,1]-9.25, w[:,2])
wide = r_all > 15.0
plate_h = h_all[wide].min()
print(f"lowest wide-body vertex (r>15 mm): {plate_h:.1f} mm above the hand face;  r>12: {h_all[r_all>12].min():.1f};  r>25: {h_all[r_all>25].min():.1f}")
sel = (h_all >= plate_h-0.5) & (h_all <= plate_h+30)
pts = numpy.column_stack([w[sel,1]-9.25, w[sel,2]])   # (u,v) around the hand axis
hull = ConvexHull(pts); tri = Delaunay(pts[hull.vertices])
print(f"housing outline (convex hull of vertices {plate_h:.0f}..{plate_h+30:.0f} mm above the hand face): max radius from rod axis {numpy.hypot(*pts.T).max():.1f} mm")
ang=numpy.degrees(numpy.arctan2(pts[:,1],pts[:,0])); rr=numpy.hypot(*pts.T)
for a0 in range(-180,180,30):
    m=(ang>=a0)&(ang<a0+30); print(f"   sector {a0:4d}..{a0+30:4d} deg: outline reaches {rr[m].max():5.1f} mm" if m.any() else "")

CASE, DEPTH, BARREL_L, BARREL_R = 38.0, 15.0, 12.0, 7.0
def body_points(out_mm, h0_mm, tilt_deg, pan_deg):
    """Sample points of the camera body for a lens at (out_mm from the rod axis, h0_mm above the hand face),
    optical axis tilt_deg from straight down towards 'out', the whole thing rotated by pan_deg about the rod axis."""
    t=math.radians(tilt_deg)
    fwd=numpy.array([1.0,0.0,0.0]); up=numpy.array([0.0,0.0,1.0]); side=numpy.array([0.0,1.0,0.0])
    axis = math.sin(t)*fwd - math.cos(t)*up          # optical axis (down and forward)
    pup  = math.cos(t)*fwd + math.sin(t)*up          # picture 'up' direction, in the body plane
    lens = numpy.array([out_mm,0.0,h0_mm])
    pts=[]
    # barrel: cylinder from the lens back BARREL_L along -axis
    for s in numpy.linspace(0,BARREL_L,7):
        for a in numpy.linspace(0,2*math.pi,12,endpoint=False):
            pts.append(lens - s*axis + BARREL_R*(math.cos(a)*pup + math.sin(a)*side))
    # case: box behind the barrel
    front = lens - BARREL_L*axis
    for d in numpy.linspace(0,DEPTH,4):
        for a in numpy.linspace(-CASE/2,CASE/2,9):
            for b in numpy.linspace(-CASE/2,CASE/2,9):
                pts.append(front - d*axis + a*pup + b*side)
    P=numpy.array(pts)
    c,s=math.cos(math.radians(pan_deg)),math.sin(math.radians(pan_deg))
    R=numpy.array([[c,-s,0],[s,c,0],[0,0,1]])
    return P@R.T

def collides(P):
    inside_housing = (P[:,2] > plate_h) & (tri.find_simplex(P[:,:2]) >= 0)
    r=numpy.hypot(P[:,0],P[:,1])
    inside_rod = (r < 10.5+0.5) & (P[:,2] > -0.5) & (P[:,2] < 28.5)   # rod cylinder (z measured up from hand face: rod goes DOWN to -28.5!)
    return inside_housing.any(), None

# NOTE on signs: the rod hangs DOWN from the hand face. In this script h is 'above the hand face' (towards the wrist),
# so the rod occupies h in [-28.5, 0] and the block h in [-28.5-30, -28.5]. The housing plate is at h = plate_h > 0.
def rod_hit(P):
    r=numpy.hypot(P[:,0],P[:,1]); return ((r < 10.5) & (P[:,2] < 0) & (P[:,2] > -28.5)).any()
def block_min_r(P):
    m=(P[:,2] < -28.5) & (P[:,2] > -58.5); return numpy.hypot(P[m,0],P[m,1]).min() if m.any() else float('inf')
def lowest(P): return P[:,2].min()

designs = {
 "E1 as in option_geometry (lens 30 out, at hand-origin height, tilt 32)": (30,0,32),
 "E1 lens 30 out, 10 mm below hand origin, tilt 32": (30,-10,32),
 "E1 lens 30 out, 20 mm below hand origin, tilt 32": (30,-20,32),
 "E1b lens 40 out, hand-origin height, tilt 14": (40,0,14),
 "E1b lens 40 out, 15 mm below, tilt 14": (40,-15,14),
 "E1d lens 35 out, 12 mm below, tilt 30": (35,-12,30),
 "E1d' lens 35 out, 20 mm below, tilt 30": (35,-20,30),
 "far-out lens 55 out, hand-origin height, tilt 32": (55,0,32),
 "far-out lens 60 out, hand-origin height, tilt 40": (60,0,40),
}
pans=list(range(-180,181,5))
print("\nPan angles (deg, 0 = camera pointing to the wrist's camera-side face... taken as the direction where the rod axis is 9 mm from the housing face) at which the camera BODY enters the housing plate volume;")
print(f"housing plate at h={plate_h:.1f} mm above the hand face. Camera body assumed {CASE:.0f}x{CASE:.0f}x{DEPTH:.0f} mm + barrel d{2*BARREL_R:.0f}x{BARREL_L:.0f} mm.")
for name,(out,h0,tilt) in designs.items():
    bad=[p for p in pans if collides(body_points(out,h0,tilt,p))[0]]
    P0=body_points(out,h0,tilt,0)
    rodhit=any(rod_hit(body_points(out,h0,tilt,p)) for p in (0,90,180))
    print(f"\n{name}:")
    print(f"   body top {P0[:,2].max():.1f} mm above hand face (plate at {plate_h:.1f}); lowest body point {lowest(P0):.1f} mm (rod end at -28.5, block bottom at about -58.5, puck top at about -{5.95*10-29:.0f})")
    print(f"   body hits the housing at {len(bad)} of {len(pans)} pan angles" + (f": free sector only {[p for p in pans if p not in bad][0]}..{[p for p in pans if p not in bad][-1]} deg" if bad and len(bad)<len(pans) else ""))
    print(f"   body inside the rod cylinder at some pan: {rodhit};  min radial distance of body points in the block's height band: {block_min_r(P0):.1f} mm (block corner at about 21 mm if 30x30, 25 mm if 40x30)")
