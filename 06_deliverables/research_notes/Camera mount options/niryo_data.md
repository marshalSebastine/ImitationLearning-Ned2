# Niryo Ned2 manufacturer data for camera-mount design

Collected 2026-10-26. Everything here is from Niryo's own publications (docs.niryo.com, niryo.com, the NiryoRobotics GitHub organisation, Niryo datasheets hosted by RS) unless a line says otherwise. Numbers that I derived myself (mesh parsing, field-of-view arithmetic, frame conversions) are marked **derived**. Things I could not find are listed in section 6.

Downloaded files live in `niryo_files\` next to this note (full list with sizes in section 7). They are untrusted data: nothing in that folder was executed; STL/STEP/PDF were parsed with scripts kept in the session scratchpad and run with `py -3.13 -I`.

---

## 1. Vision Set camera

### 1.1 Camera module and sensor

| Item | Value | Source |
|---|---|---|
| Camera model | **ELP-USBFHD06H-L21** (ELP USB module) | Niryo "VISION SET technical specifications" datasheet, hosted by RS: https://assets.rs-online.com/v1678366719/Datasheets/74f80c52e6e10f4bf9381f9dbe4fe1be.pdf (local copy `rs_vision_set_datasheet_74f80c52.pdf`); also RS product page https://uk.rs-online.com/web/p/vision-sensors/2052814 |
| Sensor | Sony IMX322, CMOS, electronic rolling shutter / frame exposure | same datasheet; RS Ned2 Vision Set page https://uk.rs-online.com/web/p/vision-sensors/2412218 |
| Lens | "Lens size (mm): 2.1" (i.e. a 2.1 mm focal-length board lens) | same datasheet |
| "Pixel size (mm)" | printed as "12.8 x 11.6" in the datasheet. Taken literally that is impossible for a pixel; it is most likely a mislabelled module/sensor dimension. Reported verbatim, do not use. | same datasheet |
| Image area | "2000(H) x 1121(V) approx. 2.24 M pixels" | same datasheet |
| Maximum resolution | 1920 x 1080 | same datasheet |
| Resolution / frame-rate modes | Compressed and raw: 320x240@30 fps, 352x288@30, 640x360@30, 640x480@30. Raw only: 800x600@15. Compressed only: 800x600@30, 1280x720@30, 1920x1080@30. Raw format YUV; compressed H264 and MJPEG | same datasheet |
| S/N, dynamic range, sensitivity, min. illumination | 42 dB; 86 dB; 5.0 V/lux-sec @550 nm; 0.01 lux | same datasheet |
| Interface and cable | USB 2.0 High Speed, plug and play; USB cable length 1 m; kit lists the cable as "PH-4 to USB 1m" | datasheet; docs Kit Overview https://docs.niryo.com/accessories/vision-set/article/ |
| Power | 5 V, max 190 mA (bus powered) | RS page 2052814 |
| Working temperature | -20 to 85 C | datasheet |
| What Niryo's software actually uses | 640 x 480 (`width_img: 640`, `height_img: 480`); `frame_rate: 15` with the comment "15, 25 or 30"; `flip_img: true`; stream JPEG quality 55 | ned_ros `niryo_robot_vision/config/cam_intrinsics.yaml` and `config/ned2/video_server_setup.yaml` (local copies `ned_ros_vision_cam_intrinsics.yaml`, `ned_ros_vision_ned2_video_server_setup.yaml`): https://github.com/NiryoRobotics/ned_ros/tree/master/niryo_robot_vision/config |

Horizontal and vertical field of view: **Niryo does not publish a figure.** The only numbers that exist are:

* Published camera intrinsics (ned_ros `cam_intrinsics.yaml`, 640x480): fx = 508.069, fy = 506.445, cx = 332.821, cy = 242.984; distortion k1 = -0.4146402, k2 = 0.2407384, p1 = p2 = 0, k3 = -0.1040042. The Gazebo camera in `niryo_ned2_camera.urdf.xacro` uses the same distortion and `<horizontal_fov>1.21</horizontal_fov>` with the comment "69,33 degrees", 640x480, 15 Hz.
* **Derived** from those intrinsics (script `fov_from_intrinsics.py` in the scratchpad): pinhole (undistorted) FOV = **64.4 deg horizontal, 50.7 deg vertical, 76.5 deg diagonal**. Undoing the barrel distortion at the picture borders gives the angle the raw 640x480 picture actually reaches: about **75 deg horizontal, 55 deg vertical** (left 39.4, right 36.0, top 28.1, bottom 27.4 deg half-angles; the polynomial is only valid inside the calibrated field, so treat the raw-border numbers as +-3 deg). Straight-down footprint of the pinhole field at 10 / 15 / 20 cm height: 12.6 x 9.5 / 18.9 x 14.2 / 25.2 x 19.0 cm.
* Not verified: the login-gated Niryo Academy course text that a search engine quoted ("field of view approximately 104 degrees, focal length approximately 3 mm, pixel size 12.8 x 11.6 um") at https://academy.niryo.com/course/view.php?id=57 (login page only). Vendor listings for ELP 2.1 mm modules quote "100 degree" to "170/180 degree" lenses for sibling models; none names the L21 variant. I could not confirm any of these.
* The team's measured ~52 deg equivalent horizontal field is narrower than every published or derived figure; the difference is worth checking against the lens mounting (the raw picture at 640x480 may be a centre crop of the 16:9 sensor, which would narrow the horizontal field) before it is used in a mount design.

### 1.2 Physical size of the camera module and its bracket

Niryo publishes no dimension for the camera module or the bracket. What exists:

| Item | Value | Source |
|---|---|---|
| URDF/Gazebo stand-in | box 0.01 x 0.037 x 0.037 m (cameraWidth 0.01, cameraHeight 0.037), cameraMass 0.05 kg | `niryo_ned2_camera.urdf.xacro` (local copy) |
| Official printable camera holder for the **original Ned** (not Ned2) | `CAMERA HOLDER VISION NED.STL` (header "SUPPORT CAMERA VISION NED"): bounding box **37.0 x 22.0 x 43.5 mm**; base plate 34 x 22 x 3 mm with four 3.2 mm holes at x = +-13.5, y = +-8 mm (27 x 16 mm pattern, M3 clearance); camera face with a 16 mm lens hole and four 1.8 mm holes at x = +-14, 28 mm apart vertically (28 x 28 mm board pattern). `CAMERA COVER VISION NED.STL`: **37 x 37 x 5.7 mm** with two diagonal screw holes (2.4 mm through, 4.6 mm counterbore) at (+-14, -+14) mm. **Derived** by parsing the STLs. | https://github.com/NiryoRobotics/ned/tree/master/STL/25-KV%20STL%20EN (local copies `ned_v1_CAMERA_HOLDER_VISION_NED.STL`, `ned_v1_CAMERA_COVER_VISION_NED.STL`) |
| Board size | ELP boards of this family are listed by resellers as 38 x 38 mm (also 32 x 32), about 30 g; consistent with the 37 mm cover and 28 mm hole pattern above, but not confirmed by Niryo or ELP for the L21 | reseller listings found via search (eBay/desertcart), unverified |
| Kit contents (Ned2) | 1 camera (black housing with lens, see photo `docsie_vision_kit_overview.jpg`), lens cap, 2 screws, calibration tip, workspace board, workspace support, mechanical connector (aluminium base), 3 squares, 3 circles, USB cable PH-4 to USB 1 m, 2 "scratches" (hook-and-loop straps to tie the cable to the arm) | https://docs.niryo.com/accessories/vision-set/article/ ; RS page 2412218 ("2 x Screws to fix the camera on the support") |

### 1.3 How and where the bracket attaches on the Ned2

Official setup page (JavaScript-rendered; read with the browser): https://docs.niryo.com/accessories/vision-set/setup/ — section "Hardware / The Vision wrist":

> Step 1. Place the screw  
> Step 2. tighten the first screw with the given tool  
> Step 3. tighten the second screw  
> Step 4. Pass the cable through the end effector, between the motor and the aluminum brace

Ned2 user manual, "Vision Set mounting" (https://docs.niryo.com/robots/ned2/mechanical-interface/ and the PDF `ned2_user_manual_v1.0.0_generationrobots.pdf`, p. 23): "To mount the Vision Set, all you need to do is attach the camera to Ned2's wrist. The cable of the vision set need to pass through the wrist to avoid trouble during movement of the robot."

The four step photos (local copies `docsie_vision_setup_step1..4_*.jpg`, 2000x1333) show:

* The camera sits in a **black 3D-printed holder with two side ears**; each ear takes one hex-socket screw (the two screws in the kit; an Allen key is "the given tool"). The screws go into a machined **aluminium brace of the wrist** that also carries the joint-6 motor (visible: the motor, its silver shaft collar, and the brace with a rectangular window). The holder clamps around the front of that brace, under the rectangular opening in the blue wrist cover, right next to the hand shaft.
* The mounted camera (step 4 photo) hangs on the wrist beside the hand, lens looking along the tool direction, with the cover closed around it; the cable runs inward between the motor and the brace and then out of the wrist.
* So the bracket is on the **wrist link** (the body between joint 5 and joint 6), which matches the URDF (`camera_joint` parent `wrist_link`) and the team's photo evidence. The camera does not turn with joint 6.
* **Screw size and hole spacing are not published.** The STEP of the Ned2 (section 2.4) contains the wrist as one solid but not the camera holder, and its small holes are listed in 2.2; the holder's own screw pattern must be measured on the robot (two screws, one per ear, about 40-45 mm apart judging from the photo scale against the 21 mm hand; this is a guess, not data).

Cable route: camera cable through the wrist, tied to the arm with the two straps, to a USB port on the **back panel** (2 x USB 2.0, 2 x USB 3.0; the camera is a USB 2.0 device) — https://docs.niryo.com/robots/ned2/electrical-interface/ ("1: 2 x USB 2.0, 2: 2 x USB 3.0, 3: RJ45 ..."). Vision Set warnings page: "Be careful that the cable does not wrap around Ned"; "Take the lens protection off ... Do not touch the lens" (https://docs.niryo.com/accessories/vision-set/warnings-typical-issues/).

### 1.4 Camera pose in the URDF

`niryo_ned2_camera.urdf.xacro` (ned_ros master):

```
camera_joint: parent wrist_link, child camera_link
  origin xyz = (cameraHeight/2 + 0.02, 0, 0) = (0.0385, 0, 0) m
  rpy = (0, 0, -PI/2 + 10 deg)  -> yaw -80 deg
camera_link: box 0.01 x 0.037 x 0.037 m, mass 0.05 kg (gazebo camera, 640x480, 15 Hz)
```

**Derived** (wrist_link frame, see 2.1 for the frame): the camera box centre is 38.5 mm from the joint-5 axis along +x, 19.7 mm behind the wrist/hand interface plane along the tool axis and 29.25 mm laterally off the joint-6 (tool) axis; the Gazebo optical axis is (0.174, -0.985, 0) in the wrist frame, i.e. along the tool direction (-y) tilted 10 deg away from the tool axis. The original Ned's xacro uses origin (0, cameraHeight/2 + 0.02, 0), rpy (-PI/2, 0, 0) instead. These are simulation stand-ins, not measured poses.

---

## 2. Ned2 wrist, hand and tool interface

### 2.1 Link frames (ned_ros `niryo_ned2.urdf.xacro`, local copy)

```
joint_5: origin xyz (0 0 0.17) rpy (-PI/2 0 0), parent forearm_link, child wrist_link, axis z
joint_6: origin xyz (0.00925 -0.0197 0) rpy (PI/2 0 0), parent wrist_link, child hand_link, axis z
hand_tool_joint: origin xyz (0 0 0.0285) rpy (PI -PI/2 0), parent hand_link, child tool_link
wrist_link visual/collision mesh origin rpy (0 0 -PI/2); hand_link mesh origin rpy (0 0 -PI/2)
wrist_link inertial: mass 0.22126 kg, com (-0.019666, 0.037312, 0); hand_link mass 0.0070027 kg
```
The STL/STEP meshes are stored in a "mesh frame" that is the link frame rotated by +90 deg about z. **Derived** conversion: x_link = y_mesh, y_link = -x_mesh, z_link = z_mesh (check: mesh hand-axis point (19.7, 9.25, 0) mm -> link (9.25, -19.7, 0) = joint_6 origin).

### 2.2 wrist_link mesh (`meshes/ned2/stl/wrist_link.stl`, 57,184 bytes, binary STL in metres)

**Derived** by parsing (script `stl_bbox.py`, `stl_planes.py`):

* 1142 triangles, 571 unique vertices. Bounding box, mesh frame: x [-49.0, 19.7], y [-18.0, 18.25], z [-29.5, 23.0] mm -> **68.7 x 36.25 x 52.5 mm**. In the link frame: x [-18.0, 18.25] (camera side is +x), y [-19.7, 49.0] (hand face at y = -19.7), z [-29.5, 23.0] (joint-5 axis).
* Joint-5 bearing features: circles d 19.5 mm centred on the z axis on the planes z = +17 and z = +23 mm; d 6 mm hole with d 9 mm counterbore on the opposite end (z = -27 / -29.5).
* Joint-6 (hand) axis features: circles d 8 / 17 / 17.5 mm centred on the hand axis (y_mesh = 9.25) on the planes x_mesh = 19.7 (hand face), 11, -11, -13.3, -21.8, -25; d 7 mm at the hand face.
* The NED2 STEP (section 2.4) contains the same wrist as one solid (SolidWorks name "Cong4", 34 cylindrical surfaces). Small holes in it (link frame, mm, **derived** with `step_faces.py`):
  * four **d 2.0 mm** holes parallel to x, at (y = +-10, z = +-16), running from the -x face (x = -18) to x = +10 (28 mm long) — these open on the face opposite the camera side;
  * two **d 3.0 mm** holes parallel to the tool axis (y) at x = 10, z = +-15.5, from y = +10 to y = -10;
  * two **d 4.0 mm** holes parallel to the tool axis at x = 13.5, z = +-7, from y = +10 to y = -10;
  * no hole opens on the +x (camera) face in the STEP; the camera holder therefore does not screw into the plastic wrist cover but into the internal aluminium brace, as the setup photos show.
* The official printable wrist (`ned2_print_6-WRIST-V1.STL`, in mm, assembly pose) has the identical 68.7 x 52.5 x 36.25 mm box and the same features; it is the same body as the ROS mesh.

### 2.3 hand_link mesh (`meshes/ned2/stl/hand_link.stl`, 118,084 bytes) = the "rod"

**Derived**:

* 2360 triangles. Bounding box: x, y [-10.5, 10.5], z [-6.5, 28.5] mm -> **21 x 21 x 35 mm** cylinder along the joint-6 axis; tool_link (distance_hand_tool 0.0285) is the far end face.
* Profile along the axis (from the STL slices): d 15 mm neck for z = -5 ... 0 mm, cone widening to d 21 mm by z = 8.6 mm, d 21 mm up to z = 27.5, chamfer to the end at 28.5. The end face is not a full disc (extent x +-9.5, y -8.8 ... +9.5 mm) and has a **13 mm bore** (r 6.5) that runs from z = 27 down to z = 9.6 (17.4 mm deep). Below it: d 8 mm bore z = -6 ... 0 and a **d 4.3 mm** hole z = 0 ... 7.65 (M4 clearance). The STEP assembly contains a "VIS FHC M4x16 - 8.8" (countersunk M4 x 16 screw) solid, so the hand is held to the joint-6 shaft by one central M4 screw entered from the tool end.
* Small holes (STEP, hand frame, **derived**): four d 2 mm at (x +-1.94, y +-2.5), z 0 ... -6 (base, motor-horn pattern); two d 2 mm cross holes at x = +-3.5, z = 27.5; two d 2 mm cross holes at x = +-1.5, z = 12.5. A notch is visible in the STL at z 10-13 mm on one side (y -6 ... -10).
* Circular patterns on the tool face: only the concentric 13 mm / 8 mm / 4.3 mm bores; there is **no bolt circle** on the Ned2 hand face.
* STEP solid name: "SUPPORT HAND - NED 2". Ned2 printable `ned2_print_7-HAND-V1.STL` is the same part (35 x 21 x 21 mm).

### 2.4 Official tool interface ("EasyConnect")

* Manual (p. 22, docs "Tool mounting"): "Each tool has the same mechanical connector interface. This option allows a quick-change option of the tool and good stability. Put the groove on the side of the motor, pointing down; fix the gripper to the robot axis using the magnetic part." The figure `docsie_ned2_mech_tool_mounting_outils2.jpg` labels the hand's **positioning groove**, the tool's **holding magnet**, and the **electrical connection** (tool cable to the wrist control panel: "GRIPPER" 4-pin and "ELECTROMAGNET" 5V/GND/DO connectors on the wrist panel, photo `docsie_vision_setup_step4_cable_through_end_effector.jpg`). Source: https://docs.niryo.com/robots/ned2/mechanical-interface/ ; https://docs.niryo.com/robots/ned2/electrical-interface/
* Electromagnet STEP (`ELECTROMAGNET_ned2.step`, AP242, units metres): three solids — "ELECTROMAGNET SUPPORT" (about 40 x 30 x 30 mm), "ELECTROMAGNET COVER" (coil housing diameters 20.5 / 24.5 / 26 mm) and **"D13 - M3 POT MAGNET THREADED SLEEVE"** (13 mm diameter pot magnet with M3 thread). **Derived**: the 13 mm pot magnet is what plugs into the hand's 13 mm bore and holds against the steel M4 screw head; that is the "magnetic part".
* Tool specifications (Ned2 user manual pp. 13-17, also https://docs.niryo.com/robots/ned2/article/):

| Tool | Max operating width | Picking distance from end-effector base | Motor / control | Weight | Supply |
|---|---|---|---|---|---|
| Custom Gripper | 43 mm | 85 mm | XL330 servo | 87 g | 5 V |
| Large Gripper | 79.2 mm | 109.35 mm | XL330 | 72.5 g | 5 V |
| Adaptive Gripper | 97.49 mm | 113.79 mm | XL330 | 83 g | 5 V |
| Electromagnet | 20 mm (picking surface 20 mm) | **27 mm** | digital output | **44.5 g** | 5 V |
| Vacuum Pump | vacuum tube 20 mm | 47.5 mm | XL330 (pump control), payload 300 g | 18 g head / 183 g total | 5 V |

  Product pages add: Custom Gripper 77 x 39.3 x 104 mm (https://niryo.com/product/custom-gripper/); Vacuum Pump 105 x 216.2 x 61.4 mm, "magnetic box", "Snaplink technology" (https://niryo.com/product/vacuum-pump/). No Niryo page gives the rod/hand diameter or length in text; the numbers above come from the meshes.

* Team measurement vs. spec: the team reports the magnet tip 2.6 cm below the rod end and 1.3 cm wide; Niryo's electromagnet spec is 27 mm picking distance and 20 mm picking surface (the 13 mm figure matches the D13 pot magnet, not the coil face). Worth re-measuring which face is meant.

### 2.5 Does Niryo publish STEP files? Yes

Download Center https://niryo.com/resources/download-center/ lists "Ned2 3D STL" and "Ned2 3D STEP"; both short links redirect to GitHub (`https://niryo.com/ned2-step` -> https://github.com/NiryoRobotics/ned2/blob/master/NED2_STEP.step ; `https://niryo.com/ned2-stl` -> https://github.com/NiryoRobotics/ned2/tree/master/STL). Repository contents (https://github.com/NiryoRobotics/ned2):

| File | Size | Downloaded |
|---|---|---|
| NED2_STEP.step (whole robot, AP203, SolidWorks 2020, 8 solids incl. the wrist "Cong4" and "SUPPORT HAND - NED 2", mm) | 7,250,770 B | yes -> `NED2_STEP.step` |
| ELECTROMAGNET.step | 266,976 B | yes -> `ELECTROMAGNET_ned2.step` |
| CUSTOM_GRIPPER_NED2_STEP.STEP | 6,894,989 B | no |
| LARGE_GRIPPER_NED2_STEP.STEP | 7,880,772 B | no |
| ADAPTATIVE_GRIPPER_NED2_STEP.STEP | 10,439,704 B | no |
| STL/6- WRIST - V1.STL, STL/7- HAND - V1.STL (+ base, shoulder, arm, elbow, forearm, gripper, safety box) | 167,784 B / 214,684 B | yes -> `ned2_print_6-WRIST-V1.STL`, `ned2_print_7-HAND-V1.STL` |

There is **no separate wrist or hand STEP**; both are solids inside NED2_STEP.step. **No STEP/STL of the Vision Set camera holder for Ned2 is published.** For the original Ned, https://github.com/NiryoRobotics/ned has printable STLs of the whole robot including the camera holder and cover (downloaded, section 1.2) and the electromagnet hand. Ned3 Pro files: https://github.com/NiryoRobotics/ned3pro.

### 2.6 Overall dimensions (official drawing)

`docsie_ned2_mech_3dview_a.jpg` / `_b.png` from https://docs.niryo.com/robots/ned2/mechanical-interface/: footprint 181.6 x 187.5 mm; base-to-joint-2 height 166.3 mm; upper arm 221 mm; forearm + wrist 235 mm (horizontal in the drawn pose); height 419.8 mm in that pose; reach R 498 mm; vertical workspace arc 128.4 deg; reach below table level 229 mm. The URDF values (joint_1 z 0.1013, joint_2 z 0.065 -> 166.3 mm; joint_3 y 0.221) agree.

---

## 3. Specifications that bound a mount

| Item | Value | Source |
|---|---|---|
| Payload | **300 g**; the 2024/2025 datasheets add "(z = 20 mm TCP)" | https://niryo.com/wp-content/uploads/2026/06/10001874-4_TECHNICAL_SPECIFICATIONS_NED2.pdf (local `ned2_technical_specifications_10001874-4.pdf`); RS datasheet `rs_niryo_datasheet_A700000015717952.pdf`; manual p. 11; https://docs.niryo.com/robots/ned2/technical-specifications/ |
| Weight / reach | 7 kg; reach 440 mm (manual 2022) or 490 mm (datasheets and docs 2025) | same |
| Joint ranges | datasheet: J1 335, J2 125, J3 160, J4 235, J5 220, J6 290 deg; URDF limits as in the project context | datasheet; `niryo_ned2_param.urdf.xacro` |
| Joint speeds | 2025 datasheet: J1 45, J2 30, J3 45, J4-J6 90 deg/s; 2024 datasheet: J1/J3 90, J2 60, J4-J6 180 deg/s, tool typical 0.2 m/s; docs: J5 1.57 rad/s, J6 1.775 rad/s, TCP max 468 mm/s | datasheets; docs technical specifications |
| Repeatability | +-0.5 mm (ISO 9283) | datasheets |
| Loads on the wrist / accessories | "Never use loads greater than those stated in the robot specifications as this could cause mechanical failures." "Be careful not to mount a gripper that is not supplied by Niryo. If you would like to design your own gripping tool, contact our Support Service." | manual p. 23; https://docs.niryo.com/robots/ned2/mechanical-interface/ |
| Modifications | Safety section "Hardware and/or software modifications: Any modification of the equipment is prohibited. ... Niryo disclaims all liability for any damage caused to and by the robot as a result of these modifications." (The professor's FAQ explicitly allows a modified camera mount; this is the vendor's warranty position, not the course rule.) | manual pp. 5-6; https://docs.niryo.com/robots/ned2/safety-instructions/ |
| Camera mass in the model | 0.05 kg (URDF stand-in; real module not published; ELP boards ~30 g per resellers) | xacro |
| Cable routing | through the wrist, between the motor and the aluminium brace; strapped to the arm; "Be careful that the cable does not wrap around Ned" | setup and warnings pages |
| Back panel | 2 x USB 2.0, 2 x USB 3.0, RJ45, Wi-Fi, 3 DI, 3 DO, 2 AI, 2 AO, conveyor x2, IR sensor, vacuum pump control, power; I/O supply 5 V / 7 A and 12 V / 7 A | https://docs.niryo.com/robots/ned2/electrical-interface/ ; datasheet |
| Wrist control panel | FreeMotion / Save / Custom buttons, GRIPPER (4-pin) and ELECTROMAGNET (5V, GND, DO) connectors | photo `docsie_vision_setup_step4_cable_through_end_effector.jpg`; manual p. 31 |

---

## 4. Alternative camera positions and custom mounts (Niryo and users)

* **Niryo:** nothing published. The only official mounting is the wrist holder above. The Ned (v1) docs show the same idea on the older wrist (`docsie_ned_v1_camera_vision_set.png`: camera on top of the wrist, lens looking along the tool; https://docs.niryo.com/robots/ned/mechanical-interface/ "Vision Set mounting"). The old Sphinx docs (docs.niryo.com/product/vision-set/..., archive-docs.niryo.com) are gone: the new Docsie site returns "does not exist" for those paths and archive-docs.niryo.com no longer resolves. The Niryo forum domain forum.niryo.com no longer resolves either.
* **Thingiverse** (search "niryo", 65 results, almost all Kia Niro car parts): only one Niryo-robot part, "Niryo Ned end effector adapter" by Keylinsky, 10 Dec 2021, GPL — an adapter to use Niryo One tools on the Ned (v1), not a camera mount: https://www.thingiverse.com/thing:5160146 . No Ned/Ned2 camera mount on Thingiverse.
* **Printables:** search page is behind a bot check in both browsers; not searched (do not bypass). **Cults3D:** web search returned only generic camera mounts, nothing for Niryo.
* **GitHub** (search "niryo ned2", "niryo camera"): no repository with a custom Ned2 camera bracket. Related: ncbdrck/niryo_ned2_description_extras (simulation only, adds a head-mounted Kinect v2 to the Ned2 URDF); Layan-Barakat/niryo-ned2-vision-pick-place (uses an M5Stack UnitV camera with a Ned2; mount not described); 11hayri11/niryo-ned2-ip (Charuco hand-eye calibration of the wrist camera); Sarxell/Ficton_niryo ("testing the robot niryo using a usb camera"). GitHub issue search in the NiryoRobotics organisation for "camera mount"/"camera wrist": no hits.
* **Unverified anecdote:** a reviewer quoted on qviro.com / GlobalSpec: "I equipped its wrist with an Intel RealSense camera and the engines suffered a lot." The page sits behind a bot check and a 403, so I could not read it; treat it as a hint that extra wrist mass hurts joints 5/6, nothing more (https://qviro.com/product/niryo/ned2-educational-robot).
* **Niryo Academy** "From Camera to Grasping with the Niryo Ned2" (https://academy.niryo.com/course/view.php?id=57) describes the wrist camera as a monocular camera that moves with the arm; login required, not read.

---

## 5. Points that matter for the mount decision (facts above, summarised)

1. The camera is fixed to the wrist link's aluminium brace with two screws under the cover opening; it cannot be turned with joint 6, and it sits ~29 mm off the tool axis and ~20 mm behind the hand face (URDF model). A new mount would either replace the printed holder (same two screws into the brace) or clamp onto the 21 mm hand (then it would turn with joint 6 and must not foul the tool's 13 mm magnet plug and cable).
2. Published field of view is ~64 deg (pinhole) / ~75 deg (raw border) horizontal from Niryo's own intrinsics, not 104 deg; the team's 52 deg measurement should be reconciled before any geometry is designed around it.
3. Payload 300 g at 20 mm TCP height; the electromagnet already uses 44.5 g, the camera about 30-50 g. The vendor forbids loads above spec and disclaims modified hardware; the professor's FAQ allows a printed camera mount.
4. STEP of the whole robot (with wrist and hand as separate solids) and the electromagnet are available and downloaded; they are enough to model a holder. The one missing piece is the Vision Set holder itself, which has to be measured on the robot (two screws, their spacing and thread).

---

## 6. Not found

* Numeric horizontal/vertical FOV from Niryo (only intrinsics and the Gazebo 69.33 deg comment).
* Physical dimensions and weight of the Ned2 camera module and of its black holder; screw size and spacing of the holder; any STEP/STL of the holder.
* Rod/hand diameter or length in any Niryo text (only in meshes).
* Any Niryo or community document on alternative camera positions for Ned/Ned2.
* The old Sphinx "Vision Set" setup page (with the 25-step Niryo One wrist procedure) and the 2020 Vision Set PDF at stg.robu.in (host unreachable).
* ELP's own spec for ELP-USBFHD06H-L21 (manuals.plus and ELP pages returned 403 / no hit).

---

## 7. Downloaded files (`D:\AI_Project\06_deliverables\research_notes\Camera mount options\niryo_files\`)

| File | Bytes | Origin |
|---|---|---|
| ned2_wrist_link.stl | 57,184 | https://raw.githubusercontent.com/NiryoRobotics/ned_ros/master/niryo_robot_description/meshes/ned2/stl/wrist_link.stl |
| ned2_hand_link.stl | 118,084 | same path, hand_link.stl |
| ned2_forearm_link.stl | 251,984 | same path, forearm_link.stl |
| ned2_gripper1_base.STL | 120,184 | .../meshes/gripper_1/base_gripper_1.STL |
| niryo_ned2.urdf.xacro, niryo_ned2_param.urdf.xacro, niryo_ned2_camera.urdf.xacro, niryo_ned2_gripper1_n_camera.urdf.xacro | 10,027 / 1,191 / 2,798 / 5,371 | https://github.com/NiryoRobotics/ned_ros/tree/master/niryo_robot_description/urdf/ned2 |
| niryo_ned_camera.urdf.xacro, niryo_ned3pro_camera.urdf.xacro | 2,553 / 2,802 | same repo, urdf/ned and urdf/ned3pro |
| ned_ros_vision_cam_intrinsics.yaml, ned_ros_vision_ned2_video_server_setup.yaml, ned_ros_vision_node.launch | 221 / 184 / 1,332 | https://github.com/NiryoRobotics/ned_ros/tree/master/niryo_robot_vision |
| NED2_STEP.step | 7,250,770 | https://github.com/NiryoRobotics/ned2 (Download Center "Ned2 3D STEP") |
| ELECTROMAGNET_ned2.step | 266,976 | same repo |
| ned2_print_6-WRIST-V1.STL, ned2_print_7-HAND-V1.STL, ned2_repo_README.md | 167,784 / 214,684 / 142 | same repo, STL/ |
| ned_v1_CAMERA_HOLDER_VISION_NED.STL, ned_v1_CAMERA_COVER_VISION_NED.STL, ned_v1_HAND_COVER.STL, ned_v1_HAND_ELECTROMAGNET.STL, ned_v1_README.md | 182,484 / 82,584 / 794,884 / 157,084 / 319 | https://github.com/NiryoRobotics/ned (STL/25-KV, 10-NED, 24-EM) |
| ned2_technical_specifications_10001874-4.pdf | 5,519,309 | https://niryo.com/wp-content/uploads/2026/06/10001874-4_TECHNICAL_SPECIFICATIONS_NED2.pdf |
| rs_niryo_datasheet_A700000015717952.pdf | 4,982,958 | https://docs.rs-online.com/7c9c/A700000015717952.pdf (Ned2 datasheet 10001874-1, 2024) |
| rs_vision_set_datasheet_74f80c52.pdf | 542,764 | https://assets.rs-online.com/v1678366719/Datasheets/74f80c52e6e10f4bf9381f9dbe4fe1be.pdf |
| ned2_user_manual_v1.0.0_generationrobots.pdf | 8,257,462 | https://static.generation-robots.com/media/manuel-utilisation-ned2-niryo-en.pdf (Niryo Ned2 User Manual v1.0.0, 2022, 38 pp.) |
| docsie_vision_setup_step1_place_screw.jpg, ..._step2_tighten_first_screw.jpg, ..._step3_tighten_second_screw.jpg, ..._step4_cable_through_end_effector.jpg, docsie_vision_setup_camera_settings.png, docsie_vision_kit_overview.jpg | 81,787 / 73,069 / 93,558 / 59,841 / 34,763 / 86,026 | images of https://docs.niryo.com/accessories/vision-set/setup/ and /article/ (cdn.docsie.io) |
| docsie_ned2_mech_overview_parts.jpg, docsie_ned2_mech_3dview_a.jpg, docsie_ned2_mech_3dview_b.png, docsie_ned2_mech_workspace.png, docsie_ned2_mech_installation.jpg, docsie_ned2_mech_tool_mounting_outils2.jpg | 9,459 / 53,346 / 32,054 / 29,669 / 45,260 / 41,341 | images of https://docs.niryo.com/robots/ned2/mechanical-interface/ |
| docsie_ned_v1_camera_vision_set.png, docsie_ned_v1_tool_mounting.png, docsie_ned_v1_dimensions.jpg | 766,270 / 563,446 / 69,902 | images of https://docs.niryo.com/robots/ned/mechanical-interface/ |

Parsing scripts (not in the project): `C:\Users\e7dc\AppData\Local\Temp\claude\D--AI-Project\dd705f28-62f1-4a95-afe4-b6bd1767c0dc\scratchpad\niryo_mesh\` — `stl_bbox.py`, `stl_planes.py`, `stl_features.py`, `step_solids.py`, `step_faces.py`, `pdf_text.py`, `fov_from_intrinsics.py`.
