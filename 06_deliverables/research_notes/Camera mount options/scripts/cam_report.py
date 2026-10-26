"""
Turn cam_options_results.json, cam_calibration.json and sheet_paths.json into markdown tables.

Usage: py -3.13 cam_report.py   (writes cam_options_tables.md next to this script)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

results = json.load(open(os.path.join(HERE, "cam_options_results.json")))
calibration = json.load(open(os.path.join(HERE, "cam_calibration.json")))
sheets = json.load(open(os.path.join(HERE, "sheet_paths.json")))

SHEET_ORDER = ["01_straight_line", "02_arc", "03_two_curve", "04_meander", "05_alpha_loop", "06_alpha_loop_dashed"]
SHEET_SHORT = {"01_straight_line": "01", "02_arc": "02", "03_two_curve": "03", "04_meander": "04", "05_alpha_loop": "05", "06_alpha_loop_dashed": "06"}
OPTION_ORDER = ["A", "B10", "B20", "B30", "C", "D10", "D20", "E1_collar", "E2_arm", "F", "G1_E2_lean10", "G1_E2_lean20", "G2_F_lean20"]
lines = []



def emit(text: str = "") -> None:
    lines.append(text)



def fmt_intervals(intervals) -> str:
    if not intervals:
        return "none"
    return ", ".join(f"{a:+d} to {b:+d}" for a, b in intervals)



def radius_keys(option: dict) -> list:
    return sorted(option["per_radius"].keys(), key = float)



def middle_radius(option: dict) -> str:
    keys = radius_keys(option)
    return keys[len(keys) // 2]



def window_cells(window: dict) -> str:
    return (f"{window['near_from_rod_axis_cm']} / {window['far_from_rod_axis_cm']} | {window['near_from_puck_centre_cm']} / {window['far_from_puck_centre_cm']} | "
            f"{window['near_width_cm']} / {window['far_width_cm']}")



def fraction_cells(option: dict) -> str:
    return " | ".join(f"{option['sheets'][name]['best']['observable_fraction']:.2f}" for name in SHEET_ORDER)



def coverage_summary(option: dict, key: str) -> str:
    """Covered headings at the near, middle and far radius."""

    keys = radius_keys(option)
    picks = [keys[0], keys[len(keys) // 2], keys[-1]]
    parts = []
    for r in picks:
        record = option["per_radius"][r]
        if record.get("infeasible"):
            parts.append(f"r {r}: infeasible")
            continue
        field = record.get(key) if key in record else record.get("union_headings_deg")
        parts.append(f"r {r}: {fmt_intervals(field)}")
    return "; ".join(parts)



def camera_source(fov_key: str) -> str:
    for key, value in results["cameras"].items():
        if int(float(key)) == int(fov_key):
            return value["source"] + f" (horizontal field of view {float(key)} deg)"
    return "unknown"


for fov_key in sorted(results["options"], key = float):
    options = results["options"][fov_key]
    emit(f"## Field of view {fov_key} degrees")
    emit()
    emit(f"Camera model: {camera_source(fov_key)}")
    emit()
    emit("### Window and heading coverage per option (model)")
    emit()
    emit("| Option | Rod tilt (deg) | Tip on line | Window from rod axis near / far (cm) | from puck centre near / far (cm) | width bottom / top (cm) | Headings with 2 cm of line ahead AND puck visible (deg from radial-outward) | Look-ahead 0 / 45 / 90 / 135 / 180 (cm) |")
    emit("|---|---|---|---|---|---|---|---|")
    for key in OPTION_ORDER:
        if key not in options:
            continue
        option = options[key]
        r = middle_radius(option)
        record = option["per_radius"][r]
        if key in ("C", "D10", "D20"):
            inner = record["by_roll"]
            zero = inner.get("0.0") or inner.get("0") or next(iter(inner.values()))
            window = zero.get("window_along_camera_azimuth") or {}
            tilt = zero.get("rod_tilt_deg", "-")
            lookahead = zero.get("lookahead", {})
            coverage = f"union over the roll band: {coverage_summary(option, 'union_headings_deg')}"
            window_text = window_cells(window) if window else "see A (same pose at roll 0) | |"
            tip = "yes (tip exact)"
        else:
            window = record["window"]
            tilt = record["rod_tilt_deg"]
            lookahead = record["lookahead"]
            coverage = coverage_summary(option, "covered_headings_deg")
            window_text = window_cells(window)
            tip = "yes" if record["tip_error_mm"] < 1.0 else f"no ({record['tip_error_mm']} mm)"
        look = " / ".join(str(lookahead[h]["furthest_cm"]) for h in ("0", "45", "90", "135", "180")) if lookahead else "-"
        emit(f"| {option['label']} | {tilt} | {tip} | {window_text} | {coverage} | {look} |")
    emit()
    emit("Look-ahead = furthest visible point of a straight line ahead of the puck, from the puck centre. Window rows for C and D are along the camera's azimuth at roll 0 (= A).")
    emit()
    emit("### Observable fraction of each sheet's length (best orientation and start end; model)")
    emit()
    emit("| Option | " + " | ".join(SHEET_SHORT[n] for n in SHEET_ORDER) + " | mean 03 to 06 |")
    emit("|---|" + "---|" * (len(SHEET_ORDER) + 1))
    for key in OPTION_ORDER:
        if key not in options:
            continue
        option = options[key]
        values = [option["sheets"][n]["best"]["observable_fraction"] for n in SHEET_ORDER]
        emit(f"| {key} | " + " | ".join(f"{v:.2f}" for v in values) + f" | {sum(values[2:]) / 4:.2f} |")
    emit()
    # B contact
    emit("### Option B: pusher contact with the rod leaned towards the base (model)")
    emit()
    emit("| Lean (deg) | Magnet tip shift towards base (cm) | Magnet tip rise (cm) | Clearance change at outward rim (cm) | Contact height on puck, pushing outward / sideways / back (cm) | Overhang at puck top pushing back (cm) | Housing bottom shift towards base (cm) |")
    emit("|---|---|---|---|---|---|---|")
    for lean in (10, 20, 30):
        c = options[f"B{lean}"]["contact"]
        emit(f"| {lean} | {c['magnet_tip_shift_towards_base_cm']} | {c['magnet_tip_rise_cm']} | {c['clearance_change_at_outward_rim_cm']:+} | "
             f"{c['contact_height_cm_heading_0']} / {c['contact_height_cm_heading_90']} / {c['contact_height_cm_heading_180']} | {c['overhang_at_puck_top_cm_heading_180']} | {c['housing_shift_towards_base_at_its_bottom_cm']} |")
    emit()
    # C band
    emit("### Option C: joint-4 roll as camera yaw with the rod within 8 degrees of vertical (model)")
    emit()
    emit("| Tip radius (m) | Max joint-4 roll (deg) | Camera azimuth at max roll (deg) | Rod tilt at max roll (deg) | Headings covered, union over the roll band | Width (deg) | Added over A (deg) |")
    emit("|---|---|---|---|---|---|---|")
    C = options["C"]
    for r in radius_keys(C):
        record = C["per_radius"][r]
        by_roll = record["by_roll"]
        max_roll = record["max_roll_deg"]
        at_max = by_roll.get(str(max_roll)) or by_roll.get(str(float(max_roll))) or {}
        emit(f"| {r} | {max_roll} | {at_max.get('camera_azimuth_deg', '-')} | {at_max.get('rod_tilt_deg', '-')} | {fmt_intervals(record['union_headings_deg'])} | {record['union_width_deg']} | {record['added_over_A_deg']} |")
    emit()
    emit("Rod tilt by roll angle (deg) at each radius, (tilt, tip error mm):")
    emit()
    for r in radius_keys(C):
        band = C["roll_band"][r]["tilt_by_roll"]
        emit(f"- r {r}: " + ", ".join(f"roll {k}: {v[0]}" for k, v in band.items()))
    emit()
    # D
    emit("### Option D: lean plus roll (model)")
    emit()
    emit("| Lean (deg) | Tip radius (m) | Roll (deg): feasible, rod tilt, camera azimuth, covered headings | Union width (deg) |")
    emit("|---|---|---|---|")
    for lean in (10, 20):
        D = options[f"D{lean}"]
        for r in radius_keys(D):
            record = D["per_radius"][r]
            cells = []
            for roll, detail in record["by_roll"].items():
                if detail["feasible"]:
                    cells.append(f"{roll}: ok, tilt {detail['rod_tilt_deg']}, az {detail['camera_azimuth_deg']}, {fmt_intervals(detail['covered_headings_deg'])}")
                else:
                    cells.append(f"{roll}: not feasible (dir err {detail['direction_error_deg']})")
            emit(f"| {lean} | {r} | {'; '.join(cells)} | {record['union_width_deg']} |")
    emit()
    # E
    emit("### Option E: hand mount with joint-6 pan (model)")
    emit()
    for key in ("E1_collar", "E2_arm"):
        E = options[key]
        d = E["design"]
        emit(f"**{key}**: lens {100 * d['forward_m']:.1f} cm ahead of the rod axis, {100 * d['up_m']:.1f} cm above the hand origin "
             f"({d['lens_height_above_table_cm']} cm above the table), tilt {d['tilt_deg']} deg from straight down; {d['note']}.")
        emit()
        emit("| Tip radius (m) | Covered headings (deg) | Width (deg) | Look-ahead 0 / 45 / 90 / 135 / 180 (cm) | Pan used at 0 / 90 / 180 (deg) |")
        emit("|---|---|---|---|---|")
        for r in radius_keys(E):
            record = E["per_radius"][r]
            la = record["lookahead"]
            emit(f"| {r} | {fmt_intervals(record['covered_headings_deg'])} | {record['covered_width_deg']} | "
                 + " / ".join(str(la[h]["furthest_cm"]) for h in ("0", "45", "90", "135", "180"))
                 + f" | {la['0'].get('pan_deg')} / {la['90'].get('pan_deg')} / {la['180'].get('pan_deg')} |")
        emit()
        emit("Arm occlusion check (coverage width with and without the forearm in the model, look-ahead at 180 deg):")
        emit()
        for r, check in E["arm_occlusion_check"].items():
            emit(f"- r {r}: covered {check['covered_with_arm_deg']} deg with arm, {check['covered_without_arm_deg']} without; "
                 f"look-ahead at 180: {check['lookahead_180_with_arm_cm']} cm with arm, {check['lookahead_180_without_arm_cm']} without")
        emit()
        emit("| Sheet | Best rotation / reversed | Observable fraction | Reachable fraction | Pan travel (deg) | Pan time if separate moves (s) | Push time at 0.77 s per cm (s) |")
        emit("|---|---|---|---|---|---|---|")
        for name in SHEET_ORDER:
            b = E["sheets"][name]["best"]
            emit(f"| {SHEET_SHORT[name]} | {b['rotation_deg']} / {b['reversed']} | {b['observable_fraction']:.2f} | {b['reachable_fraction']:.2f} | {b['pan_travel_deg']} | {b['pan_time_if_separate_moves_s']} | {b['push_time_s']} |")
        emit()
    # F
    emit("### Option F: printed wrist mount, no pan (model)")
    emit()
    F = options["F"]
    b = F["best_design"]
    emit(f"Best of the grid: lens {100 * b['forward_m']:.0f} cm ahead of the wrist origin ({100 * b['forward_m'] - 0.925:.1f} cm ahead of the rod axis), "
         f"{100 * b['up_m']:.0f} cm above the wrist axis, tilt {b['tilt_deg']} deg; covered headings {fmt_intervals(b['covered_headings_deg'])} "
         f"({b['covered_width_deg']} deg); mean observable fraction over the six sheets {b['mean_fraction']:.2f}.")
    emit()
    emit("| Forward (cm, from wrist origin) | Up (cm) | Tilt (deg) | Covered width (deg) | Mean fraction | Covered headings |")
    emit("|---|---|---|---|---|---|")
    for t in F["grid_top_10"]:
        emit(f"| {100 * t['forward_m']:.0f} | {100 * t['up_m']:.0f} | {t['tilt_deg']} | {t['covered_width_deg']} | {t['mean_fraction']:.2f} | {fmt_intervals(t['covered_headings_deg'])} |")
    emit()
    emit("Window of the best F design at the middle radius: " + window_cells(F["per_radius"][middle_radius(F)]["window"])
         + f"; lens {F['per_radius'][middle_radius(F)]['window']['lens_height_above_table_cm']} cm above the table.")
    emit()

emit("## Sheets (traced from the photos; model)")
emit()
emit("| Sheet | Length (cm) | Max turn from best fixed direction (deg) | Total turning (deg) | Sharpest turn (deg per cm) | Pixels per cm |")
emit("|---|---|---|---|---|---|")
for name in SHEET_ORDER:
    s = sheets[name]
    emit(f"| {SHEET_SHORT[name]} | {s['length_cm']} | {s['max_turn_from_best_direction_deg']} | {s['total_turning_deg']} | {s['sharpest_turn_deg_per_cm']} | {s['pixels_per_cm']} |")
emit()

with open(os.path.join(HERE, "cam_options_tables.md"), "w", encoding = "utf-8") as handle:
    handle.write("\n".join(lines))
print("written cam_options_tables.md")
