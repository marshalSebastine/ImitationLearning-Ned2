"""
Is "the camera looks along the line" (the plan's rule 3) the easier style to learn? A test on the recordings we have.

The 209 recordings hold both kinds of picture: rows where the puck travels up the picture (the camera looks along the
line) and rows where it does not. If the models answer better on the first kind ON SHEETS THEY NEVER SAW, the rule has
the project's own numbers behind it; and the same split gives a base line for a pilot of the new style.

Measured on the fold models' answers on held-out sheets (the outputs the step-length analyst saved and the verifier
checked) and the twist analyst's puck positions (as in m01); nothing is trained, nothing connects to the robot.
    1. By how far the puck's travel lies off the picture's top: the direction within 45 degrees of the shown step, the
       twist asked against the twist shown (correlation, right sense, size), and the asked size of the step.
    2. The same for the net turn over windows of 10 rows.

Run from this folder:
    python m06_is_the_style_learnable.py > m06_is_the_style_learnable.out.txt
"""

from typing import Dict, Optional

import numpy
import pandas

import m01_style_against_the_rules
import method_common

LEARNABLE_PARAMETERS = {
    "bands_deg": [(0.0, 20.0), (20.0, 45.0), (45.0, 90.0), (90.0, 180.1)],
    "window_rows": 10,
    "net_turn_least_deg": 30.0,
    "aligned_most_deg": 30.0
}



def learnable_report(parameters: Optional[Dict] = None) -> None:
    """Print the two measurements named in the module's text."""

    settings = method_common.merged_parameters(m01_style_against_the_rules.STYLE_PARAMETERS)
    settings.update(LEARNABLE_PARAMETERS)
    settings.update(parameters or {})
    frames = m01_style_against_the_rules.puck_travel_in_picture(method_common.cleaned_frames(settings), settings)
    for recipe, path in settings["fold_outputs"].items():
        data = numpy.load(path)
        answers = pandas.DataFrame({"frame_index": data["frame_index"], "asked_forward": data["exact"][:, 0], "asked_left": data["exact"][:, 1],
                                    "asked_twist": data["exact"][:, 2]})
        table = frames.merge(answers, on = "frame_index", how = "left")
        table = table[table["action"] != "stop"].copy()
        table["shown_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
        table["shown_twist"] = table["forearm_change_deg"].fillna(0.0)
        table["asked_cm"] = numpy.hypot(table["asked_forward"], table["asked_left"])
        shown_angle = numpy.degrees(numpy.arctan2(table["tip_left_m"].fillna(0.0), table["tip_forward_m"].fillna(0.0)))
        asked_angle = numpy.degrees(numpy.arctan2(table["asked_left"], table["asked_forward"]))
        table["angle_error"] = numpy.abs(method_common.wrapped_degrees(asked_angle - shown_angle))
        table["off_deg"] = table["travel_picture_deg"].abs()
        known = table[table["off_deg"].notna()]
        print(f"================ recipe {recipe}: {len(known)} step rows with a known direction of puck travel, each answered by the model that never saw the sheet")
        print("1. By how far the puck's travel lies off the picture's top")
        print("   off the top      rows   direction within 45 (shown move 0.3 cm or more)   twist: correlation   right sense (shown 2 deg or more)   asked / shown mean size   asked length median   shown")
        for low, high in settings["bands_deg"]:
            band = known[(known["off_deg"] >= low) & (known["off_deg"] < high)]
            moving = band[band["shown_cm"] >= 0.3]
            twisting = band[band["shown_twist"].abs() >= 2.0]
            print(f"   {low:3.0f} to {min(high, 180.0):3.0f}   {len(band):7d}   {100 * (moving['angle_error'] <= 45).mean():46.1f}%   {numpy.corrcoef(band['asked_twist'], band['shown_twist'])[0, 1]:18.2f}"
                  f"   {100 * (numpy.sign(twisting['asked_twist']) == numpy.sign(twisting['shown_twist'])).mean():32.1f}%"
                  f"   {band['asked_twist'].abs().mean() / band['shown_twist'].abs().mean():23.2f}   {band['asked_cm'].median():19.2f}   {band['shown_cm'].median():5.2f}")
        print("   the same split inside each held-out sheet (direction within 45 / twist correlation), travel within"
              f" {settings['aligned_most_deg']:.0f} degrees of the picture's top against further off:")
        for sheet, group in known.groupby("sheet"):
            cells = []
            for name, part in (("along", group[group["off_deg"] <= settings["aligned_most_deg"]]), ("off", group[group["off_deg"] > settings["aligned_most_deg"]])):
                moving = part[part["shown_cm"] >= 0.3]
                cells.append(f"{name} {len(part):5d} rows: {100 * (moving['angle_error'] <= 45).mean():5.1f} % / {numpy.corrcoef(part['asked_twist'], part['shown_twist'])[0, 1]:+.2f}")
            print(f"      {sheet[-2:]}: " + "; ".join(cells))

        print(f"2. Net turn over windows of {settings['window_rows']} rows with a shown net turn of {settings['net_turn_least_deg']:.0f} degrees or more,"
              f" by the window's median distance of the puck's travel from the picture's top")
        rows = []
        for episode_id, group in table.groupby("episode_id", sort = False):
            shown, asked, off = group["shown_twist"].to_numpy(), group["asked_twist"].to_numpy(), group["off_deg"].to_numpy()
            for start in range(0, len(group) - settings["window_rows"] + 1, settings["window_rows"]):
                part = slice(start, start + settings["window_rows"])
                if abs(shown[part].sum()) >= settings["net_turn_least_deg"] and numpy.isfinite(off[part]).sum() >= settings["window_rows"] // 2:
                    rows.append({"off": numpy.nanmedian(off[part]), "shown": abs(shown[part].sum()), "given": numpy.sign(shown[part].sum()) * asked[part].sum()})
        windows = pandas.DataFrame(rows)
        for low, high in settings["bands_deg"]:
            band = windows[(windows["off"] >= low) & (windows["off"] < high)]
            print(f"   {low:3.0f} to {min(high, 180.0):3.0f}: {len(band):4d} windows; the model gives back {band['given'].sum() / band['shown'].sum():.2f} of the shown net turn,"
                  f" right sense in {100 * (band['given'] > 0).mean():.0f} %")
        print()



if __name__ == "__main__":
    learnable_report()
