"""
Verifier check 6: the "walk round the puck" and the wrist, in the demonstrations and in runs 7 and 8.

Two ways to find the walk, both mine: (a) from the forearm angle alone (joint 4 in frames.csv and steps.csv, no camera
model): from the forearm's extreme inside the loop to its opposite extreme afterwards; (b) by place: the pictures whose
puck is 14 to 20 cm after the first pass of the crossing on my own placed line (v05 and v04 files). Also: the puck
beside the line per centimetre in the demonstrations against run 8, and the wrist angle (joint 5) along the line.

Run:  D:/AI_Project/.venv/Scripts/python.exe v06_walk_round_and_wrist.py > v06_walk_round_and_wrist.out.txt
Needs v04_run_tracks.csv and v05_demonstration_tracks_05.csv / _06.csv. A few seconds. Nothing connects to the robot.
"""

import numpy
import pandas

from verify_common import VERIFY_FOLDER, VERIFY_PARAMETERS

pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 300)
BACKWARD_BEYOND_DEG = 90.0
WHOLE_LINE_LEAST_PROGRESS_CM = 35.0


def forearm_walk(joint_4: numpy.ndarray, twist: numpy.ndarray, angle: numpy.ndarray, length: numpy.ndarray, turn_sign: float) -> dict:
    """
    Find the walk from the forearm angle alone: the picture where joint 4 is furthest in the direction ``turn_sign``
    (+1: sheet 06 from end A, the forearm first winds up to positive angles; -1: sheet 05 from end B) and the later
    picture where it is furthest the other way.
    """

    signed = turn_sign * joint_4
    first = int(numpy.argmax(signed))
    last = first + int(numpy.argmin(signed[first:]))
    inside = slice(first, last)
    is_backward = (numpy.abs(angle[inside]) > BACKWARD_BEYOND_DEG) & (length[inside] > 0.05)
    return {"first_row": first, "last_row": last, "pictures": last - first, "joint_4_first": joint_4[first], "joint_4_last": joint_4[last],
            "unwound_deg": joint_4[last] - joint_4[first], "twist_label_sum": float(numpy.nansum(twist[inside])),
            "backward": int(is_backward.sum())}


def main() -> None:

    runs = pandas.read_csv(VERIFY_FOLDER / "v04_run_tracks.csv")
    run_logs = {}
    for run_label, key in (("run_8", "run_8_folder"), ("run_7", "run_7_folder")):
        steps = pandas.read_csv(VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS[key] / "steps.csv")
        steps["asked_length_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
        steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
        steps["joint_5_deg"] = numpy.degrees(steps["joint_5"])
        steps = steps.merge(runs[runs["run"] == run_label][["step", "along_cm", "beside_all_cm"]], on = "step")
        run_logs[run_label] = steps

    for sheet, run_label, turn_sign, travel in (("06", "run_8", 1.0, "first_end"), ("05", "run_7", -1.0, "last_end")):
        tracks = pandas.read_csv(VERIFY_FOLDER / f"v05_demonstration_tracks_{sheet}.csv")
        progress = tracks.groupby("plan")["from_start_cm"].agg(lambda values: values.max() - values.min())
        whole = tracks[(tracks["travel_from"] == travel) & tracks["plan"].map(progress).ge(WHOLE_LINE_LEAST_PROGRESS_CM)]
        plans = sorted(whole["plan"].unique())
        steps = run_logs[run_label]
        print(f"\n===== sheet {sheet}: {len(plans)} whole-line demonstrations travelling from '{travel}' ({len(whole)} pictures): {plans[0]} to {plans[-1]} =====")
        print(f"kinds: {sorted(whole['start_kind'].unique())}; loop length on my line {whole['loop_length_cm'].iloc[0]:.1f} cm")

        print("\n(a) The walk from the forearm angle alone (no camera model):")
        rows = []
        for plan, part in whole.groupby("plan"):
            part = part.sort_values("row")
            found = forearm_walk(part["joint_4_deg"].to_numpy(), part["forearm_change_deg"].to_numpy(), part["move_angle_deg"].to_numpy(),
                                 part["move_length_cm"].to_numpy(), turn_sign)
            inside = part.iloc[found["first_row"]:found["last_row"]]
            found.update({"plan": plan, "along_first_cm": part["along_cm"].iloc[found["first_row"]], "along_last_cm": part["along_cm"].iloc[found["last_row"]],
                          "wrist_most_deg": inside["joint_5_deg"].max(), "wrist_median_deg": inside["joint_5_deg"].median(),
                          "beside_at_first_cm": part["beside_cm"].iloc[found["first_row"]]})
            rows.append(found)
        walks = pandas.DataFrame(rows).set_index("plan")
        print(walks.round(1).to_string())
        print("medians:", walks.median(numeric_only = True).round(1).to_dict())
        print("ranges: pictures", int(walks["pictures"].min()), "to", int(walks["pictures"].max()),
              "; unwound", round(walks["unwound_deg"].min(), 0), "to", round(walks["unwound_deg"].max(), 0),
              "; backward", int(walks["backward"].min()), "to", int(walks["backward"].max()))
        found = forearm_walk(steps["joint_4_deg"].to_numpy(), steps["model_twist_deg"].to_numpy(), steps["model_angle_deg"].to_numpy(),
                             steps["asked_length_cm"].to_numpy(), turn_sign)
        print(f"{run_label} the same way: {found['pictures']} pictures from step {found['first_row']} to {found['last_row']}, forearm from"
              f" {found['joint_4_first']:.0f} to {found['joint_4_last']:.0f} deg (unwound {found['unwound_deg']:.0f}), asked twist summed"
              f" {found['twist_label_sum']:.1f} deg, {found['backward']} answers beyond 90 deg; along the line {steps['along_cm'].iloc[found['first_row']]:.1f}"
              f" to {steps['along_cm'].iloc[found['last_row']]:.1f} cm after the crossing")

        print("\n(b) By place: pictures per centimetre along my placed line (0 = first pass of the crossing), per demonstration, and the forearm twist spent there")
        whole = whole.assign(bin = numpy.floor(whole["along_cm"]))
        per_cm = whole.groupby("bin").agg(demonstrations = ("plan", "nunique"), pictures = ("row", "size"), twist = ("forearm_change_deg", "sum"),
                                         joint_4_median = ("joint_4_deg", "median"), wrist_median = ("joint_5_deg", "median"),
                                         beside_10 = ("beside_cm", lambda values: values.quantile(0.1)), beside_median = ("beside_cm", "median"),
                                         beside_90 = ("beside_cm", lambda values: values.quantile(0.9)),
                                         beside_least = ("beside_cm", "min"), beside_most = ("beside_cm", "max"))
        per_cm["pictures_per_demonstration"] = per_cm["pictures"] / len(plans)
        per_cm["twist_per_demonstration"] = per_cm["twist"] / len(plans)
        run_bins = steps.assign(bin = numpy.floor(steps["along_cm"])).groupby("bin").agg(run_pictures = ("step", "size"), run_twist = ("model_twist_deg", "sum"),
                                                                                         run_beside_most = ("beside_all_cm", lambda values: values.abs().max()))
        per_cm = per_cm.join(run_bins)
        print(per_cm[["demonstrations", "pictures_per_demonstration", "twist_per_demonstration", "joint_4_median", "wrist_median", "beside_10", "beside_median",
                      "beside_90", "beside_least", "beside_most", "run_pictures", "run_twist", "run_beside_most"]].round(2).to_string())
        busy = per_cm.index[per_cm["pictures_per_demonstration"] >= 2.5]
        print(f"centimetres with 2.5 or more pictures per demonstration: {[int(value) for value in busy]}")

        zone = (14.0, 20.0) if sheet == "06" else (float(min(busy)), float(max(busy)) + 1.0)
        zone_rows = []
        for plan, part in whole.groupby("plan"):
            inside = part[(part["along_cm"] >= zone[0]) & (part["along_cm"] < zone[1])].sort_values("row")
            is_backward = (inside["move_angle_deg"].abs() > BACKWARD_BEYOND_DEG) & (inside["move_length_cm"] > 0.05)
            first_backward = inside[is_backward].head(1)
            zone_rows.append({"plan": plan, "pictures": len(inside), "twist_sum": inside["forearm_change_deg"].sum(), "backward": int(is_backward.sum()),
                              "joint_4_first": inside["joint_4_deg"].iloc[0], "joint_4_last": inside["joint_4_deg"].iloc[-1], "wrist_most": inside["joint_5_deg"].max(),
                              "beside_at_first_backward": first_backward["beside_cm"].iloc[0] if len(first_backward) else numpy.nan,
                              "joint_4_at_first_backward": first_backward["joint_4_deg"].iloc[0] if len(first_backward) else numpy.nan,
                              "wrist_at_first_backward": first_backward["joint_5_deg"].iloc[0] if len(first_backward) else numpy.nan})
        zone_table = pandas.DataFrame(zone_rows).set_index("plan")
        print(f"\nThe zone {zone[0]:.0f} to {zone[1]:.0f} cm after the crossing, per demonstration:")
        print(zone_table.round(1).to_string())
        print("medians:", zone_table.median().round(1).to_dict())
        print("ranges:", {column: (round(zone_table[column].min(), 1), round(zone_table[column].max(), 1)) for column in zone_table.columns})
        inside = steps[(steps["along_cm"] >= zone[0]) & (steps["along_cm"] < zone[1])]
        is_backward = (inside["model_angle_deg"].abs() > BACKWARD_BEYOND_DEG) & (inside["asked_length_cm"] > 0.05)
        print(f"{run_label} in that zone: {len(inside)} pictures (steps {inside['step'].min()} to {inside['step'].max()}), asked twist summed {inside['model_twist_deg'].sum():.1f} deg,"
              f" {int(is_backward.sum())} answers beyond 90 deg, asked length median {inside['asked_length_cm'].median():.2f} cm and summed {inside['asked_length_cm'].sum():.1f} cm,"
              f" forearm {inside['joint_4_deg'].iloc[0]:.0f} to {inside['joint_4_deg'].iloc[-1]:.0f} deg, puck beside the line {inside['beside_all_cm'].abs().min():.1f} to {inside['beside_all_cm'].abs().max():.1f} cm")
        first_backward = steps[(steps["model_angle_deg"].abs() > BACKWARD_BEYOND_DEG) & (steps["along_cm"] >= zone[0] - 2.0)].head(1)
        if len(first_backward):
            row = first_backward.iloc[0]
            print(f"{run_label}: first answer beyond 90 deg near the zone at step {int(row['step'])}: puck {row['beside_all_cm']:.1f} cm beside the line, forearm {row['joint_4_deg']:.0f} deg, wrist {row['joint_5_deg']:.1f} deg")

        # the run against the demonstrations' spread, step by step
        print(f"\n{run_label} step by step against the demonstrations within 0.75 cm along the line (my placing):")
        table = []
        for row in steps.itertuples():
            near = whole[(whole["along_cm"] - row.along_cm).abs() <= 0.75]
            if len(near) < 5:
                continue
            is_backward = (near["move_angle_deg"].abs() > BACKWARD_BEYOND_DEG) & (near["move_length_cm"] > 0.05)
            table.append({"step": row.step, "along": row.along_cm, "beside": row.beside_all_cm, "d_pictures": len(near), "d_10": near["beside_cm"].quantile(0.1),
                          "d_median": near["beside_cm"].median(), "d_90": near["beside_cm"].quantile(0.9), "d_least": near["beside_cm"].min(), "d_most": near["beside_cm"].max(),
                          "asked_twist": row.model_twist_deg, "d_twist_median": near["forearm_change_deg"].median(), "asked_len": row.asked_length_cm,
                          "d_len_median": near["move_length_cm"].median(), "d_backward_share": is_backward.mean(), "joint_4": row.joint_4_deg,
                          "d_joint_4_median": near["joint_4_deg"].median(), "d_wrist_median": near["joint_5_deg"].median()})
        table = pandas.DataFrame(table)
        print(table.round(2).to_string(index = False))
        outside_spread = (table["beside"] > table["d_90"]) | (table["beside"] < table["d_10"])
        outside_all = (table["beside"] > table["d_most"]) | (table["beside"] < table["d_least"])
        for name, flags in (("outside the 10th to 90th percentile", outside_spread), ("outside every demonstration picture", outside_all)):
            stays = None
            for position in range(len(table) - 1, -1, -1):
                if flags.iloc[position]:
                    stays = int(table["step"].iloc[position])
                else:
                    break
            print(f"  first step from which the run stays {name}: {stays}")

        print(f"\nThe wrist (joint 5) in the demonstrations per 2 cm of line, sheet {sheet} (the driver holds 13.5 deg):")
        whole = whole.assign(bin_2 = 2.0 * numpy.floor(whole["along_cm"] / 2.0))
        wrist = whole.groupby("bin_2")["joint_5_deg"].agg(pictures = "size", median = "median", tenth = lambda values: values.quantile(0.1),
                                                          ninetieth = lambda values: values.quantile(0.9),
                                                          share_over_5_from_held = lambda values: ((values - 13.5).abs() > 5.0).mean())
        print(wrist.round(2).to_string())
        print(f"all pictures of these demonstrations: wrist more than 5 deg from 13.5 in {100 * ((whole['joint_5_deg'] - 13.5).abs() > 5).mean():.0f} %;"
              f" median {whole['joint_5_deg'].median():.1f}, 90th percentile {whole['joint_5_deg'].quantile(0.9):.1f} deg")



if __name__ == "__main__":
    main()
