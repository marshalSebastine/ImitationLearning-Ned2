"""
Step 4 of the dashed-line analysis: how the models that never saw a sheet score on it, inside the loop against on
the straight parts, sheet 06 (dashed) against sheet 05 (solid), and whether a gap ahead of the puck goes with errors.

The fold models (trained without sheet 05, or without sheet 06) are run on the cached training pictures of their
held-out sheet; what is scored is the exact step, the output the robot driver carries out (the saved offline scores
of the chunk recipe use the chunk's first step instead; both are shown). Every picture's place on the line comes
from tracks.csv (step 2). Measures per part of the line:
    path within 45     the asked direction is within 45 degrees of the direction to the demonstrated tip place three
                       steps later (the report's main measure), on pictures where that place is 1.5 cm or more away;
    twist              on pictures where the demonstration twists the forearm by 2 degrees or more: right sign, and
                       the mean size asked against demonstrated;
    length             the asked step length against the demonstrated one, and how often under 0.3 cm is asked;
    backward           where the demonstration moves the tip backwards in the picture (the walk round the puck): how
                       often the model also asks for a backward step.

Writes 04_fold_models_on_loops.png, 04_fold_rows.csv and prints the tables. No training; about two minutes.

Run: D:/AI_Project/.venv/Scripts/python.exe 04_fold_models_on_loops.py > 04_fold_models_on_loops.out.txt
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER
from model_answers import angle_error_deg, answers, loaded_policy, saved_predictions
from training_targets import DIRECTION_NAMES, chunk_targets, exact_step_targets, load_training_frames

FOLD_PARAMETERS = {
    "models": {"05": ["E recipe without 05", "E recipe without 05, seed 2", "A recipe without 05"],
               "06": ["E recipe without 06", "E recipe without 06, seed 2", "A recipe without 06"]},
    "path_steps": 3,
    "path_least_cm": 1.5,
    "twist_least_deg": 2.0,
    "short_step_cm": 0.3,
    "backward_cm": -0.3,
    "tight_radius_cm": 3.0,
    "placed_ink_share": 0.75,
    # the puck's centre is 1.2 cm behind its front edge; "a gap ahead" means the first centimetre of line beyond the
    # front edge is mostly gap
    "puck_radius_mm": 12,
    "ahead_mm": 10
}



def labelled_rows() -> pandas.DataFrame:
    """Return the cleaned training rows of sheets 05 and 06 with their targets and their place on the line."""

    settings = FOLD_PARAMETERS
    frames = load_training_frames({"is_audit_cleaning": True})
    frames = frames[frames["sheet"].isin(["path_05", "path_06"])].reset_index(drop = True)
    exact = exact_step_targets(frames)
    chunk = chunk_targets(frames, 5)
    path = chunk[:, settings["path_steps"] - 1, :]
    rows = pandas.DataFrame({
        "frame_index": frames["frame_index"].to_numpy(), "episode_id": frames["episode_id"].to_numpy(), "sheet": frames["sheet"].str[-2:].to_numpy(),
        "action": frames["action"].to_numpy(), "demo_forward_cm": exact[:, 0], "demo_left_cm": exact[:, 1], "demo_twist_deg": exact[:, 2],
        "demo_angle_deg": numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0])), "demo_length_cm": numpy.hypot(exact[:, 0], exact[:, 1]),
        "path_angle_deg": numpy.degrees(numpy.arctan2(path[:, 1], path[:, 0])),
        "is_path_row": (frames["action"].to_numpy() != "stop") & (numpy.linalg.norm(path, axis = 1) >= settings["path_least_cm"]),
        "is_moving": frames["action"].isin(DIRECTION_NAMES).to_numpy(), "is_stop": (frames["action"] == "stop").to_numpy()})
    tracks = pandas.read_csv(EVIDENCE_FOLDER / "tracks.csv")
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv")
    placed = set(registrations.loc[registrations["ink_share_on_seen_ink_points"] >= settings["placed_ink_share"], "recording"])
    tracks = tracks[tracks["frame_index"] >= 0]
    rows = rows.merge(tracks[["frame_index", "recording", "start_kind", "start_end", "puck_along_cm", "puck_beside_cm", "is_in_loop", "loop_length_cm",
                              "line_radius_cm", "ink_ahead_picture", "joint_4_deg", "joint_5_deg"]], on = "frame_index", how = "left")
    is_placed = rows["recording"].isin(placed) & rows["puck_along_cm"].notna()
    rows["part"] = numpy.where(~is_placed, "not placed", numpy.where(rows["puck_along_cm"] < 0, "before the crossing",
                               numpy.where(rows["puck_along_cm"] <= rows["loop_length_cm"], "loop", "after the loop")))
    rows["is_tight"] = is_placed & (rows["line_radius_cm"] < settings["tight_radius_cm"])

    # sheet 06: is the line just beyond the puck's front edge ink or gap (from the reference line's dashes)
    reference = pandas.read_csv(EVIDENCE_FOLDER / "reference_line_06.csv")
    is_ink = reference["is_ink"].to_numpy() > 0
    in_loop = numpy.flatnonzero(reference["is_in_loop"].to_numpy() > 0)
    rows["ink_first_cm_ahead"] = numpy.nan
    rows["is_puck_over_ink"] = numpy.nan
    for position in numpy.flatnonzero((rows["sheet"] == "06").to_numpy() & is_placed.to_numpy()):
        along_mm = int(round(10.0 * rows["puck_along_cm"].iloc[position]))
        if rows["start_end"].iloc[position] == "A":
            flags, here = is_ink, int(in_loop[0]) + along_mm
        else:
            flags, here = is_ink[::-1], len(is_ink) - 1 - int(in_loop[-1]) + along_mm
        first = here + settings["puck_radius_mm"]
        if 0 <= here < len(flags):
            rows.iat[position, rows.columns.get_loc("is_puck_over_ink")] = float(flags[here])
        if 0 <= first and first + settings["ahead_mm"] <= len(flags):
            rows.iat[position, rows.columns.get_loc("ink_first_cm_ahead")] = float(flags[first:first + settings["ahead_mm"]].mean())
    return rows



def measures(
        rows: pandas.DataFrame,
        prefix: str
) -> dict:
    """Return the measures of one model's answers (columns starting with ``prefix``) on a set of rows."""

    settings = FOLD_PARAMETERS
    result = {"pictures": len(rows)}
    path_rows = rows[rows["is_path_row"]]
    if len(path_rows):
        error = angle_error_deg(path_rows[f"{prefix}angle_deg"], path_rows["path_angle_deg"])
        result.update({"path_pictures": len(path_rows), "path_within_45": float((error <= 45.0).mean()), "path_error_median_deg": float(numpy.median(error))})
        if f"{prefix}chunk_first_deg" in rows:
            result["chunk_first_step_within_45"] = float((angle_error_deg(path_rows[f"{prefix}chunk_first_deg"], path_rows["path_angle_deg"]) <= 45.0).mean())
            result["chunk_third_step_within_45"] = float((angle_error_deg(path_rows[f"{prefix}chunk_third_deg"], path_rows["path_angle_deg"]) <= 45.0).mean())
    twisting = rows[(rows["demo_twist_deg"].abs() >= settings["twist_least_deg"]) & ~rows["is_stop"]]
    if len(twisting):
        result.update({"twist_pictures": len(twisting),
                       "twist_sign_right": float((numpy.sign(twisting[f"{prefix}twist_deg"]) == numpy.sign(twisting["demo_twist_deg"])).mean()),
                       "twist_asked_mean_deg": float(twisting[f"{prefix}twist_deg"].abs().mean()),
                       "twist_demonstrated_mean_deg": float(twisting["demo_twist_deg"].abs().mean())})
    moving = rows[~rows["is_stop"]]
    if len(moving):
        result.update({"length_asked_median_cm": float(moving[f"{prefix}length_cm"].median()), "length_demonstrated_median_cm": float(moving["demo_length_cm"].median()),
                       "asked_under_0.3_cm": float((moving[f"{prefix}length_cm"] < settings["short_step_cm"]).mean()),
                       "demonstrated_under_0.3_cm": float((moving["demo_length_cm"] < settings["short_step_cm"]).mean()),
                       "stop_chance_mean": float(moving[f"{prefix}stop_probability"].mean())})
    backward = rows[(rows["demo_forward_cm"] <= settings["backward_cm"]) & ~rows["is_stop"]]
    if len(backward):
        result.update({"backward_pictures": len(backward), "backward_also_asked": float((backward[f"{prefix}forward_cm"] < 0.0).mean())})
    return result



def main() -> None:

    settings = FOLD_PARAMETERS
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    rows = labelled_rows()
    pictures = numpy.load(ANALYSIS_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    print("cleaned training pictures:", rows.groupby("sheet").size().to_dict(), "| by part of the line:")
    print(rows.groupby(["sheet", "part"]).size().unstack(fill_value = 0).to_string())

    model_columns = {}
    for sheet, model_names in settings["models"].items():
        is_sheet = (rows["sheet"] == sheet).to_numpy()
        batch = numpy.stack([pictures[index] for index in rows.loc[is_sheet, "frame_index"]])
        for model_name in model_names:
            policy = loaded_policy(model_name)
            answer = answers(policy, batch)
            prefix = f"{model_name}|"
            for name in ("forward_cm", "left_cm", "twist_deg", "angle_deg", "length_cm", "stop_probability"):
                rows.loc[is_sheet, prefix + name] = answer[name]
            if "chunk_cm" in answer:
                rows.loc[is_sheet, prefix + "chunk_first_deg"] = numpy.degrees(numpy.arctan2(answer["chunk_cm"][:, 0, 1], answer["chunk_cm"][:, 0, 0]))
                rows.loc[is_sheet, prefix + "chunk_third_deg"] = numpy.degrees(numpy.arctan2(answer["chunk_cm"][:, 2, 1], answer["chunk_cm"][:, 2, 0]))
            saved = saved_predictions(model_name)
            if saved is not None:
                ours = rows.loc[is_sheet].set_index("frame_index")[prefix + ("chunk_first_deg" if "chunk_cm" in answer else "angle_deg")]
                difference = angle_error_deg(ours.reindex(saved["frame_index"]).to_numpy(), saved["angle_deg"])
                print(f"check, {model_name}: this script's direction against the direction saved when the fold was scored, "
                      f"largest difference {numpy.nanmax(difference):.4f} deg on {len(saved['frame_index'])} pictures")
            model_columns.setdefault(sheet, []).append(model_name)

    table_rows = []
    for sheet, model_names in model_columns.items():
        sheet_rows = rows[rows["sheet"] == sheet]
        subsets = [("whole sheet", sheet_rows)] + [(part, sheet_rows[sheet_rows["part"] == part]) for part in ("before the crossing", "loop", "after the loop")]
        subsets.append(("loop, bend tighter than 3 cm radius", sheet_rows[(sheet_rows["part"] == "loop") & sheet_rows["is_tight"]]))
        subsets.append(("loop, bend wider than 3 cm radius", sheet_rows[(sheet_rows["part"] == "loop") & ~sheet_rows["is_tight"]]))
        for model_name in model_names:
            for subset_name, subset in subsets:
                table_rows.append({"sheet": sheet, "model": model_name, "part": subset_name, **measures(subset, f"{model_name}|")})
    table = pandas.DataFrame(table_rows)
    print("\nDirection (path within 45 degrees) by part of the line; the exact step is what the robot carries out")
    print(table[["sheet", "model", "part", "pictures", "path_pictures", "path_within_45", "path_error_median_deg", "chunk_first_step_within_45",
                 "chunk_third_step_within_45"]].round(3).to_string(index = False))
    print("\nTwist, step length, backward steps and stop chance by part of the line")
    print(table[["sheet", "model", "part", "twist_pictures", "twist_sign_right", "twist_asked_mean_deg", "twist_demonstrated_mean_deg", "length_asked_median_cm",
                 "length_demonstrated_median_cm", "asked_under_0.3_cm", "demonstrated_under_0.3_cm", "backward_pictures", "backward_also_asked",
                 "stop_chance_mean"]].round(3).to_string(index = False))

    # the gaps of sheet 06
    print("\nSheet 06, models that never saw a dashed line: does a gap just ahead of the puck go with direction errors?")
    dashed = rows[(rows["sheet"] == "06") & rows["is_path_row"] & rows["ink_first_cm_ahead"].notna()].copy()
    dashed["ahead"] = numpy.where(dashed["ink_first_cm_ahead"] <= 0.2, "gap ahead", numpy.where(dashed["ink_first_cm_ahead"] >= 0.8, "dash ahead", "edge of a dash ahead"))
    dashed["under"] = numpy.where(dashed["is_puck_over_ink"] > 0, "puck over a dash", "puck over a gap")
    gap_rows = []
    for model_name in model_columns["06"]:
        error = angle_error_deg(dashed[f"{model_name}|angle_deg"], dashed["path_angle_deg"])
        dashed[f"{model_name}|error"] = error
        for part in ("all placed", "before the crossing", "loop", "after the loop"):
            subset = dashed if part == "all placed" else dashed[dashed["part"] == part]
            for column in ("ahead", "under"):
                for value, group in subset.groupby(column):
                    group_error = group[f"{model_name}|error"]
                    gap_rows.append({"model": model_name, "part": part, "what": value, "path_pictures": len(group), "path_within_45": float((group_error <= 45).mean()),
                                     "error_median_deg": float(group_error.median())})
            seen = subset[subset["ink_ahead_picture"].notna()]
            if len(seen) > 30:
                low = seen[seen["ink_ahead_picture"] <= seen["ink_ahead_picture"].quantile(0.33)]
                high = seen[seen["ink_ahead_picture"] >= seen["ink_ahead_picture"].quantile(0.67)]
                for value, group in (("least ink seen ahead (lowest third, picture)", low), ("most ink seen ahead (highest third, picture)", high)):
                    group_error = group[f"{model_name}|error"]
                    gap_rows.append({"model": model_name, "part": part, "what": value, "path_pictures": len(group), "path_within_45": float((group_error <= 45).mean()),
                                     "error_median_deg": float(group_error.median())})
    gap_table = pandas.DataFrame(gap_rows)
    print(gap_table.round(3).to_string(index = False))

    rows.drop(columns = [name for name in rows.columns if name.endswith(("lean_deg", "hand_deg"))]).round(4).to_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", index = False)

    figure, axes = pyplot.subplots(1, 3, figsize = (22, 6.5))
    parts = ["before the crossing", "loop", "after the loop"]
    width = 0.13
    axis = axes[0]
    position = 0
    for sheet, colour in (("05", "tab:blue"), ("06", "tab:red")):
        for number, model_name in enumerate(model_columns[sheet]):
            values = [table[(table["model"] == model_name) & (table["part"] == part)]["path_within_45"].iloc[0] for part in parts]
            axis.bar(numpy.arange(len(parts)) + (position - 2.5) * width, values, width = width, color = colour, alpha = 1.0 - 0.25 * number, label = f"sheet {sheet}: {model_name}")
            position += 1
    axis.set_xticks(numpy.arange(len(parts)))
    axis.set_xticklabels(parts)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share of pictures with the asked direction within 45 degrees of the path")
    axis.set_title("Unseen sheet: direction by part of the line (exact step)")
    axis.grid(True, axis = "y", linewidth = 0.3)
    axis.legend(fontsize = 7, loc = "lower left")

    axis = axes[1]
    position = 0
    for sheet, colour in (("05", "tab:blue"), ("06", "tab:red")):
        model_name = model_columns[sheet][0]
        asked = [table[(table["model"] == model_name) & (table["part"] == part)]["twist_asked_mean_deg"].iloc[0] for part in parts]
        demonstrated = [table[(table["model"] == model_name) & (table["part"] == part)]["twist_demonstrated_mean_deg"].iloc[0] for part in parts]
        axis.bar(numpy.arange(len(parts)) + (position - 1.5) * 0.2, demonstrated, width = 0.2, color = colour, alpha = 0.35, label = f"sheet {sheet}: demonstrated")
        axis.bar(numpy.arange(len(parts)) + (position - 0.5) * 0.2, asked, width = 0.2, color = colour, label = f"sheet {sheet}: asked by {model_name}")
        position += 2
    axis.set_xticks(numpy.arange(len(parts)))
    axis.set_xticklabels(parts)
    axis.set_ylabel("mean forearm twist per picture (degrees), where the demonstration twists 2 or more")
    axis.set_title("Unseen sheet: twist asked against twist demonstrated")
    axis.grid(True, axis = "y", linewidth = 0.3)
    axis.legend(fontsize = 7)

    axis = axes[2]
    model_name = model_columns["06"][0]
    shown = gap_table[(gap_table["model"] == model_name) & gap_table["what"].isin(["gap ahead", "edge of a dash ahead", "dash ahead"])]
    for number, (what, colour) in enumerate((("gap ahead", "tab:orange"), ("edge of a dash ahead", "0.6"), ("dash ahead", "0.2"))):
        values = [shown[(shown["part"] == part) & (shown["what"] == what)]["path_within_45"].iloc[0] for part in parts]
        counts = [int(shown[(shown["part"] == part) & (shown["what"] == what)]["path_pictures"].iloc[0]) for part in parts]
        bars = axis.bar(numpy.arange(len(parts)) + (number - 1) * 0.25, values, width = 0.25, color = colour, label = what)
        for bar, count in zip(bars, counts):
            axis.annotate(f"n={count}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha = "center", va = "bottom", fontsize = 7)
    axis.set_xticks(numpy.arange(len(parts)))
    axis.set_xticklabels(parts)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share within 45 degrees of the path")
    axis.set_title(f"Sheet 06, {model_name}: is the first centimetre beyond the puck a gap or a dash?", fontsize = 10)
    axis.grid(True, axis = "y", linewidth = 0.3)
    axis.legend(fontsize = 8, loc = "lower left")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "04_fold_models_on_loops.png", dpi = 70)



if __name__ == "__main__":
    main()
