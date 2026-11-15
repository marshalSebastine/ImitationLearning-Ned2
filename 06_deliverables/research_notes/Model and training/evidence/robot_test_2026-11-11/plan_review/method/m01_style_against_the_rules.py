"""
How the 209 recorded demonstrations stand against the demonstration rules the plan proposes in its step 3 (11.11.26).

Measured from the cleaned training rows and the twist analyst's puck positions; nothing is trained, nothing connects to
the robot. Four questions:
    1. Rule 3 ("stop pushing, turn the forearm until the line ahead runs up the picture, push on"): in how many of the
       recorded pushing steps did the puck travel in a direction that does NOT run up the picture?
    2. The same rule against the forearm's limit: how often would "the line runs up the picture" need joint 4 beyond
       the 117 degrees the driver allows?
    3. How large is one recorded step by the recording rule (1 cm of tip travel or 5 degrees on a joint), counting all
       six joints and counting only what the model predicts (tip travel and forearm twist)?
    4. How much of the data each sheet holds, and what 48 new episodes on 12 new shapes add (the "effective number of
       shapes" of a training set when every picture counts the same).

Run from this folder:
    python m01_style_against_the_rules.py > m01_style_against_the_rules.out.txt
"""

from typing import Dict, Optional

import numpy
import pandas

import method_common

STYLE_PARAMETERS = {
    # the puck's direction of travel is taken between this many rows back and ahead, and trusted only when the puck
    # moved at least this far on the table between them
    "travel_rows": 4,
    "travel_least_cm": 2.0,
    # a pushing step: the tip moves at least this far
    "push_least_cm": 0.7,
    # the plan's recording round 2: new shapes and episodes per shape
    "new_shapes": 12,
    "episodes_per_new_shape": 4
}



def puck_travel_in_picture(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Add the puck's direction of travel on the table and in each picture's frame.

    Parameters
    ----------
    frames : pandas.DataFrame
        The cleaned rows, sorted by episode and step.
    parameters : dict
        "puck_positions", "travel_rows", "travel_least_cm".

    Returns
    -------
    pandas.DataFrame
        ``frames`` with "travel_table_deg" and "travel_picture_deg" (the puck's travel between "travel_rows" rows back
        and ahead: where the puck is going now; 0 = up the picture, positive = to its left; NaN where the puck was not
        found or hardly moved), and "ahead_table_deg" and "ahead_picture_deg" (the puck's travel from this row to twice
        "travel_rows" rows ahead: where the line ahead of the puck runs).
    """

    puck = pandas.read_csv(parameters["puck_positions"]).set_index("frame_index")
    frames = frames.copy()
    frames["puck_x_cm"] = frames["frame_index"].map(puck["puck_x_cm"])
    frames["puck_y_cm"] = frames["frame_index"].map(puck["puck_y_cm"])
    is_rough = frames["frame_index"].map(puck["is_position_rough"]).fillna(1.0) > 0
    frames.loc[is_rough, ["puck_x_cm", "puck_y_cm"]] = numpy.nan
    grouped = frames.groupby("episode_id", sort = False)
    rows = parameters["travel_rows"]
    for name, back, ahead in (("travel", rows, rows), ("ahead", 0, 2 * rows)):
        moved_x = grouped["puck_x_cm"].shift(-ahead) - grouped["puck_x_cm"].shift(back)
        moved_y = grouped["puck_y_cm"].shift(-ahead) - grouped["puck_y_cm"].shift(back)
        moved_cm = numpy.hypot(moved_x, moved_y)
        table_deg = numpy.degrees(numpy.arctan2(moved_y, moved_x))
        table_deg[moved_cm < parameters["travel_least_cm"]] = numpy.nan
        frames[f"{name}_table_deg"] = table_deg
        frames[f"{name}_picture_deg"] = method_common.wrapped_degrees(table_deg - frames["picture_heading_deg"])
    return frames



def style_report(parameters: Optional[Dict] = None) -> None:
    """Print the four measurements named in the module's text."""

    settings = method_common.merged_parameters(STYLE_PARAMETERS)
    settings.update(parameters or {})
    frames = puck_travel_in_picture(method_common.cleaned_frames(settings), settings)
    is_step = frames["action"] != "stop"
    length_cm = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    twist_deg = frames["forearm_change_deg"].fillna(0.0)
    joint_4_deg = numpy.degrees(frames["joint_4"])
    frames["length_cm"] = length_cm
    frames["joint_4_deg"] = joint_4_deg

    print("1. Rule 3 against the recorded style: where did the puck travel in the picture while the tip pushed?")
    print("   (pushing step = tip move of 0.7 cm or more; the puck's travel between 4 rows back and 4 ahead, at least 2 cm)")
    is_known = frames["travel_picture_deg"].notna() & is_step
    is_push = is_known & (length_cm >= settings["push_least_cm"])
    print(f"   rows with a known direction of puck travel: {int(is_known.sum())} of {int(is_step.sum())} step rows; pushing rows among them: {int(is_push.sum())}")
    print("   sheet     pushing rows   median |angle|   within 20   within 30   within 45   beyond 60   beyond 90")
    for sheet, group in list(frames[is_push].groupby("sheet")) + [("all", frames[is_push])]:
        angle = group["travel_picture_deg"].abs()
        print(f"   {sheet:8s}  {len(group):11d}   {angle.median():13.1f}   {100 * (angle <= 20).mean():8.1f}%   {100 * (angle <= 30).mean():8.1f}%"
              f"   {100 * (angle <= 45).mean():8.1f}%   {100 * (angle > 60).mean():8.1f}%   {100 * (angle > 90).mean():8.1f}%")
    is_ahead_known = frames["ahead_picture_deg"].notna() & is_step
    is_ahead_push = is_ahead_known & (length_cm >= settings["push_least_cm"])
    print("   the same for the line AHEAD of the puck (the puck's travel from this row to 8 rows ahead), which is what rule 3 names:")
    print("   sheet     pushing rows   median |angle|   within 20   within 30   within 45   beyond 60   beyond 90")
    for sheet, group in list(frames[is_ahead_push].groupby("sheet")) + [("all", frames[is_ahead_push])]:
        angle = group["ahead_picture_deg"].abs()
        print(f"   {sheet:8s}  {len(group):11d}   {angle.median():13.1f}   {100 * (angle <= 20).mean():8.1f}%   {100 * (angle <= 30).mean():8.1f}%"
              f"   {100 * (angle <= 45).mean():8.1f}%   {100 * (angle > 60).mean():8.1f}%   {100 * (angle > 90).mean():8.1f}%")
    off_line = frames[is_ahead_known & (frames["ahead_picture_deg"].abs() > 30)]
    print(f"   rows where the line ahead runs more than 30 degrees off the picture's top: {len(off_line)} of {int(is_ahead_known.sum())};"
          f" in these the demonstrator pushed on (0.7 cm or more) in {100 * (off_line['length_cm'] >= settings['push_least_cm']).mean():.1f} %,"
          f" moved 0.3 to 0.7 cm in {100 * ((off_line['length_cm'] >= 0.3) & (off_line['length_cm'] < 0.7)).mean():.1f} % and turned nearly in place"
          f" (under 0.3 cm) in {100 * (off_line['length_cm'] < 0.3).mean():.1f} %; median |twist| {off_line['forearm_change_deg'].abs().median():.1f} degrees")
    twisting = frames[is_known & (twist_deg.abs() >= 3.0)]
    correlation = numpy.corrcoef(twisting["travel_picture_deg"].clip(-90, 90), twisting["forearm_change_deg"])[0, 1]
    print(f"   twisting rows (3 degrees or more) with a known travel direction: {len(twisting)}; correlation of the twist with the puck's travel"
          f" direction in the picture (clipped at 90 degrees): {correlation:+.2f}")
    print("   (a twist of +1 degree turns the picture's heading by about -0.9 degrees, so turning the camera towards the puck's travel gives a NEGATIVE"
          " correlation; -1 would be a demonstrator who always turns the camera towards where the puck goes)")
    for low, high in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90)):
        for side, sign in (("left", 1.0), ("right", -1.0)):
            band = frames[is_known & (sign * frames["travel_picture_deg"] > low) & (sign * frames["travel_picture_deg"] <= high)]
            towards = (-sign * band["forearm_change_deg"] >= 2.0).mean()
            away = (sign * band["forearm_change_deg"] >= 2.0).mean()
            print(f"   puck travels {low:2d} to {high:2d} degrees to the picture's {side:5s}: {len(band):5d} rows; camera turned towards it (2 degrees or more) in"
                  f" {100 * towards:4.1f} %, away in {100 * away:4.1f} %, neither in {100 * (1 - towards - away):4.1f} %; median tip step"
                  f" {band['length_cm'].median():.2f} cm")

    print()
    print("2. Rule 3 against the forearm's limit: the joint-4 angle that would put the puck's travel straight up the picture")
    frames["required_joint_4_deg"] = joint_4_deg - frames["travel_picture_deg"] / settings["heading_per_twist"]
    frames["required_ahead_joint_4_deg"] = joint_4_deg - frames["ahead_picture_deg"] / settings["heading_per_twist"]
    aligned = frames[is_known & (frames["travel_picture_deg"].abs() <= 10)]
    print(f"   check of the formula: in the {len(aligned)} rows where the puck already travels within 10 degrees of the picture's top the needed angle"
          f" differs from the recorded one by a median {numpy.median(numpy.abs(aligned['required_joint_4_deg'] - aligned['joint_4_deg'])):.1f} degrees")
    limit = settings["joint_4_most_deg"]
    print("   sheet     rows   needs |joint 4| over 117   over 105   over 90   recorded |joint 4| over 90   over 105   episodes that need over 117 somewhere")
    for sheet, group in list(frames[is_known].groupby("sheet")) + [("all", frames[is_known])]:
        needed = group["required_joint_4_deg"].abs()
        recorded = group["joint_4_deg"].abs()
        episodes_over = group[needed > limit]["episode_id"].nunique()
        print(f"   {sheet:8s} {len(group):5d}   {100 * (needed > limit).mean():22.1f}%   {100 * (needed > 105).mean():6.1f}%   {100 * (needed > 90).mean():5.1f}%"
              f"   {100 * (recorded > 90).mean():23.1f}%   {100 * (recorded > 105).mean():6.1f}%   {episodes_over:4d} of {group['episode_id'].nunique()}")
    print("   the same with the line AHEAD of the puck (from this row to 8 rows ahead):")
    for sheet, group in list(frames[is_ahead_known].groupby("sheet")) + [("all", frames[is_ahead_known])]:
        needed = group["required_ahead_joint_4_deg"].abs()
        episodes_over = group[needed > limit]["episode_id"].nunique()
        print(f"   {sheet:8s} {len(group):5d}   {100 * (needed > limit).mean():22.1f}%   {100 * (needed > 105).mean():6.1f}%   {100 * (needed > 90).mean():5.1f}%"
              f"   episodes that need over 117 somewhere: {episodes_over:4d} of {group['episode_id'].nunique()}")
    radial_deg = numpy.degrees(numpy.arctan2(frames["tip_y_m"], frames["tip_x_m"]))
    from_radial = pandas.Series(numpy.abs(method_common.wrapped_degrees(frames["travel_table_deg"] - radial_deg)), index = frames.index)
    print("   the puck's travel direction against 'straight away from the base' (radial), rows with a known direction:")
    for low, high in ((0, 45), (45, 90), (90, 110), (110, 135), (135, 180)):
        is_band = is_known & (from_radial > low) & (from_radial <= high)
        print(f"   {low:3d} to {high:3d} degrees from radial: {100 * is_band.sum() / is_known.sum():5.1f} % of rows; of these the rule needs joint 4 beyond 117 in"
              f" {100 * (frames.loc[is_band, 'required_joint_4_deg'].abs() > limit).mean():5.1f} %")

    print()
    print("3. The size of one recorded step by the recording rule (1 cm of tip travel or 5 degrees on a joint)")
    joint_columns = [f"joint_{number}" for number in range(1, 7)]
    grouped = frames.groupby("episode_id", sort = False)
    is_consecutive = (grouped["row_in_episode"].shift(-1) - frames["row_in_episode"]) == 1
    joint_change_deg = numpy.degrees((grouped[joint_columns].shift(-1) - frames[joint_columns]).abs())
    largest_joint_deg = joint_change_deg.max(axis = 1)
    is_sized = is_step & is_consecutive & (grouped["action"].shift(-1) != "stop")
    size_all = numpy.maximum(length_cm / settings["unit_length_cm"], largest_joint_deg / settings["unit_turn_deg"])[is_sized]
    size_model = numpy.maximum(length_cm / settings["unit_length_cm"], twist_deg.abs() / settings["unit_turn_deg"])[is_sized]
    print(f"   rows used (a step followed by the next recorded row, not the last step before the stop): {int(is_sized.sum())}")
    for name, size in (("all six joints", size_all), ("tip travel and forearm twist only (what the model predicts)", size_model)):
        print(f"   size by {name}: 5th percentile {numpy.percentile(size, 5):.2f}, 25th {numpy.percentile(size, 25):.2f}, median {numpy.median(size):.2f},"
              f" 75th {numpy.percentile(size, 75):.2f}, 95th {numpy.percentile(size, 95):.2f}; under 0.8 in {100 * (size < 0.8).mean():.1f} %, under 0.5 in"
              f" {100 * (size < 0.5).mean():.1f} %")
    which_joint = joint_change_deg[is_sized].to_numpy().argmax(axis = 1) + 1
    is_turn_cut = ((largest_joint_deg[is_sized] / settings["unit_turn_deg"]) > (length_cm[is_sized] / settings["unit_length_cm"])).to_numpy()
    print(f"   steps whose size comes from a joint turn, not from tip travel: {100 * is_turn_cut.mean():.1f} %; the joint that turned most in those steps: "
          + ", ".join(f"joint {joint} {100 * (which_joint[is_turn_cut] == joint).mean():.1f} %" for joint in range(1, 7)))
    last_steps = frames[is_step & (grouped["action"].shift(-1) == "stop")]
    print(f"   the last step before the stop row ({len(last_steps)} rows): median length {last_steps['length_cm'].median():.2f} cm, under 0.5 cm in"
          f" {100 * (last_steps['length_cm'] < 0.5).mean():.0f} %")

    print()
    print("4. How much each shape weighs in training, now and after the plan's recording round 2")
    counts = frames.groupby("sheet").agg(pictures = ("frame_index", "size"), episodes = ("episode_id", "nunique"))
    counts["pictures_per_episode"] = counts["pictures"] / counts["episodes"]
    counts["share"] = counts["pictures"] / counts["pictures"].sum()
    print(counts.round(3).to_string())
    pictures_per_new_episode = frames[frames["sheet"] != "path_01"].groupby("episode_id").size().mean()
    new_pictures = settings["new_shapes"] * settings["episodes_per_new_shape"] * pictures_per_new_episode
    old = counts["pictures"].to_numpy(dtype = float)
    new = numpy.full(settings["new_shapes"], new_pictures / settings["new_shapes"])
    for name, pictures in (("the six sheets now", old), ("the six sheets plus 12 new shapes x 4 episodes", numpy.concatenate([old, new]))):
        share = pictures / pictures.sum()
        print(f"   {name}: {pictures.sum():.0f} pictures; effective number of shapes when every picture counts the same: {1.0 / (share ** 2).sum():.2f}"
              f" (shapes: {len(pictures)}); largest share {100 * share.max():.1f} % (sheet 04)")
    print(f"   pictures per new episode assumed: {pictures_per_new_episode:.0f} (mean of the recorded episodes of sheets 02 to 06); new pictures"
          f" {new_pictures:.0f} = {100 * new_pictures / (old.sum() + new_pictures):.1f} % of the enlarged set, old-style pictures"
          f" {100 * old.sum() / (old.sum() + new_pictures):.1f} %")



if __name__ == "__main__":
    style_report()
