"""
Can one picture tell where the forearm stands (joint 4), and do the models act as if they knew?

The plan's rules 3 and 4 ("turn the forearm until the line runs up the picture", "one way through a loop: the walk round
the puck or unwinding early") need the forearm's angle: whether there is room left to turn, and which way to unwind.
The camera turns with the forearm, so the angle is not in the picture by construction; it could still show through the
robot's own base coming into view, the perspective of the table or the paper's edges.

A. A read-out, not a model: a least-squares fit (ridge, closed form, seconds on the processor) from the SAVED frozen
   ResNet-18 features of 9.11.26 to joint 4, fitted on five sheets and scored on the sixth. No network is trained and
   nothing is written but this script's text output. Frozen ImageNet features are a floor: a fine-tuned network could
   read more.
B. The behaviour of the fold models (each scored on the sheet it never saw; the outputs the step-length analyst saved
   and the verifier checked): near the forearm's limit, and in the stretches where the demonstrators unwind the forearm,
   does the model twist the way the demonstrators do?

Run from this folder:
    python m03_forearm_angle_from_the_picture.py > m03_forearm_angle_from_the_picture.out.txt
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import method_common

READ_OUT_PARAMETERS = {
    "feature_folder": Path("D:/AI_Project/03_data/processed/v01_training_index/features"),
    "feature_files": {"one summary of the picture (512 numbers)": "resnet18_imagenet_global.npy",
                      "a grid of 16 cells (8,192 numbers, projected to 2,048)": "resnet18_imagenet_grid.npy"},
    "projected_size": 2_048,
    "projection_seed": 11_11_26,
    # the ridge strength as a multiple of the number of training rows; the larger one is the safer fit
    "ridge_per_row": [0.01, 0.1, 1.0],
    "near_limit_deg": 100.0,
    "turned_deg": 60.0,
    # part B: windows of this many rows (as in the verifier's step-length script v08) and what counts as a real turn
    "window_rows": 10,
    "net_turn_least_deg": 30.0
}



def ridge_read_out(
        train_features: numpy.ndarray,
        train_target: numpy.ndarray,
        test_features: numpy.ndarray,
        ridge: float
) -> numpy.ndarray:
    """
    Return the least-squares read-out of a target on the test rows.

    Parameters
    ----------
    train_features, test_features : numpy.ndarray
        (rows, numbers) float64.
    train_target : numpy.ndarray
        (rows,) the quantity to read out.
    ridge : float
        The ridge strength added to the diagonal of the normal equations (features standardised first).

    Returns
    -------
    numpy.ndarray
        The read-out on the test rows.
    """

    mean = train_features.mean(axis = 0)
    spread = train_features.std(axis = 0) + 1e-6
    train_scaled = (train_features - mean) / spread
    test_scaled = (test_features - mean) / spread
    target_mean = train_target.mean()
    gram = train_scaled.T @ train_scaled
    gram[numpy.diag_indices_from(gram)] += ridge
    weights = numpy.linalg.solve(gram, train_scaled.T @ (train_target - target_mean))
    return test_scaled @ weights + target_mean



def area_under_curve(
        score: numpy.ndarray,
        is_positive: numpy.ndarray
) -> float:
    """Return the area under the ROC curve of ``score`` for the positive rows (rank formula, ties averaged)."""

    if is_positive.sum() == 0 or (~is_positive).sum() == 0:
        return float("nan")
    ranks = pandas.Series(score).rank().to_numpy()
    positives = int(is_positive.sum())
    negatives = int((~is_positive).sum())
    return float((ranks[is_positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))



def read_out_report(
        frames: pandas.DataFrame,
        settings: Dict
) -> None:
    """Print part A: joint 4 read from frozen features, one sheet left out at a time."""

    joint_4_deg = numpy.degrees(frames["joint_4"].to_numpy())
    sheets = frames["sheet"].to_numpy()
    frame_index = frames["frame_index"].to_numpy()
    start_kind = frames["start_kind"].astype(str).to_numpy()
    print("A. Joint 4 read out of frozen ResNet-18 features by a least-squares fit, the scored sheet left out of the fit")
    print(f"   rows: {len(frames)}; joint 4 over all rows: spread (standard deviation) {joint_4_deg.std():.1f} degrees, |joint 4| of {settings['near_limit_deg']:.0f} or more in"
          f" {100 * (numpy.abs(joint_4_deg) >= settings['near_limit_deg']).mean():.1f} % of rows, of {settings['turned_deg']:.0f} or more in"
          f" {100 * (numpy.abs(joint_4_deg) >= settings['turned_deg']).mean():.1f} %")
    for label, file_name in settings["feature_files"].items():
        features = numpy.load(settings["feature_folder"] / file_name, mmap_mode = "r")
        selected = numpy.asarray(features[frame_index], dtype = numpy.float32)
        if selected.shape[1] > settings["projected_size"]:
            # a fixed random projection keeps what a linear read-out can use and makes the normal equations small
            projection = numpy.random.default_rng(settings["projection_seed"]).standard_normal(
                (selected.shape[1], settings["projected_size"])).astype(numpy.float32) / numpy.sqrt(settings["projected_size"])
            selected = selected @ projection
        selected = selected.astype(numpy.float64)
        print(f"   features: {label}")
        for ridge_per_row in settings["ridge_per_row"]:
            read = numpy.zeros(len(frames))
            for sheet in sorted(set(sheets)):
                is_test = sheets == sheet
                read[is_test] = ridge_read_out(selected[~is_test], joint_4_deg[~is_test], selected[is_test], ridge_per_row * (~is_test).sum())
            cells = []
            for sheet in sorted(set(sheets)):
                is_test = sheets == sheet
                error = read[is_test] - joint_4_deg[is_test]
                explained = 1.0 - (error ** 2).sum() / ((joint_4_deg[is_test] - joint_4_deg[~is_test].mean()) ** 2).sum()
                correlation = numpy.corrcoef(read[is_test], joint_4_deg[is_test])[0, 1]
                cells.append(f"{sheet[-2:]}: r {correlation:+.2f}, explained {explained:+.2f}, median error {numpy.median(numpy.abs(error)):.0f}")
            is_turned = numpy.abs(joint_4_deg) >= settings["turned_deg"]
            side_right = (numpy.sign(read[is_turned]) == numpy.sign(joint_4_deg[is_turned])).mean()
            is_near = numpy.abs(joint_4_deg) >= settings["near_limit_deg"]
            print(f"      ridge {ridge_per_row:5.2f} per row | " + " | ".join(cells))
            print(f"         all sheets pooled: r {numpy.corrcoef(read, joint_4_deg)[0, 1]:+.2f}, median error {numpy.median(numpy.abs(read - joint_4_deg)):.0f} degrees"
                  f" (always answering 0: {numpy.median(numpy.abs(joint_4_deg)):.0f}); side right where the forearm is turned {settings['turned_deg']:.0f} or more:"
                  f" {100 * side_right:.0f} %; 'within {settings['joint_4_most_deg'] - settings['near_limit_deg']:.0f} degrees of the limit' told from the read-out's size:"
                  f" area under the curve {area_under_curve(numpy.abs(read), is_near):.2f} (0.5 = chance)")
            if ridge_per_row == settings["ridge_per_row"][1]:
                near = numpy.abs(read[is_near])
                far = numpy.abs(read[~is_near])
                limit = numpy.percentile(far, 90)
                print(f"         at a read-out size that 10 % of the other rows also reach ({limit:.0f} degrees), {100 * (near >= limit).mean():.0f} % of the rows near the limit are found")
                for kinds, name in ((("SQP", "SQL"), "sheet 02 lying square (SQP, SQL)"), (("STD",), "sheet 02 standard starts")):
                    is_group = (sheets == "path_02") & numpy.isin(start_kind, kinds)
                    if is_group.sum() > 50:
                        print(f"         {name}: {int(is_group.sum())} rows, r {numpy.corrcoef(read[is_group], joint_4_deg[is_group])[0, 1]:+.2f},"
                              f" median error {numpy.median(numpy.abs(read[is_group] - joint_4_deg[is_group])):.0f} degrees")



def behaviour_report(
        frames: pandas.DataFrame,
        settings: Dict
) -> None:
    """Print part B: what the fold models ask for near the limit and where the demonstrators unwind."""

    print()
    print("B. The fold models on the sheets they never saw: twist asked against twist demonstrated, by where the forearm stands")
    print("   ('towards the limit' = the twist has the sign of joint 4; negative = unwinding)")
    for recipe, path in settings["fold_outputs"].items():
        data = numpy.load(path)
        predicted = pandas.DataFrame({"frame_index": data["frame_index"], "asked_twist": data["exact"][:, 2]})
        table = frames.merge(predicted, on = "frame_index", how = "left")
        table = table[table["action"] != "stop"].copy()
        table["joint_4_deg"] = numpy.degrees(table["joint_4"])
        table["side"] = numpy.sign(table["joint_4_deg"])
        table["shown_towards"] = table["side"] * table["forearm_change_deg"].fillna(0.0)
        table["asked_towards"] = table["side"] * table["asked_twist"]
        print(f"   recipe {recipe}: {len(table)} step rows")
        print("      |joint 4|        rows   shown mean   asked mean   shown unwinds 2 deg or more   of these the model also unwinds   shown winds on 2 or more   of these the model also winds on")
        for low, high in ((0, 30), (30, 60), (60, 90), (90, 105), (105, 121)):
            band = table[(table["joint_4_deg"].abs() >= low) & (table["joint_4_deg"].abs() < high)]
            unwinding = band[band["shown_towards"] <= -2.0]
            winding = band[band["shown_towards"] >= 2.0]
            print(f"      {low:3d} to {high:3d}   {len(band):7d}   {band['shown_towards'].mean():+10.2f}   {band['asked_towards'].mean():+10.2f}"
                  f"   {100 * len(unwinding) / len(band):25.1f}%   {100 * (unwinding['asked_towards'] < 0).mean():29.1f}%"
                  f"   {100 * len(winding) / len(band):22.1f}%   {100 * (winding['asked_towards'] > 0).mean():30.1f}%")
        window = settings["window_rows"]
        least = settings["net_turn_least_deg"]
        rows = []
        for episode_id, group in table.groupby("episode_id", sort = False):
            shown = group["forearm_change_deg"].fillna(0.0).to_numpy()
            asked = group["asked_twist"].to_numpy()
            joint_4 = group["joint_4_deg"].to_numpy()
            for start in range(0, len(group) - window + 1, window):
                part = slice(start, start + window)
                rows.append({"sheet": group["sheet"].iloc[0], "start_joint_4": joint_4[start], "shown_net": shown[part].sum(), "asked_net": asked[part].sum()})
        windows = pandas.DataFrame(rows)
        windows["side"] = numpy.sign(windows["start_joint_4"])
        kinds = {
            f"unwinding from {settings['turned_deg']:.0f} or more (net {least:.0f} or more towards zero)":
                (windows["start_joint_4"].abs() >= settings["turned_deg"]) & (windows["side"] * windows["shown_net"] <= -least),
            f"winding on from under {settings['turned_deg']:.0f} (net {least:.0f} or more away from zero)":
                (windows["start_joint_4"].abs() < settings["turned_deg"]) & (windows["shown_net"].abs() >= least)
                & (numpy.sign(windows["shown_net"]) * (windows["start_joint_4"] + windows["shown_net"]) > windows["start_joint_4"].abs()),
            f"every window with a net turn of {least:.0f} or more": windows["shown_net"].abs() >= least
        }
        print(f"      windows of {window} rows ({len(windows)}):")
        for name, is_kind in kinds.items():
            part = windows[is_kind]
            share = (numpy.sign(part["shown_net"]) * part["asked_net"]).sum() / part["shown_net"].abs().sum()
            same_sign = (numpy.sign(part["shown_net"]) * part["asked_net"] > 0).mean()
            per_sheet = ", ".join(f"{sheet[-2:]} {(numpy.sign(group['shown_net']) * group['asked_net']).sum() / group['shown_net'].abs().sum():.2f} ({len(group)})"
                                  for sheet, group in part.groupby("sheet"))
            print(f"         {name}: {len(part)} windows; the model gives back {share:.2f} of the shown net turn, right sense in {100 * same_sign:.0f} %"
                  f" (per held-out sheet, windows in brackets: {per_sheet})")



def forearm_report(parameters: Optional[Dict] = None) -> None:
    """Print both parts named in the module's text."""

    settings = method_common.merged_parameters(READ_OUT_PARAMETERS)
    settings.update(parameters or {})
    frames = method_common.cleaned_frames(settings)
    read_out_report(frames, settings)
    behaviour_report(frames, settings)



if __name__ == "__main__":
    forearm_report()
