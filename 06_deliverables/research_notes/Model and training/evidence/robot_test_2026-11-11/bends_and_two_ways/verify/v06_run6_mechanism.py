"""
Verifier's own check of what happened in the back-and-forth of run 6 (sheet 04, steps 19 to 50).

A. From the log alone: is the back-and-forth an overshoot round a balance point (the answer points back towards a
   middle place whichever side the tip is on, and the fixed 1 cm step jumps across it), or do the answers flip without
   regard to where the tip is?
B. Model E run again on the logged pictures with the verifier's forward pass: the logged step, and the chunk's points
   3 and 5 steps ahead (how often they reverse).
C. Look-alike demonstration pictures: the 12 nearest of the 18,032 demonstration pictures, in two descriptions of the
   picture (the head's hidden layer, and the pooled backbone features before the head), against two yardsticks:
   demonstration pictures and their neighbours from other episodes, and 12 demonstration pictures drawn at random.

Read-only; no training; nothing connects to the robot. Prints to the screen and to v06_run6_mechanism.out.txt.
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy
import pandas
import torch

import verify_common

MECHANISM_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "run": "2026-11-11_174108_E_sheet04_run1",
    "all_sheets_folder": "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets",
    "all_sheets_name": "resnet18_model_4_chunk_stop_25_all_sheets",
    "first_step": 19,
    "last_step": 50,
    "neighbours": 12,
    "group_cm": 0.3,
    "group_size": 3,
    "yardstick_pictures": 600,
    "seed": 111126
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def hidden_and_outputs(loaded: Dict, pictures: numpy.ndarray) -> Dict[str, numpy.ndarray]:
    """
    Return the model's outputs, the head's hidden layer (through a hook on the head's shared layer) and the head's
    standardised input (the pooled backbone features) for a set of pictures.
    """

    model = loaded["model"]
    kept = {}
    hook_hidden = model.head.shared.register_forward_hook(lambda module, inputs, output: kept.__setitem__("hidden", output.detach().numpy().copy()))
    hook_input = model.head.shared.register_forward_hook(lambda module, inputs, output: kept.__setitem__("features", inputs[0].detach().numpy().copy()))
    parts = []
    torch.set_num_threads(verify_common.VERIFY_PARAMETERS["threads"])
    with torch.inference_mode():
        for start in range(0, len(pictures), verify_common.VERIFY_PARAMETERS["batch_size"]):
            batch = torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + verify_common.VERIFY_PARAMETERS["batch_size"]]))
            inputs, _ = verify_common.augmented_pictures(batch, loaded["settings"], numpy.random.default_rng(0), is_training = False)
            outputs = model(inputs)
            part = {name: value.numpy() for name, value in outputs.items()}
            part["hidden"] = kept["hidden"].astype(numpy.float32)
            part["features"] = kept["features"].astype(numpy.float16)
            parts.append(part)
    hook_hidden.remove()
    hook_input.remove()
    return {name: numpy.concatenate([part[name] for part in parts]) for name in parts[0]}


def neighbour_summary(labels: numpy.ndarray, twists: numpy.ndarray, parameters: Dict) -> Dict:
    """
    Summarise the labelled moves of one set of neighbours.

    Parameters
    ----------
    labels : numpy.ndarray
        (neighbours, 2): forward and left cm of each neighbour's labelled step.
    twists : numpy.ndarray
        (neighbours,): each neighbour's labelled forearm twist.

    Returns
    -------
    dict
        "resultant" (length of the mean unit direction of the real moves), "mean_length_cm", "length_of_mean_cm",
        "left_group", "right_group", "is_both_groups", "mean_left_cm", "is_both_twists".
    """

    lengths = numpy.linalg.norm(labels, axis = 1)
    real = labels[lengths >= parameters["group_cm"]]
    resultant = float(numpy.linalg.norm((real / numpy.linalg.norm(real, axis = 1, keepdims = True)).mean(axis = 0))) if len(real) else numpy.nan
    left_group = int((labels[:, 1] >= parameters["group_cm"]).sum())
    right_group = int((labels[:, 1] <= -parameters["group_cm"]).sum())
    return {"resultant": resultant, "mean_length_cm": float(lengths.mean()), "length_of_mean_cm": float(numpy.linalg.norm(labels.mean(axis = 0))),
            "left_group": left_group, "right_group": right_group,
            "is_both_groups": left_group >= parameters["group_size"] and right_group >= parameters["group_size"],
            "mean_left_cm": float(labels[:, 1].mean()),
            "is_both_twists": bool((twists >= 2.0).sum() >= parameters["group_size"] and (twists <= -2.0).sum() >= parameters["group_size"])}


if __name__ == "__main__":
    settings = dict(MECHANISM_PARAMETERS)
    run_folder = settings["model_runs_folder"] / settings["run"]
    steps = pandas.read_csv(run_folder / "steps.csv")
    moves = steps[steps["action"] == "model_step"].reset_index(drop = True)
    moves["asked_cm"] = numpy.hypot(moves["model_tip_forward_cm"], moves["model_tip_left_cm"])
    moves["asked_picture_deg"] = numpy.degrees(numpy.arctan2(moves["model_tip_left_cm"], moves["model_tip_forward_cm"]))
    moves["asked_table_deg"] = moves["picture_heading_deg"] + moves["asked_picture_deg"]
    part = moves[(moves["step"] >= settings["first_step"]) & (moves["step"] <= settings["last_step"])].reset_index(drop = True)

    say("== A. overshoot round a balance point, or flips without regard to the place? (log only) ==")
    tips = 100.0 * part[["tip_x", "tip_y"]].to_numpy()
    centred = tips - tips.mean(axis = 0)
    # the axis along which the tip swung: the main axis of the tip positions
    _, _, axes = numpy.linalg.svd(centred, full_matrices = False)
    axis = axes[0]
    along = centred @ axis
    asked_unit = numpy.stack([numpy.cos(numpy.radians(part["asked_table_deg"])), numpy.sin(numpy.radians(part["asked_table_deg"]))], axis = 1)
    asked_along_unit = asked_unit @ axis
    asked_along_cm = asked_along_unit * part["asked_cm"].to_numpy()
    say(f"swing axis on the table: ({axis[0]:+.2f}, {axis[1]:+.2f}); tip positions along it from {along.min():+.1f} to {along.max():+.1f} cm; "
        f"spread across it {float(numpy.std(centred @ axes[1])):.2f} cm (standard deviation)")
    say(f"correlation between where the tip is along the swing axis and the asked direction's part along it: unit direction "
        f"{float(numpy.corrcoef(along, asked_along_unit)[0, 1]):+.2f}, with its length {float(numpy.corrcoef(along, asked_along_cm)[0, 1]):+.2f} "
        f"(-1 = always pointing back to the middle)")
    is_back = numpy.sign(asked_along_unit) == -numpy.sign(along)
    say(f"answers pointing back towards the middle of the swing: {int(is_back.sum())} of {len(part)}; among the {int((numpy.abs(along) > 0.5).sum())} pictures "
        f"more than 0.5 cm from the middle: {int((is_back & (numpy.abs(along) > 0.5)).sum())}")
    slope, intercept = numpy.polyfit(along, asked_along_cm, 1)
    say(f"straight-line fit: asked part along the axis = {slope:+.3f} cm per cm of tip place {intercept:+.3f} cm; the fixed step moved 1 cm per picture, so "
        f"the loop gain was about {abs(1.0 / (numpy.abs(asked_along_cm).mean())):.1f} times the asked move (mean asked part along the axis "
        f"{float(numpy.abs(asked_along_cm).mean()):.2f} cm)")
    forward_table = numpy.stack([numpy.cos(numpy.radians(part["picture_heading_deg"])), numpy.sin(numpy.radians(part["picture_heading_deg"]))], axis = 1)
    say(f"asked forward part (picture frame) over these pictures: mean {part['model_tip_forward_cm'].mean():+.3f} cm, largest {part['model_tip_forward_cm'].max():+.2f}; "
        f"asked twist: mean {part['model_twist_deg'].mean():+.2f} deg, mean size {part['model_twist_deg'].abs().mean():.2f}; stop chance highest "
        f"{part['model_stop_probability'].max():.3f}")
    say(f"the swing axis against the camera: angle between the swing axis and the picture's left axis {float(numpy.degrees(numpy.arccos(abs(float((numpy.stack([-forward_table[:, 1], forward_table[:, 0]], axis = 1).mean(axis = 0)) @ axis))))):.0f} deg")

    say("")
    say("== B. model E again on the logged pictures (verifier's forward pass) ==")
    fold_runs = verify_common.VERIFY_PARAMETERS["fold_runs_folder"]
    loaded = verify_common.load_saved_model(settings["all_sheets_name"], fold_runs / settings["all_sheets_folder"], "all_sheets")
    pictures = []
    for frame in moves["frame"]:
        picture = cv2.imdecode(numpy.frombuffer((run_folder / frame).read_bytes(), dtype = numpy.uint8), cv2.IMREAD_COLOR)
        pictures.append(cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA))
    run_outputs = hidden_and_outputs(loaded, numpy.stack(pictures))
    difference = numpy.abs(run_outputs["exact"][:, :2] - moves[["model_tip_forward_cm", "model_tip_left_cm"]].to_numpy()).max()
    say(f"largest difference between the recomputed and the logged step: {float(difference):.5f} cm; twist {float(numpy.abs(run_outputs['exact'][:, 2] - moves['model_twist_deg']).max()):.5f} deg")
    chunk = run_outputs["chunk"].reshape(len(moves), 5, 2)
    is_part = ((moves["step"] >= settings["first_step"]) & (moves["step"] <= settings["last_step"])).to_numpy()
    for label, vectors in [("exact step", run_outputs["exact"][:, :2]), ("chunk 1 ahead", chunk[:, 0]), ("chunk 2 ahead", chunk[:, 1]), ("chunk 3 ahead", chunk[:, 2]),
                           ("chunk 4 ahead", chunk[:, 3]), ("chunk 5 ahead", chunk[:, 4])]:
        angle = numpy.degrees(numpy.arctan2(vectors[:, 1], vectors[:, 0]))
        flips = int((verify_common.angle_between(angle[is_part][:-1], angle[is_part][1:]) > 90).sum())
        say(f"   {label:14s}: reversals over 90 deg in steps {settings['first_step']} to {settings['last_step']}: {flips} of {int(is_part.sum()) - 1}; median length there "
            f"{float(numpy.median(numpy.linalg.norm(vectors[is_part], axis = 1))):.2f} cm, in steps 0 to 4 {float(numpy.median(numpy.linalg.norm(vectors[:5], axis = 1))):.2f} cm")

    say("")
    say("== C. look-alike demonstration pictures ==")
    frames = verify_common.demonstration_rows()
    cache_path = verify_common.VERIFY_PARAMETERS["cache_folder"] / "all_sheets_e_demonstrations.npz"
    cache_path.parent.mkdir(parents = True, exist_ok = True)
    if cache_path.exists() and numpy.array_equal(numpy.load(cache_path)["frame_index"], frames["frame_index"].to_numpy()):
        demonstration = dict(numpy.load(cache_path))
    else:
        demonstration = hidden_and_outputs(loaded, verify_common.cached_training_pictures(frames["frame_index"].to_numpy()))
        demonstration["frame_index"] = frames["frame_index"].to_numpy()
        numpy.savez(cache_path, **demonstration)
    labels = frames[["step_forward_cm", "step_left_cm"]].to_numpy()
    twists = frames["twist_deg"].to_numpy()
    episodes = pandas.factorize(frames["episode_id"])[0]
    generator = numpy.random.default_rng(settings["seed"])
    say(f"demonstration pictures {len(frames)}; model E on its own training pictures: predicted length median {float(numpy.median(numpy.linalg.norm(demonstration['exact'][:, :2], axis = 1))):.2f} cm")
    for space in ["hidden", "features"]:
        base = demonstration[space].astype(numpy.float32)
        query = run_outputs[space].astype(numpy.float32)
        say(f"-- description of the picture: {space} ({base.shape[1]} numbers) --")
        # yardstick: demonstration pictures and their neighbours from OTHER episodes
        chosen = generator.choice(len(frames), settings["yardstick_pictures"], replace = False)
        yard, yard_nearest = [], []
        for index in chosen:
            distance = numpy.linalg.norm(base - base[index], axis = 1)
            distance[episodes == episodes[index]] = numpy.inf
            nearest = numpy.argsort(distance)[:settings["neighbours"]]
            summary = neighbour_summary(labels[nearest], twists[nearest], settings)
            summary["nearest"] = float(distance[nearest[0]])
            summary["place"] = frames["place"].iloc[index]
            yard.append(summary)
        yard = pandas.DataFrame(yard)
        say(f"yardstick ({len(yard)} random demonstration pictures, neighbours from other episodes): nearest distance median {yard['nearest'].median():.2f}, 90th percentile "
            f"{yard['nearest'].quantile(0.9):.2f}; resultant median {yard['resultant'].median():.2f}; both groups {100 * yard['is_both_groups'].mean():.0f} %; "
            f"length of the neighbours' mean move {yard['length_of_mean_cm'].mean():.2f} cm against their mean length {yard['mean_length_cm'].mean():.2f}")
        far = yard[yard["nearest"] >= yard["nearest"].quantile(0.9)]
        say(f"   the tenth of them farthest from other episodes ({len(far)}): resultant median {far['resultant'].median():.2f}, both groups {100 * far['is_both_groups'].mean():.0f} %")
        for place in ["hairpin", "sharp bend"]:
            group = yard[yard["place"] == place]
            say(f"   {place}: {len(group)} pictures, resultant median {group['resultant'].median():.2f}, both groups {100 * group['is_both_groups'].mean():.0f} %")
        # 12 demonstration pictures at random (no likeness at all), all sheets and sheet 04 only
        for label, pool in [("all sheets", numpy.arange(len(frames))), ("sheet 04 only", numpy.flatnonzero((frames["sheet"] == "path_04").to_numpy())),
                            ("hairpin and sharp-bend pictures", numpy.flatnonzero(frames["place"].isin(["hairpin", "sharp bend"]).to_numpy()))]:
            random_sets = pandas.DataFrame([neighbour_summary(labels[picked], twists[picked], settings)
                                            for picked in (generator.choice(pool, settings["neighbours"], replace = False) for _ in range(2000))])
            say(f"   12 pictures at random, {label}: resultant median {random_sets['resultant'].median():.2f}, both groups {100 * random_sets['is_both_groups'].mean():.0f} %, "
                f"length of the mean move {random_sets['length_of_mean_cm'].mean():.2f} cm against mean length {random_sets['mean_length_cm'].mean():.2f}")
        # the run's pictures
        rows = []
        for position in range(len(moves)):
            distance = numpy.linalg.norm(base - query[position], axis = 1)
            nearest = numpy.argsort(distance)[:settings["neighbours"]]
            summary = neighbour_summary(labels[nearest], twists[nearest], settings)
            summary["step"] = int(moves["step"].iloc[position])
            summary["nearest"] = float(distance[nearest[0]])
            summary["from_end_A"] = int((frames["travel_sign"].to_numpy()[nearest] > 0).sum())
            summary["sheet_04"] = int((frames["sheet"].to_numpy()[nearest] == "path_04").sum())
            summary["hairpin_or_sharp"] = int(frames["place"].isin(["hairpin", "sharp bend"]).to_numpy()[nearest].sum())
            left_rows = nearest[labels[nearest, 1] >= settings["group_cm"]]
            right_rows = nearest[labels[nearest, 1] <= -settings["group_cm"]]
            summary["left_group_from_A"] = int((frames["travel_sign"].to_numpy()[left_rows] > 0).sum())
            summary["right_group_from_A"] = int((frames["travel_sign"].to_numpy()[right_rows] > 0).sum())
            summary["left_group_twist"] = float(twists[left_rows].mean()) if len(left_rows) else numpy.nan
            summary["right_group_twist"] = float(twists[right_rows].mean()) if len(right_rows) else numpy.nan
            rows.append(summary)
        table = pandas.DataFrame(rows)
        swing = table[(table["step"] >= settings["first_step"]) & (table["step"] <= settings["last_step"])]
        start = table[table["step"] <= 4]
        say(f"run 6, steps {settings['first_step']} to {settings['last_step']}: nearest distance median {swing['nearest'].median():.2f} (steps 0 to 4: {start['nearest'].median():.2f}); "
            f"resultant median {swing['resultant'].median():.2f}; both groups {100 * swing['is_both_groups'].mean():.0f} %; both twist signs {100 * swing['is_both_twists'].mean():.0f} %; "
            f"length of the neighbours' mean move {swing['length_of_mean_cm'].mean():.2f} cm against their mean length {swing['mean_length_cm'].mean():.2f}")
        answers_left = part["model_tip_left_cm"].to_numpy()
        larger_left = (swing["left_group"] > swing["right_group"]).to_numpy()
        larger_right = (swing["right_group"] > swing["left_group"]).to_numpy()
        decided = larger_left | larger_right
        agrees = (larger_left & (answers_left > 0)) | (larger_right & (answers_left < 0))
        say(f"   the answer went to the side of the larger group in {int(agrees.sum())} of {int(decided.sum())} pictures with unequal groups; correlation between the "
            f"neighbours' mean left part and the answer's left part {float(numpy.corrcoef(swing['mean_left_cm'], answers_left)[0, 1]):+.2f}")
        say(f"   who the neighbours are (mean over these pictures, of 12): from sheet 04 {swing['sheet_04'].mean():.1f}, from episodes travelling A to B {swing['from_end_A'].mean():.1f}, "
            f"at hairpins or sharp bends {swing['hairpin_or_sharp'].mean():.1f}")
        say(f"   left group: mean twist {swing['left_group_twist'].mean():+.1f} deg, share from A-to-B episodes {swing['left_group_from_A'].sum() / max(1, swing['left_group'].sum()):.2f}; "
            f"right group: mean twist {swing['right_group_twist'].mean():+.1f} deg, share from A-to-B episodes {swing['right_group_from_A'].sum() / max(1, swing['right_group'].sum()):.2f}")
        table.to_csv(Path(__file__).resolve().parent / f"v06_run6_neighbours_{space}.csv", index = False)

    (Path(__file__).resolve().parent / "v06_run6_mechanism.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
