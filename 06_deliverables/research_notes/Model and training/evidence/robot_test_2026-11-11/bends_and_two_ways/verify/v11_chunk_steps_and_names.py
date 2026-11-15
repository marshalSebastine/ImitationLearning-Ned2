"""
Verifier's own check of two option claims.

1. Carrying out 2 or 3 steps of one chunk: how far the chunk's place 2 and 3 steps ahead is from where the
   demonstration's tip was (recipe E fold models, held-out sheets; cache of v05_offline_by_place.py), and how much the
   demonstrations twist the forearm over 2 and 3 steps.
2. The backup model's move names: is the probability split over opposite names at bends, and where the same model's
   exact step is short? (The six backup fold models are run on their held-out sheets with the verifier's forward pass;
   cached; about 4 minutes the first time.)

Read-only; no training. Prints to the screen and to v11_chunk_steps_and_names.out.txt.
"""

from pathlib import Path
from typing import List

import numpy
import pandas
import torch

import verify_common
from training_targets import DIRECTION_ANGLES_DEG

OPTION_PARAMETERS = {
    "chunk_cache": "2026-11-10_resnet18_chunk_stop_weight_25.npz",
    "names_folder": "2026-11-09_resnet18_backup_names_plus_exact",
    "names_run": "resnet18_backup_names_plus_exact",
    "opposite_from_deg": 90.0,
    "split_share": 0.25,
    "short_cm": 0.3
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


if __name__ == "__main__":
    settings = dict(OPTION_PARAMETERS)
    frames = verify_common.demonstration_rows()
    cache_folder = verify_common.VERIFY_PARAMETERS["cache_folder"]
    moving = ~frames["is_stop"].to_numpy()
    codes = pandas.factorize(frames["episode_id"])[0]

    say("== 1. the chunk's place 2 and 3 steps ahead against the demonstrated place (recipe E, held-out sheets) ==")
    held_out = numpy.load(cache_folder / settings["chunk_cache"])
    assert numpy.array_equal(held_out["frame_index"], frames["frame_index"].to_numpy())
    chunk = held_out["chunk"].reshape(len(frames), 5, 2)
    ahead = verify_common.tips_ahead(frames, 3)
    twist = frames["twist_deg"].to_numpy()
    for steps in [1, 2, 3]:
        has_future = numpy.zeros(len(frames), dtype = bool)
        has_future[:-steps] = (codes[steps:] == codes[:-steps]) & moving[:-steps]
        error = numpy.linalg.norm(chunk[:, steps - 1, :] - ahead[:, steps - 1, :], axis = 1)
        twist_sum = numpy.abs(sum(numpy.roll(twist, -offset) for offset in range(steps)))
        parts = []
        for place in ["all places", "sharp bend", "hairpin"]:
            chosen = has_future & ((frames["place"] != "no place").to_numpy() if place == "all places" else (frames["place"] == place).to_numpy())
            parts.append(f"{place}: chunk error median {numpy.median(error[chosen]):.2f} cm, demonstrated twist over these steps median "
                         f"{numpy.median(twist_sum[chosen]):.1f} deg (90th percentile {numpy.percentile(twist_sum[chosen], 90):.1f})")
        say(f"   {steps} step(s) ahead | " + " | ".join(parts))

    say("")
    say("== 2. the backup model's move names (fold models on their held-out sheets, verifier's forward pass) ==")
    names_cache = cache_folder / f"{settings['names_folder']}.npz"
    if names_cache.exists() and numpy.array_equal(numpy.load(names_cache)["frame_index"], frames["frame_index"].to_numpy()):
        cached = numpy.load(names_cache)
        direction, exact = cached["direction"], cached["exact"]
        class_names = [str(name) for name in cached["class_names"]]
    else:
        run_folder = verify_common.VERIFY_PARAMETERS["fold_runs_folder"] / settings["names_folder"]
        direction, exact, class_names = None, numpy.zeros((len(frames), 5), dtype = numpy.float32), None
        for sheet in sorted(frames["sheet"].unique()):
            rows = numpy.flatnonzero((frames["sheet"] == sheet).to_numpy())
            loaded = verify_common.load_saved_model(settings["names_run"], run_folder, f"without_{sheet}")
            outputs = verify_common.forward_pass(loaded, verify_common.cached_training_pictures(frames["frame_index"].to_numpy()[rows]))
            if direction is None:
                direction = numpy.zeros((len(frames), outputs["direction"].shape[1]), dtype = numpy.float32)
                class_names = loaded["class_names"]
            direction[rows] = outputs["direction"]
            exact[rows] = outputs["exact"]
            print(f"   fold without_{sheet}: {len(rows)} pictures done", flush = True)
        numpy.savez(names_cache, frame_index = frames["frame_index"].to_numpy(), direction = direction, exact = exact, class_names = numpy.array(class_names))
    moving_columns = [index for index, name in enumerate(class_names[:direction.shape[1]]) if name in DIRECTION_ANGLES_DEG]
    nominal = numpy.array([DIRECTION_ANGLES_DEG[class_names[index]] for index in moving_columns])
    probabilities = torch.softmax(torch.from_numpy(direction[:, moving_columns]), dim = 1).numpy()
    top = probabilities.argmax(axis = 1)
    away = verify_common.angle_between(nominal[None, :], nominal[top][:, None]) > settings["opposite_from_deg"]
    opposite_share = (probabilities * away).sum(axis = 1)
    is_split = opposite_share >= settings["split_share"]
    say(f"direction outputs {direction.shape[1]}, names used {[class_names[index] for index in moving_columns]}")
    for place in verify_common.PLACES + ["all places"]:
        chosen = moving & ((frames["place"] != "no place").to_numpy() if place == "all places" else (frames["place"] == place).to_numpy())
        say(f"   {place:12s}: {int(chosen.sum()):5d} pictures; top probability median {numpy.median(probabilities.max(axis = 1)[chosen]):.3f}; probability on names more than "
            f"90 deg from the top name: mean {opposite_share[chosen].mean():.3f}; pictures with 0.25 or more there: {100 * is_split[chosen].mean():.1f} %")
    exact_cm = numpy.linalg.norm(exact[:, :2], axis = 1)
    with_place = moving & (frames["place"] != "no place").to_numpy()
    for low, high in [(0.0, 0.3), (0.3, 0.7), (0.7, 9.0)]:
        chosen = with_place & (exact_cm >= low) & (exact_cm < high)
        say(f"   the same model's exact step {low:.1f} to {high:.1f} cm: {int(chosen.sum()):5d} pictures; top probability median "
            f"{numpy.median(probabilities.max(axis = 1)[chosen]):.3f}; split pictures {100 * is_split[chosen].mean():.1f} %")

    (Path(__file__).resolve().parent / "v11_chunk_steps_and_names.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
