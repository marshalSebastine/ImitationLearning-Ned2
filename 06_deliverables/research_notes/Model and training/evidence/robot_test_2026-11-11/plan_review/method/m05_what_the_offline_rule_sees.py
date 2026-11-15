"""
The plan's rule for retraining (step 4: "5 points of path within 45 degrees, with both seeds", one shape left out at a
time) looked at with the numbers that exist: what the measure sees, what it misses, and how exact it is when a new
shape has only four episodes.

Measured on the fold models' answers on the sheets they never saw (recipe E; the outputs the step-length analyst saved
and the verifier checked); nothing is trained, nothing connects to the robot.
    1. Per held-out sheet: path within 45 degrees (the plan's measure, recomputed here and compared with the run's own
       metrics.json), next to the share of the demonstrated net turn the model gives back and the whole-episode stop.
    2. The first bends where the robot failed on 11.11.26 (sheets 03 and 04, standard starts from end A, rows until
       the forearm has turned 60 degrees): the same measures on exactly those rows.
    3. How exact the plan's measure is for a shape with four episodes: the spread of the score between episodes of one
       sheet, and of the mean of four episodes drawn at random.

Run from this folder:
    python m05_what_the_offline_rule_sees.py > m05_what_the_offline_rule_sees.out.txt
"""

import json
from typing import Dict, Optional

import numpy
import pandas

import method_common

OFFLINE_RULE_PARAMETERS = {
    "recipe": "E",
    "metrics_path": method_common.ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_chunk_stop_weight_25" / "metrics.json",
    # the report's path direction: from the tip now to where it is this many steps later, scored where that is at least
    # this long (config.TRAINING_METRIC_PARAMETERS "path_steps" and "path_minimum_cm")
    "path_steps": 3,
    "path_minimum_cm": 1.5,
    "window_rows": 10,
    "net_turn_least_deg": 30.0,
    "first_bend_turn_deg": 60.0,
    "episodes_per_new_shape": 4,
    "draws": 20_000,
    "seed": 11_11_26
}



def offline_rule_report(parameters: Optional[Dict] = None) -> None:
    """Print the three measurements named in the module's text."""

    from training_targets import chunk_targets

    settings = method_common.merged_parameters(OFFLINE_RULE_PARAMETERS)
    settings.update(parameters or {})
    frames = method_common.cleaned_frames(settings)
    data = numpy.load(settings["fold_outputs"][settings["recipe"]])
    answers = pandas.DataFrame({"frame_index": data["frame_index"], "asked_first_forward": data["chunk"][:, 0, 0], "asked_first_left": data["chunk"][:, 0, 1],
                                "asked_twist": data["exact"][:, 2], "asked_forward": data["exact"][:, 0], "asked_left": data["exact"][:, 1]})
    table = frames.merge(answers, on = "frame_index", how = "left")
    path = chunk_targets(table, settings["path_steps"])[:, settings["path_steps"] - 1, :]
    table["path_angle"] = numpy.degrees(numpy.arctan2(path[:, 1], path[:, 0]))
    table["is_path_row"] = (table["action"] != "stop") & (numpy.linalg.norm(path, axis = 1) >= settings["path_minimum_cm"])
    table["asked_angle"] = numpy.degrees(numpy.arctan2(table["asked_first_left"], table["asked_first_forward"]))
    table["is_path_right"] = numpy.abs(method_common.wrapped_degrees(table["asked_angle"] - table["path_angle"])) <= 45.0
    table["shown_twist"] = table["forearm_change_deg"].fillna(0.0)
    table["joint_4_deg"] = numpy.degrees(table["joint_4"])
    saved = {fold["fold"][-7:]: fold for fold in json.loads(settings["metrics_path"].read_text(encoding = "utf-8"))["folds"]}

    def net_turn_share(group: pandas.DataFrame) -> float:
        """Return the share of the demonstrated net turn the model gives back over the group's windows with a real turn."""

        given, shown = 0.0, 0.0
        for episode_id, episode in group[group["action"] != "stop"].groupby("episode_id", sort = False):
            shown_twist, asked_twist = episode["shown_twist"].to_numpy(), episode["asked_twist"].to_numpy()
            for start in range(0, len(episode) - settings["window_rows"] + 1, settings["window_rows"]):
                part = slice(start, start + settings["window_rows"])
                if abs(shown_twist[part].sum()) >= settings["net_turn_least_deg"]:
                    given += numpy.sign(shown_twist[part].sum()) * asked_twist[part].sum()
                    shown += abs(shown_twist[part].sum())
        return given / shown if shown else float("nan")

    print(f"1. Recipe {settings['recipe']}, each sheet scored by the model that never saw it")
    print("   sheet   path within 45 (this script / the run's metrics.json)   net turn given back (windows of 10 rows, 30 degrees or more)   stopped right / early / never (fixed 0.5, two in a row, metrics.json)")
    for sheet, group in table.groupby("sheet"):
        fold = saved[sheet]
        scored = group[group["is_path_row"]]
        print(f"   {sheet[-2:]}      {scored['is_path_right'].mean():.3f} / {fold['path_within_45_deg']:.3f}"
              f"                                              {net_turn_share(group):.2f}"
              f"                                                            {fold['fixed_two_episodes_stopped_right']:.2f} / {fold['fixed_two_episodes_stopped_early']:.2f} / {fold['fixed_two_episodes_never_stopped']:.2f}")
    per_sheet_path = table[table["is_path_row"]].groupby("sheet")["is_path_right"].mean()
    per_sheet_turn = pandas.Series({sheet: net_turn_share(group) for sheet, group in table.groupby("sheet")})
    per_sheet_stop = pandas.Series({sheet: saved[sheet]["fixed_two_episodes_stopped_right"] for sheet in per_sheet_path.index})
    print(f"   across the six sheets: path within 45 spans {100 * (per_sheet_path.max() - per_sheet_path.min()):.1f} points, the net turn given back"
          f" {per_sheet_turn.min():.2f} to {per_sheet_turn.max():.2f}, stopped right {per_sheet_stop.min():.2f} to {per_sheet_stop.max():.2f};"
          f" rank correlation of path within 45 with the net turn {per_sheet_path.rank().corr(per_sheet_turn.rank()):+.2f},"
          f" with stopped right {per_sheet_path.rank().corr(per_sheet_stop.rank()):+.2f} (six points only)")

    print()
    print(f"2. The first bends of sheets 03 and 04 (standard starts from end A, rows until joint 4 has turned {settings['first_bend_turn_deg']:.0f} degrees)")
    for sheet in ("path_03", "path_04"):
        group = table[(table["sheet"] == sheet) & (table["start_kind"] == "STD") & (table["start_end"] == "A") & (table["action"] != "stop")]
        parts = []
        for episode_id, episode in group.groupby("episode_id", sort = False):
            turned = numpy.abs(episode["joint_4_deg"].to_numpy() - episode["joint_4_deg"].to_numpy()[0])
            reached = numpy.flatnonzero(turned >= settings["first_bend_turn_deg"])
            parts.append(episode.iloc[:int(reached[0]) if len(reached) else len(episode)])
        bend = pandas.concat(parts)
        asked_cm = numpy.hypot(bend["asked_forward"], bend["asked_left"])
        shown_cm = 100.0 * numpy.hypot(bend["tip_forward_m"].fillna(0.0), bend["tip_left_m"].fillna(0.0))
        scored = bend[bend["is_path_row"]]
        signed = sum(numpy.sign(part["shown_twist"].sum()) * part["asked_twist"].sum() for part in parts)
        shown = sum(abs(part["shown_twist"].sum()) for part in parts)
        print(f"   sheet {sheet[-2:]}: {len(parts)} episodes, {len(bend)} rows ({len(scored)} scored for the path): path within 45 {scored['is_path_right'].mean():.3f};"
              f" the model gives back {signed / shown:.2f} of the shown turn; asked travel {asked_cm.sum() / shown_cm.sum():.2f} of the shown travel;"
              f" turn per cm with a fixed 1 cm step {(signed / len(bend)) / (shown / shown_cm.sum()):.2f} of the shown, with the model's own length"
              f" {(signed / asked_cm.sum()) / (shown / shown_cm.sum()):.2f}")

    print()
    print(f"3. How exact the measure is for a shape with {settings['episodes_per_new_shape']} episodes (whole-line episodes: standard starts)")
    randomness_generator = numpy.random.default_rng(settings["seed"])
    whole = table[table["is_path_row"] & (table["start_kind"] == "STD")]
    print("   sheet   whole-line episodes   path within 45 per episode: mean   spread between episodes   mean of 4 episodes drawn at random: 5th to 95th percentile")
    half_widths = []
    for sheet, group in whole.groupby("sheet"):
        per_episode = group.groupby("episode_id")["is_path_right"].mean().to_numpy()
        draws = numpy.array([randomness_generator.choice(per_episode, size = settings["episodes_per_new_shape"], replace = False).mean()
                             for _ in range(settings["draws"])])
        low, high = numpy.percentile(draws, [5, 95])
        half_widths.append((high - low) / 2)
        print(f"   {sheet[-2:]}      {len(per_episode):19d}   {per_episode.mean():32.3f}   {per_episode.std(ddof = 1):23.3f}   {low:.3f} to {high:.3f}")
    between = per_sheet_path.std(ddof = 1)
    print(f"   so one new shape with four episodes is read to about plus or minus {100 * numpy.mean(half_widths):.1f} points (90 % range), and the mean over 12 such"
          f" shapes to about plus or minus {100 * numpy.mean(half_widths) / numpy.sqrt(12):.1f} points from the episodes alone; the six sheets themselves differ by a"
          f" spread of {100 * between:.1f} points")



if __name__ == "__main__":
    offline_rule_report()
