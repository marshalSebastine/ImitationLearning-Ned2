"""
Verifier: how often would the proposed stall guard (6 decisions, net tip move under 1.0 cm AND net forearm turn under
10 degrees) fire on a healthy run? Estimated on the demonstrations' pictures (teacher forced, held-out sheets): the
fold models' own steps (length as asked, at most 1.2 cm) are chained along each demonstration, and stall EVENTS are
counted per episode (a run of stalled windows in a row is one event), also with the proposal's stop-chance exit.
The analyst gives the share of windows (0.35 and 0.57 %); a run has 100 to 300 windows, so the share per RUN matters.

Not a closed loop: on the robot the pictures follow the model's own moves. No training, nothing connects to the robot.

Run from this folder:  python v16_stall_guard_false_alarms.py > v16_stall_guard_false_alarms.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from config import TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_targets import load_training_frames

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
pandas.set_option("display.width", 250)

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)

for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2],
                                  "p_stop": data["stop_probability"]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    rows = []
    for episode_id, group in table.groupby("episode_id", sort = False):
        heading = numpy.radians(group["picture_heading_deg"].to_numpy())
        forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
        vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
        length = numpy.linalg.norm(vector, axis = 1)
        vector = vector * (numpy.minimum(length, 1.2) / numpy.maximum(length, 1e-9))[:, None]
        twist = numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
        stop = group["p_stop"].to_numpy()
        position = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])
        turn = numpy.r_[0.0, numpy.cumsum(twist)]
        if len(group) < 7:
            continue
        net_move = numpy.linalg.norm(position[6:] - position[:-6], axis = 1)
        net_turn = numpy.abs(turn[6:] - turn[:-6])
        stalled = (net_move < 1.0) & (net_turn < 10.0)
        # the window ending after decision i covers decisions i - 5 to i
        stop_in_window = numpy.array([stop[index:index + 6].max() >= 0.25 for index in range(len(stalled))])
        events = int((stalled & ~numpy.r_[False, stalled[:-1]]).sum())
        last_tenth = int(0.9 * len(stalled))
        rows.append({"episode_id": episode_id, "sheet": group["sheet"].iloc[0], "decisions": len(group), "stalled_windows": int(stalled.sum()), "events": events,
                     "events_before_the_last_tenth": int((stalled[:last_tenth] & ~numpy.r_[False, stalled[:last_tenth][:-1]]).sum()),
                     "events_with_stop_chance": int((stalled & stop_in_window & ~numpy.r_[False, (stalled & stop_in_window)[:-1]]).sum()),
                     "early_end": bool((stalled[:last_tenth] & stop_in_window[:last_tenth]).any())})
    episodes = pandas.DataFrame(rows)
    print(f"================ {recipe}: {len(episodes)} demonstrations, {int(episodes['decisions'].sum())} decisions")
    print(f"   stalled windows {100 * episodes['stalled_windows'].sum() / (episodes['decisions'] - 5).sum():.2f} % of the windows; demonstrations with at least one stall event "
          f"{100 * (episodes['events'] > 0).mean():.1f} %; events per 100 decisions {100 * episodes['events'].sum() / episodes['decisions'].sum():.2f}")
    print(f"   stall events before the last tenth of the demonstration (a nudge in the middle of a healthy run): demonstrations with at least one "
          f"{100 * (episodes['events_before_the_last_tenth'] > 0).mean():.1f} %")
    print(f"   'stalled near the end' (a stalled window with a stop chance of 0.25 or more) BEFORE the last tenth, which would end a healthy run early: "
          f"{100 * episodes['early_end'].mean():.1f} % of the demonstrations")
    print(episodes.groupby("sheet").agg(demonstrations = ("events", "size"), with_an_event_share = ("events", lambda values: (values > 0).mean()),
                                        with_an_event_before_the_last_tenth = ("events_before_the_last_tenth", lambda values: (values > 0).mean()),
                                        early_end_share = ("early_end", "mean")).round(2).to_string())
    print()
