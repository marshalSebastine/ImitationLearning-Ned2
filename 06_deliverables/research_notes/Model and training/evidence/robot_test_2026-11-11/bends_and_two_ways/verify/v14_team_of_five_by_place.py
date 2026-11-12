"""
Verifier's check of a gap in the comparison "chunk-trained models beat every exact-only seed at bends": the model the
robot can use instead of E is the TEAM of the five exact seeds (B), which the analyst did not score per kind of place.

The saved predictions hold each seed's direction only (no length), so the team is approximated here by the mean of the
five unit directions (the real team averages the five step vectors). Path within 45 degrees per kind of place.
Read-only. Prints to the screen and to v14_team_of_five_by_place.out.txt.
"""

from pathlib import Path

import numpy

import verify_common
from v07_saved_predictions import SAVED_PARAMETERS, by_place, saved_columns

if __name__ == "__main__":
    frames = verify_common.demonstration_rows()
    ahead = verify_common.tips_ahead(frames, 3)
    frames["path_deg"] = numpy.degrees(numpy.arctan2(ahead[:, 2, 1], ahead[:, 2, 0]))
    frames["path_cm"] = numpy.linalg.norm(ahead[:, 2, :], axis = 1)
    frames["is_scored"] = ~frames["is_stop"] & (frames["path_cm"] >= SAVED_PARAMETERS["path_minimum_cm"])
    total = numpy.zeros((len(frames), 2))
    lines = []
    for run in SAVED_PARAMETERS["exact_runs"]:
        angle = numpy.radians(saved_columns(SAVED_PARAMETERS["gpu_folder"] / run, "predictions_", frames, ["angle_deg"])["angle_deg"].to_numpy())
        total += numpy.stack([numpy.cos(angle), numpy.sin(angle)], axis = 1)
    team_deg = numpy.degrees(numpy.arctan2(total[:, 1], total[:, 0]))
    lines.append(f"team of the five exact seeds (mean of unit directions): {by_place(frames, team_deg)}")
    agreement = numpy.linalg.norm(total / len(SAVED_PARAMETERS["exact_runs"]), axis = 1)
    for run in ["2026-11-10_resnet18_chunk_stop_weight_25", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"]:
        saved = saved_columns(SAVED_PARAMETERS["gpu_folder"] / run, "predictions_", frames, ["angle_deg"])
        lines.append(f"{run[11:]} (first chunk step): {by_place(frames, saved['angle_deg'].to_numpy())}")
    scored = frames["is_scored"].to_numpy() & (frames["place"] != "no place").to_numpy()
    is_hit = verify_common.angle_between(team_deg, frames["path_deg"].to_numpy()) <= 45.0
    for low, high in [(0.0, 0.6), (0.6, 0.9), (0.9, 1.01)]:
        chosen = scored & (agreement >= low) & (agreement < high)
        lines.append(f"where the five seeds' agreement (length of the mean unit direction) is {low:.1f} to {high:.1f}: {int(chosen.sum())} pictures, path within 45 deg {float(is_hit[chosen].mean()):.3f}")
    print("\n".join(lines))
    (Path(__file__).resolve().parent / "v14_team_of_five_by_place.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
