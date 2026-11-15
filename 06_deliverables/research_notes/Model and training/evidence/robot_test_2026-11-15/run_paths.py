"""
How the robot moved in each run of the robot test of 15.11.26, from the runs' own step tables: steps, how often the step
turned back (more than 120 degrees from the step before, on the table), how often the rod tip came back within 1.5 cm
of a place it had passed more than 15 steps earlier (going round again), the stop chance at the end, and a drawing of
every run's tip path. Read with Emranul's marks (2026-11-15_robot_test_marks.csv).

Run from 02_code:  python "<this file>"
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import numpy
import pandas


RUN_PATH_PARAMETERS = {
    "run_folder": Path("D:/AI_Project/03_data/model_runs"),
    "run_prefix": "2026-11-15_1",
    "marks": Path(__file__).resolve().parents[2] / "2026-11-15_robot_test_marks.csv",
    "turn_back_deg": 120.0,
    "revisit_cm": 1.5,
    "revisit_gap_steps": 15,
    "output_table": Path(__file__).resolve().parent / "run_paths.csv",
    "output_figure": Path(__file__).resolve().parent / "run_paths.png",
}


def run_measures(steps: pandas.DataFrame, parameters: dict) -> dict:
    """Steps, turn-backs and revisits of one run, from its tip positions on the table."""

    tip = steps[["tip_x", "tip_y"]].to_numpy(dtype = float)
    moves = numpy.diff(tip, axis = 0)
    lengths = numpy.linalg.norm(moves, axis = 1)
    moves = moves[lengths > 1e-4]
    angles = numpy.degrees(numpy.arctan2(moves[:, 1], moves[:, 0]))
    change = numpy.abs((numpy.diff(angles) + 180.0) % 360.0 - 180.0)
    turn_backs = int((change > parameters["turn_back_deg"]).sum())
    revisits = 0
    gap = parameters["revisit_gap_steps"]
    for index in range(gap, len(tip)):
        distances = numpy.linalg.norm(tip[:index - gap] - tip[index], axis = 1)
        if len(distances) and distances.min() < parameters["revisit_cm"] / 100.0:
            revisits += 1
    stop = steps["model_stop_probability"].to_numpy(dtype = float)
    return {"steps": len(steps), "turn_backs": turn_backs, "revisited_steps": revisits,
            "path_cm": round(float(lengths.sum() * 100.0), 1), "stop_chance_last": round(float(stop[-1]), 2),
            "stop_chance_highest": round(float(stop.max()), 2), "end": str(steps["action"].iloc[-1])}


def main(parameters: dict = RUN_PATH_PARAMETERS) -> None:
    """Write the table and the drawing."""

    marks = pandas.read_csv(parameters["marks"]).drop_duplicates("run_folder", keep = "last").set_index("run_folder")
    rows, paths = [], []
    for folder in sorted(parameters["run_folder"].glob(parameters["run_prefix"] + "*")):
        if not (folder / "steps.csv").exists():
            continue
        steps = pandas.read_csv(folder / "steps.csv")
        line = json.loads((folder / "run_list_line.json").read_text(encoding = "utf-8"))["line"] if (folder / "run_list_line.json").exists() else {}
        row = {"run": folder.name, "label": folder.name.split("_")[-1], "model": line.get("model_key", ""), "sheet": line.get("sheet", "")[:40]}
        row.update(run_measures(steps, parameters))
        mark = marks.loc[folder.name] if folder.name in marks.index else None
        row["finished"] = "" if mark is None else mark["finished"]
        row["stop_mark"] = "" if mark is None else mark["stop"]
        rows.append(row)
        paths.append((row, steps[["tip_x", "tip_y"]].to_numpy(dtype = float)))
    table = pandas.DataFrame(rows)
    table.to_csv(parameters["output_table"], index = False)
    print(table.to_string(index = False))

    columns = 6
    figure, axes = matplotlib.pyplot.subplots(int(numpy.ceil(len(paths) / columns)), columns, figsize = (3.0 * columns, 3.0 * numpy.ceil(len(paths) / columns)))
    for axis, (row, tip) in zip(axes.flat, paths):
        axis.plot(tip[:, 1] * 100, tip[:, 0] * 100, "-", color = "tab:blue" if row["model"] == "15" else ("tab:orange" if row["model"] == "E" else "tab:green"), linewidth = 1)
        axis.plot(tip[0, 1] * 100, tip[0, 0] * 100, "go", markersize = 4)
        axis.plot(tip[-1, 1] * 100, tip[-1, 0] * 100, "rs", markersize = 4)
        axis.set_title(f"{row['label']} model {row['model']}: {row['steps']} steps\nback {row['turn_backs']}, again {row['revisited_steps']}, done {row['finished']}", fontsize = 8)
        axis.set_aspect("equal")
        axis.invert_xaxis()
        axis.tick_params(labelsize = 6)
    for axis in list(axes.flat)[len(paths):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(parameters["output_figure"], dpi = 80)


if __name__ == "__main__":
    main()
