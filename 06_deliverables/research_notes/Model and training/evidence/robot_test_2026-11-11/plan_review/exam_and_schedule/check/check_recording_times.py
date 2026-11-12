"""
Check (key: exam_and_schedule_check) of the critic's unit times for new demonstrations.

Reads only: the episode timing table written by the earlier analysis. Nothing connects to the robot.
Run:  D:/AI_Project/.venv/Scripts/python.exe check_recording_times.py > check_recording_times.out.txt
"""

from pathlib import Path

import pandas

EPISODE_TABLE = Path(__file__).resolve().parents[4] / "design_review" / "improvement" / "episode_timing_episodes.csv"


def main() -> None:
    table = pandas.read_csv(EPISODE_TABLE)
    print("columns:", list(table.columns))
    table["plan"] = table["folder"].str.extract(r"_plan_(\d\d-\d\d)_")[0]
    table["date"] = table["folder"].str.slice(0, 10)
    kept = table[table["is_kept"]].sort_values("started_s")
    print("\n=== when each sheet's kept episodes were recorded, and its first five cycle times ===")
    for sheet, group in kept.groupby("sheet"):
        first = group.head(5)
        print(f"  {sheet}: dates {group['date'].value_counts().sort_index().to_dict()}; first five on {sorted(set(first['date']))}; "
              f"first five cycles (min) {[round(value, 1) for value in first['cycle_min']]}; operators of the first five {first['operator'].value_counts().to_dict()}")
    print("\n=== cycle per kept episode by date (learning the tool against a new shape) ===")
    for date, group in kept.groupby("date"):
        cycle = group["cycle_min"].dropna()
        print(f"  {date}: n {len(group)}, median {cycle.median():.2f}, mean {cycle.mean():.2f} min; sheets {group['sheet'].value_counts().sort_index().to_dict()}")
    print("\n=== 48 episodes on new shapes under four unit times ===")
    cycle = kept["cycle_min"].dropna()
    first_five_all = pandas.concat([group["cycle_min"].head(5) for _, group in kept.groupby("sheet")]).dropna()
    late_sheets = kept[kept["sheet"].isin(["path_05", "path_06"])]
    first_five_late = pandas.concat([group["cycle_min"].head(5) for _, group in late_sheets.groupby("sheet")]).dropna()
    for name, minutes in [("median cycle inside a sheet", cycle.median()), ("mean cycle", cycle.mean()),
                          ("first five of all six sheets, mean (the critic's 10.6)", first_five_all.mean()),
                          ("first five of all six sheets, median", first_five_all.median()),
                          ("first five of the two sheets recorded last (05, 06), mean", first_five_late.mean())]:
        print(f"  {name}: {minutes:.2f} min -> 48 episodes {48 * minutes / 60:.1f} h; plus 12 new tracks at the report's guess of 0.5 h each: {48 * minutes / 60 + 6:.1f} h")
    print("\n=== sittings: longest stretch at the robot ===")
    for sitting, group in table.groupby("sitting"):
        span = (group["finished_s"].max() - group["started_s"].min()) / 60
        print(f"  sitting {sitting}: {span / 60:.1f} h on {sorted(set(group['date']))}, kept {int(group['is_kept'].sum())}, operators {group[group['is_kept']]['operator'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
