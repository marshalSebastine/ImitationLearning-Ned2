"""
Are the robot's JPEGs the same kind of file as the training JPEGs (size, colour mode, quantisation tables)?
The verifier's own check on every robot picture and on every 25th training picture.

Run from this folder (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe own_file_format_check.py
Output: own_file_format_check.out.txt
"""

from collections import Counter

from PIL import Image

from verify_common import RUN_FOLDERS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, run_log, run_picture_path, training_frames



def file_kind(path) -> tuple:
    """Return (size, mode, sum of the luminance table, hash of all tables) of a JPEG."""

    with Image.open(path) as picture:
        tables = picture.quantization
        return picture.size, picture.mode, sum(tables[0]), hash(tuple(tuple(table) for _, table in sorted(tables.items())))



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_file_format_check.out.txt")
    robot = Counter(file_kind(run_picture_path(run_number, step)) for run_number in RUN_FOLDERS for step in range(len(run_log(run_number))))
    frames = training_frames()
    training = Counter(file_kind(VERIFY_PARAMETERS["raw_folder"] / frame_path) for frame_path in frames["frame_path"].iloc[::25])
    report.say(f"robot pictures: {sum(robot.values())} files, kinds (size, mode, luminance table sum): {[(kind[:3], count) for kind, count in robot.items()]}")
    report.say(f"training pictures: {sum(training.values())} files, kinds: {[(kind[:3], count) for kind, count in training.items()]}")
    report.say(f"same kinds of file in both (tables compared entry by entry through their hash): {set(robot) == set(training)}")
    report.save()



if __name__ == "__main__":
    main()
