"""
Shared helpers of the independent verification of the view-novelty check (robot test of 11.11.26).

Written by the verifier, not by the analyst: nothing here imports the analyst's scripts. Nothing connects to the
robot: only saved pictures, saved logs and saved models are read.

Run any script of this folder with the project's Python from this folder:
    D:/AI_Project/.venv/Scripts/python.exe <script>.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

VERIFY_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "processed_folder": Path("D:/AI_Project/03_data/processed/v01_training_index"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "trained_models_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "own_cache_folder": Path(os.environ.get("VERIFY_CACHE", "C:/Users/e7dc/AppData/Local/Temp/claude/1808d3456aab7b9a/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/verify_view_novelty_cache")),
    "analyst_cache_folder": Path("C:/Users/e7dc/AppData/Local/Temp/claude/1808d3456aab7b9a/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/view_novelty_cache"),
    "threads": 2
}

RUN_FOLDERS = {
    1: "2026-11-11_172021_check_no_puck",
    2: "2026-11-11_172421_E_sheet01_run1",
    3: "2026-11-11_172743_E_sheet02_run1",
    4: "2026-11-11_173036_E_sheet02_run2",
    5: "2026-11-11_173549_E_sheet03_run1",
    6: "2026-11-11_174108_E_sheet04_run1",
    7: "2026-11-11_174448_E_sheet05_run1",
    8: "2026-11-11_174936_E_sheet06_run1",
    9: "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    10: "2026-11-11_175544_B_team5_sheet03"
}

RUN_SHEETS = {1: "path_01", 2: "path_01", 3: "path_02", 4: "path_02", 5: "path_03", 6: "path_04", 7: "path_05", 8: "path_06", 9: "path_03", 10: "path_03"}

RUN_MODELS = {1: "A", 2: "E", 3: "E", 4: "E", 5: "E", 6: "E", 7: "E", 8: "E", 9: "E", 10: "B"}



class Report:

    def __init__(self, output_path):
        """Collect printed lines and save them beside the script."""

        self.output_path = Path(output_path)
        self.lines = []


    def say(self, text = "") -> None:
        """Print ``text`` and keep it."""

        print(text, flush = True)
        self.lines.append(str(text))


    def save(self) -> None:
        """Write the kept lines as UTF-8 text."""

        self.output_path.write_text("\n".join(self.lines) + "\n", encoding = "utf-8")



def run_log(run_number: int) -> pandas.DataFrame:
    """Return steps.csv of one robot run."""

    return pandas.read_csv(VERIFY_PARAMETERS["model_runs_folder"] / RUN_FOLDERS[run_number] / "steps.csv")



def run_picture_path(run_number: int, step: int) -> Path:
    """Return the JPEG path of one robot picture."""

    return VERIFY_PARAMETERS["model_runs_folder"] / RUN_FOLDERS[run_number] / "frames" / f"{step:03d}.jpg"



def read_picture(path) -> numpy.ndarray:
    """Read a picture as OpenCV holds it (BGR); paths with spaces or umlauts are safe."""

    picture = cv2.imdecode(numpy.frombuffer(Path(path).read_bytes(), dtype = numpy.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        raise ValueError(f"{path} could not be decoded")
    return picture



def small_rgb(picture_bgr: numpy.ndarray) -> numpy.ndarray:
    """Shrink a 640 x 480 BGR picture to the 160 x 120 RGB form of the training cache (area averaging)."""

    return cv2.resize(cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA)



def training_frames() -> pandas.DataFrame:
    """Return frames.csv in ``frame_index`` order."""

    frames = pandas.read_csv(VERIFY_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    return frames.sort_values("frame_index").reset_index(drop = True)



def labelled_montage(pictures, labels, columns: int = 5, cell_width: int = 384) -> numpy.ndarray:
    """
    Lay BGR pictures out in a grid with a text label above each.

    Parameters
    ----------
    pictures : list of numpy.ndarray
        BGR pictures of any size (all are scaled to ``cell_width``).
    labels : list of str
        One label per picture.
    columns : int, optional
        Pictures per row.
    cell_width : int, optional
        Width of each cell in pixels.

    Returns
    -------
    numpy.ndarray
        The montage (BGR).
    """

    cells = []
    for picture, label in zip(pictures, labels):
        height = int(round(picture.shape[0] * cell_width / picture.shape[1]))
        cell = cv2.resize(picture, (cell_width, height), interpolation = cv2.INTER_AREA)
        bar = numpy.zeros((22, cell_width, 3), dtype = numpy.uint8)
        cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(numpy.vstack([bar, cell]))
    while len(cells) % columns:
        cells.append(numpy.full_like(cells[0], 255))
    rows = [numpy.hstack(cells[start:start + columns]) for start in range(0, len(cells), columns)]
    return numpy.vstack(rows)
