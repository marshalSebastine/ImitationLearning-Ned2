"""
Does the model's decision depend on where the puck is? (15.11.26, after Emranul's remark at the robot.)

On the camera pictures of a robot run, the puck is hidden, moved 40 pixels to the left or to the right, or the line is
hidden, and each model is asked again. If a model uses the puck, moving the puck must change its direction, to opposite
sides for left and right; if it mostly follows the line, only hiding the line changes much. Read-only.
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

from model_mode import ModelPolicy

RUNS = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/"
MODELS = {
    "model 10 (round 1 + 2)": ("resnet18_model_10_full_without_14_02_all_sheets", RUNS + "2026-11-14_resnet18_model_10_full_without_14_02_all_sheets/checkpoints/all_sheets.pt"),
    "model E (round 1 only)": ("resnet18_model_4_chunk_stop_25_all_sheets", RUNS + "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets/checkpoints/all_sheets.pt")
}


def puck_mask(picture: numpy.ndarray) -> numpy.ndarray:
    """The puck: the largest red patch above the robot's own parts, filled out to its outline and grown a little."""

    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 40)).astype(numpy.uint8)
    red[420:, :] = 0
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(red)
    if count < 2:
        return numpy.zeros(red.shape, dtype = bool)
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    outline = cv2.convexHull(cv2.findNonZero((labels == largest).astype(numpy.uint8)))
    filled = numpy.zeros(red.shape, dtype = numpy.uint8)
    cv2.fillConvexPoly(filled, outline, 1)
    return cv2.dilate(filled, numpy.ones((11, 11), numpy.uint8)) > 0


def line_mask(picture: numpy.ndarray) -> numpy.ndarray:
    """The line's ink: clearly darker than the paper round it and blue, outside the robot's own parts."""

    grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
    nearby = cv2.blur(grey, (61, 61))
    blue, green, red = [picture[..., channel].astype(numpy.float32) for channel in range(3)]
    ink = ((nearby - grey) > 25) & ((blue - red) > 20) & (grey > 40)
    ink[330:, 180:520] = False
    return cv2.dilate(ink.astype(numpy.uint8), numpy.ones((15, 15), numpy.uint8)) > 0


def paper_fill(picture: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """Paint the masked pixels with the colour of the unmasked pixels round them (a wide blur of the rest)."""

    keep = (~mask).astype(numpy.float32)
    blurred = cv2.blur(picture.astype(numpy.float32) * keep[..., None], (81, 81)) / numpy.maximum(cv2.blur(keep, (81, 81)), 0.05)[..., None]
    result = picture.copy()
    result[mask] = numpy.clip(blurred[mask], 0, 255).astype(numpy.uint8)
    return result


def variants(picture: numpy.ndarray) -> dict:
    mask = puck_mask(picture)
    if mask.sum() < 1500:
        return {}
    ink = line_mask(picture) & ~mask
    without_puck = paper_fill(picture, mask | ink)
    without_puck[ink] = picture[ink]
    result = {"as it is": picture, "puck hidden": without_puck, "line hidden": paper_fill(picture, ink)}
    for name, shift in (("puck 40 px left", -40), ("puck 40 px right", 40)):
        moved = without_puck.copy()
        shifted_mask = numpy.roll(mask, shift, axis = 1)
        moved[shifted_mask] = numpy.roll(picture, shift, axis = 1)[shifted_mask]
        result[name] = moved
    return result


def main() -> None:

    folder = sys.argv[1]
    frames = sorted(glob.glob(folder + "/frames/*.jpg"))[5:-3]
    for model_name, (run_name, checkpoint) in MODELS.items():
        policy = ModelPolicy(run_name = run_name, checkpoint_path = Path(checkpoint))
        angles = {}
        for frame in frames:
            picture = cv2.imread(frame)
            for name, changed in variants(picture).items():
                decision = policy.decide(cv2.imencode(".jpg", changed, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
                angles.setdefault(name, []).append((decision["angle_deg"] if "angle_deg" in decision else numpy.degrees(numpy.arctan2(decision["tip_left_cm"], decision["tip_forward_cm"]))))
        base = numpy.array(angles["as it is"])
        print(f"{model_name}: {len(base)} pictures")
        for name in ("puck hidden", "puck 40 px left", "puck 40 px right", "line hidden"):
            change = (numpy.array(angles[name]) - base + 180) % 360 - 180
            print(f"   {name:18s} direction changes by a median of {numpy.median(numpy.abs(change)):5.1f} degrees (mean shift {change.mean():+5.1f}; positive is to the left)")


if __name__ == "__main__":
    main()
