"""Review attack 4: the blend with pictures of other sizes and kinds, and reference files of every sort."""

import tempfile
import traceback
from pathlib import Path

import cv2
import numpy

import safe_import
from safe_import import model_mode

live = numpy.full((480, 640, 3), 200, dtype = numpy.uint8)
live[100:200, 300:400] = (10, 20, 30)


def attempt(name, action):

    try:
        result = action()
        described = f"shape {result.shape}, dtype {result.dtype}, corner {result[0, 0].tolist() if result.ndim == 3 else result[0, 0]}" if isinstance(result, numpy.ndarray) else repr(result)
        print(f"{name}: {described}")
        return result
    except Exception as error:
        print(f"{name}: RAISED {type(error).__name__}: {str(error).splitlines()[0][:150]}")
        return None


print("--- ghost_blend")
attempt("same size, strength 0.35", lambda: model_mode.ghost_blend(live, numpy.zeros((480, 640, 3), numpy.uint8), 0.35))
attempt("smaller ghost 240x320", lambda: model_mode.ghost_blend(live, numpy.zeros((240, 320, 3), numpy.uint8), 0.35))
attempt("larger ghost 1080x1920", lambda: model_mode.ghost_blend(live, numpy.zeros((1080, 1920, 3), numpy.uint8), 0.35))
attempt("other shape 100x900 (stretched)", lambda: model_mode.ghost_blend(live, numpy.zeros((100, 900, 3), numpy.uint8), 0.35))
attempt("1x1 ghost", lambda: model_mode.ghost_blend(live, numpy.full((1, 1, 3), 100, numpy.uint8), 0.5))
attempt("grey ghost as a 2-D array, same size", lambda: model_mode.ghost_blend(live, numpy.zeros((480, 640), numpy.uint8), 0.35))
attempt("grey ghost as a 2-D array, other size", lambda: model_mode.ghost_blend(live, numpy.zeros((240, 320), numpy.uint8), 0.35))
attempt("ghost with 4 channels", lambda: model_mode.ghost_blend(live, numpy.zeros((480, 640, 4), numpy.uint8), 0.35))
attempt("float ghost", lambda: model_mode.ghost_blend(live, numpy.zeros((480, 640, 3), numpy.float32), 0.35))
attempt("grey live as 2-D", lambda: model_mode.ghost_blend(numpy.zeros((480, 640), numpy.uint8), numpy.zeros((480, 640, 3), numpy.uint8), 0.35))
for strength in (-1.0, 0.0, 1.0, 2.5, float("nan"), "0.35", None):
    attempt(f"strength {strength!r}", lambda: model_mode.ghost_blend(live, numpy.zeros((480, 640, 3), numpy.uint8), strength))
before = live.copy()
ghost = numpy.full((480, 640, 3), 7, numpy.uint8)
ghost_before = ghost.copy()
result = model_mode.ghost_blend(live, ghost, 0.35)
print("live and ghost left untouched:", numpy.array_equal(live, before), numpy.array_equal(ghost, ghost_before), "| result shares memory with live:", numpy.shares_memory(result, live))
print("strength 0 returns a copy, not the live picture itself:", model_mode.ghost_blend(live, ghost, 0.0) is not live)

print("--- reference_picture on files of every sort")
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    cv2.imwrite(str(folder / "colour.jpg"), live)
    cv2.imwrite(str(folder / "grey.jpg"), numpy.full((480, 640), 128, numpy.uint8))
    cv2.imwrite(str(folder / "alpha.png"), numpy.full((480, 640, 4), 128, numpy.uint8))
    cv2.imwrite(str(folder / "sixteen_bit.png"), numpy.full((480, 640), 40_000, numpy.uint16))
    cv2.imwrite(str(folder / "tiny.jpg"), numpy.full((2, 2, 3), 128, numpy.uint8))
    (folder / "cut_short.jpg").write_bytes((folder / "colour.jpg").read_bytes()[:400])
    (folder / "half.jpg").write_bytes((folder / "colour.jpg").read_bytes()[:len((folder / "colour.jpg").read_bytes()) // 2])
    (folder / "text.jpg").write_text("not a picture", encoding = "utf-8")
    (folder / "empty.jpg").write_bytes(b"")
    (folder / "with space in name.jpg").write_bytes((folder / "colour.jpg").read_bytes())
    (folder / "a_folder.jpg").mkdir()
    for name in ("colour.jpg", "grey.jpg", "alpha.png", "sixteen_bit.png", "tiny.jpg", "cut_short.jpg", "half.jpg", "text.jpg", "empty.jpg", "with space in name.jpg",
                 "a_folder.jpg", "missing.jpg"):
        picture = attempt(name, lambda: model_mode.reference_picture(folder / name))
        if picture is not None:
            attempt(f"   blended with the live picture", lambda: model_mode.ghost_blend(live, picture, 0.35))
    attempt("None as the path", lambda: model_mode.reference_picture(None))
    attempt("'' as the path", lambda: model_mode.reference_picture(""))
    attempt("a number as the path", lambda: model_mode.reference_picture(5))
    attempt("a path with a character Windows forbids", lambda: model_mode.reference_picture("C:/x/<nul>|?.jpg"))
    attempt("a very long path", lambda: model_mode.reference_picture("C:/" + "x" * 400 + ".jpg"))
    attempt("NUL character in the path", lambda: model_mode.reference_picture("a\x00b.jpg"))
