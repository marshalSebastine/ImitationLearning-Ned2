"""Review attack 5: the two loops with stand-ins that deliver keys early, late or never, and readings that fail."""

import contextlib
import io
import math
import threading
import time

import cv2
import numpy

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS

recorded = [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]]


def raising(error):
    """A reading function that raises ``error``."""

    def read():
        raise error

    return read


def with_time_limit(name, action, seconds = 3.0):
    """Run ``action`` in a thread; say whether it came back within ``seconds`` (a loop that never ends is left running as a daemon)."""

    outcome = {}

    def target():
        printed = io.StringIO()
        try:
            outcome["result"] = action(printed)
        except BaseException as error:
            outcome["raised"] = f"{type(error).__name__}: {error}"
        outcome["printed"] = printed

    thread = threading.Thread(target = target, daemon = True)
    started = time.monotonic()
    thread.start()
    thread.join(seconds)
    took = time.monotonic() - started
    if thread.is_alive():
        print(f"{name}: STILL WAITING after {seconds} s")
    else:
        print(f"{name}: {outcome.get('result', outcome.get('raised'))!r} after {took:.2f} s")
    return outcome


class PrintTo:
    """model_mode prints with print(); send that into a buffer for one call (threads share sys.stdout, so swap the module's print)."""

    def __init__(self, buffer):

        self.buffer = buffer

    def __call__(self, *values, **options):

        options.pop("flush", None)
        options["file"] = self.buffer
        print(*values, **options)


def placing(read_joints, read_key, settings = None):

    def action(buffer):
        model_mode.print = PrintTo(buffer)
        return model_mode.placing_help(read_joints, read_key, settings)

    return action


def ghosting(read_picture, read_key, ghost, settings = None, show = None):

    def action(buffer):
        model_mode.print = PrintTo(buffer)
        return model_mode.go_ahead_with_ghost(read_picture, read_key, ghost, settings, show = show)

    return action


def last_line(outcome):

    text = outcome["printed"].getvalue() if "printed" in outcome else ""
    lines = [line for line in text.replace("\r", "\n").splitlines() if line.strip()]
    return lines[-1].strip()[:170] if lines else None


print("=== placing_help (real refresh 0.5 s, real poll 0.05 s unless said)")

# 1. Enter exactly once, at the first look (typed before the aid was on the screen): must not count
keys = iter(["enter"])
readings = []
outcome = with_time_limit("Enter once, before anything is shown", placing(lambda: readings.append(1) or recorded, lambda: next(keys, None)), 2.2)
print("   readings taken while waiting:", len(readings))


def key_at(delay, clock):

    def read_key():
        if clock["start"] is None:
            clock["start"] = time.monotonic()
            return None
        if not clock["given"] and time.monotonic() - clock["start"] >= delay:
            clock["given"] = True
            return "enter"
        return None

    return read_key


# 2. Enter once during the first reading (0.2 s after the first line): swallowed by the two-readings rule
readings = []
outcome = with_time_limit("Enter once 0.2 s after the aid began (first reading still on the line)", placing(lambda: readings.append(1) or recorded, key_at(0.2, {"start": None, "given": False})), 2.2)
print("   readings taken:", len(readings))

# 3. Enter once at 0.7 s: counts
readings_late = []
outcome = with_time_limit("Enter once 0.7 s after the aid began", placing(lambda: readings_late.append(1) or recorded, key_at(0.7, {"start": None, "given": False})), 3.0)
print("   readings taken:", len(readings_late))

# 4. never
readings_never = []
outcome = with_time_limit("no key ever", placing(lambda: readings_never.append(1) or recorded, lambda: None), 1.7)
print("   readings taken in 1.7 s:", len(readings_never))

# 5. readings that fail: at once, at the third reading, by returning nonsense
count = {"n": 0}


def third_fails():

    count["n"] += 1
    if count["n"] == 3:
        raise ConnectionError("the robot does not answer\nsecond line of the error")
    return recorded


for name, read_joints in (("reading fails at once", raising(RuntimeError("no answer from the robot"))),
                          ("reading fails at the third time", third_fails),
                          ("reading gives None", lambda: None),
                          ("reading gives three joints", lambda: [0.0, 0.0, 0.0]),
                          ("reading gives text", lambda: ["a"] * 6),
                          ("reading raises an error without a message", raising(RuntimeError()))):
    outcome = with_time_limit(name, placing(read_joints, lambda: None, {"place_help_refresh_s": 0.05}), 3.0)
    print("   last printed line:", repr(last_line(outcome)))

# 6. readings with not-a-number joints
outcome = with_time_limit("reading gives nan joints", placing(lambda: [float("nan")] * 6, lambda: "enter", {"place_help_refresh_s": 0.0}), 3.0)
print("   last shown:", repr(last_line(outcome)))

# 7. the key reader itself fails
outcome = with_time_limit("key reader raises OSError", placing(lambda: recorded, raising(OSError("no console"))), 3.0)
print("   printed:", repr(last_line(outcome)))
outcome = with_time_limit("key reader raises KeyboardInterrupt (Ctrl+C)", placing(lambda: recorded, raising(KeyboardInterrupt())), 3.0)
print("   printed ends with a line feed:", outcome["printed"].getvalue().endswith("\n"))

# 8. other answers of the key reader
for answer in ("escape", "ENTER", "\r", True, 13):
    outcome = with_time_limit(f"key reader always answers {answer!r}", placing(lambda: recorded, lambda: answer, {"place_help_refresh_s": 0.05}), 0.6)

# 9. settings at their edges
for settings in ({"place_help_refresh_s": 0.0, "place_help_key_poll_s": 0.0}, {"place_help_refresh_s": -1.0}, {"place_help_key_poll_s": 2.0, "place_help_refresh_s": 0.1}):
    looks = {"n": 0}

    def read_key():
        looks["n"] += 1
        return "enter" if looks["n"] >= 4 else None

    outcome = with_time_limit(f"settings {settings}", placing(lambda: recorded, read_key, settings), 6.0)

print()
print("=== go_ahead_with_ghost")
ghost = numpy.zeros((480, 640, 3), numpy.uint8)
good = cv2.imencode(".jpg", numpy.full((480, 640, 3), 180, numpy.uint8))[1].tobytes()

shown = []
outcome = with_time_limit("Enter at the first look", ghosting(lambda: good, lambda wait: "enter", ghost, show = shown.append))
print("   pictures shown before it started:", len(shown))
outcome = with_time_limit("Esc at the first look", ghosting(lambda: good, lambda wait: "escape", ghost, show = lambda picture: None))
shown_never = []
outcome = with_time_limit("no key ever (stand-in waits 10 ms)", ghosting(lambda: good, lambda wait: time.sleep(0.01), ghost, show = shown_never.append), 1.0)
print("   pictures shown in 1 s:", len(shown_never))
for name, read_picture in (("camera raises at once", raising(RuntimeError("camera stream lost"))),
                           ("camera gives empty bytes for ever", lambda: b""),
                           ("camera gives None", lambda: None),
                           ("camera gives text bytes for ever", lambda: b"not a picture")):
    looks = {"n": 0}

    def read_key(wait):
        looks["n"] += 1
        return "enter" if looks["n"] >= 5 else None

    shown_now = []
    outcome = with_time_limit(name, ghosting(read_picture, read_key, ghost, show = shown_now.append))
    print("   shown:", len(shown_now), "| printed:", [line for line in outcome["printed"].getvalue().splitlines() if "camera" in line])
count_pictures = {"n": 0}


def camera_fails_at_fourth():

    count_pictures["n"] += 1
    if count_pictures["n"] == 4:
        raise RuntimeError("camera stream lost")
    return good


shown_before = []
outcome = with_time_limit("camera raises at the fourth picture", ghosting(camera_fails_at_fourth, lambda wait: None, ghost, show = shown_before.append))
print("   shown before the error:", len(shown_before))
for answer in ("ENTER", "\r", "q", " ", 13, True):
    outcome = with_time_limit(f"window key always {answer!r}", ghosting(lambda: good, lambda wait: time.sleep(0.01) or answer, ghost, show = lambda picture: None), 0.3)
for settings in ({"ghost_refresh_s": 0.0}, {"ghost_refresh_s": -2}, {"ghost_refresh_s": 0.0004}, {"ghost_strength": 5}):
    waits = []

    def read_key(wait):
        waits.append(wait)
        return "enter" if len(waits) >= 2 else None

    outcome = with_time_limit(f"settings {settings}", ghosting(lambda: good, read_key, ghost, settings, show = lambda picture: None))
    print("   waits asked of the window, ms:", waits)
small = cv2.imencode(".jpg", numpy.full((240, 320, 3), 180, numpy.uint8))[1].tobytes()
shown_small = []
outcome = with_time_limit("a live picture smaller than the reference (320x240)", ghosting(lambda: small, lambda wait: "enter", ghost, show = shown_small.append))
print("   shown shape:", shown_small[0].shape if shown_small else None)
