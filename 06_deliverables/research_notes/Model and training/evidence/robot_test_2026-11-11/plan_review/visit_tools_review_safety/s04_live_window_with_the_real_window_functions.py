"""
Safety review of the visit tools, check 4: the live go-ahead loop (go_ahead_with_ghost) with the REAL picture window
functions of the recording tool (show_in_window, read_window_key), which the builder tested with stand-ins only.

So that nothing appears on the screen, the window is made on a Windows desktop of its own that is never shown; a helper
thread posts key messages to the window, as a key pressed in it arrives. The camera is a stand-in. No robot.

It shows: how often a picture is fetched and shown, that other keys do not end the wait, that Enter and Esc do, what an
unreadable picture does, and what happens when no picture can ever be read (then there is no window at all). It cannot
show which window has the keyboard on the real desktop; that is as in the recording tool's own live loops.

Every flow runs in a process of its own, as every robot run does. Seen on 12.11.26 and not understood: from the third
time on that one and the same process made the picture window on this never-shown desktop (after it had been closed
twice), show_in_window took about 2 s instead of some hundredths (OpenCV 5.0.0; measured five times in a row, with and
without a key posted). A robot run makes its window once per process, a second time when the person closes it; only a
window closed twice in one run would meet this, and the arm then stands still for those 2 s. Noted, not followed
further.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s04_live_window_with_the_real_window_functions.py > s04_live_window_with_the_real_window_functions.out.txt
"""

import _thread
import contextlib
import ctypes
import io
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
WM_CHAR = 0x0102
FLOWS = ["enter", "escape", "unreadable", "never"]

if len(sys.argv) < 2:
    # the starter: one process per flow, each with its window on the desktop that is never shown
    print("every flow in a process of its own, its window on a desktop that is never shown")
    for flow in FLOWS:
        finished = subprocess.run([sys.executable, __file__, flow], capture_output = True, text = True, timeout = 120)
        print(finished.stdout.rstrip() or f"== {flow}: nothing was printed (exit code {finished.returncode}): {finished.stderr.strip()[-300:]}")
    sys.exit(0)

user32 = ctypes.WinDLL("user32", use_last_error = True)
user32.CreateDesktopW.restype = wintypes.HANDLE
user32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
user32.SetThreadDesktop.restype = wintypes.BOOL
user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

desktop = user32.CreateDesktopW("safety_review_desktop_never_shown", None, None, 0, 0x10000000, None)
if not desktop or not user32.SetThreadDesktop(desktop):
    print(f"== {sys.argv[1]}: the desktop that is never shown could not be made or entered (error {ctypes.get_last_error()}); nothing was shown and nothing was checked")
    sys.exit(0)

import cv2  # noqa: E402
import numpy  # noqa: E402

sys.path.insert(0, str(ROOT / "02_code"))
import robot_link  # noqa: E402


class RefusedRobotLink:
    """Stands in the real RobotLink's place, so that nothing here can connect to the robot."""

    def __init__(self, *arguments, **named_arguments):

        raise AssertionError("a connection to the robot was attempted")


robot_link.RobotLink = RefusedRobotLink

import model_mode  # noqa: E402
import record_episodes  # noqa: E402


def camera_jpeg(grey):
    """A camera-sized JPEG: one shade of grey with a dark line on it."""

    picture = numpy.full((480, 640, 3), grey, dtype = numpy.uint8)
    cv2.line(picture, (320, 470), (300, 100), (20, 20, 20), 12)
    return cv2.imencode(".jpg", picture)[1].tobytes()


def good(index):
    """The stand-in camera's n-th picture: a slowly changing grey."""

    return camera_jpeg(200 - 10 * (index % 5))


def one_flow(title, keys_plan, picture_at, interrupt_after_s = None):
    """
    The live loop with the real window: ``keys_plan`` lists (seconds, character code) posted to the window,
    ``picture_at`` gives the camera's answer to the n-th read, and ``interrupt_after_s`` raises what Ctrl+C raises.
    """

    reads, shown_at, notes = [], [], []
    real_show = record_episodes.show_in_window

    def counted_show(picture):

        shown_at.append(time.monotonic() - started)
        real_show(picture)

    def read_picture_bytes():

        reads.append(time.monotonic() - started)
        return picture_at(len(reads) - 1)

    def typist():

        user32.SetThreadDesktop(desktop)
        for at, code in keys_plan:
            time.sleep(max(0.0, at - (time.monotonic() - started)))
            window = user32.FindWindowW(None, record_episodes.WINDOW_NAME)
            is_posted = bool(window) and bool(user32.PostMessageW(window, WM_CHAR, code, 0))
            notes.append(f"key code {code} posted to the window at {time.monotonic() - started:.2f} s: {is_posted}")

    ghost = cv2.imdecode(numpy.frombuffer(camera_jpeg(90), numpy.uint8), cv2.IMREAD_COLOR)
    started, processor = time.monotonic(), time.process_time()
    thread = threading.Thread(target = typist, daemon = True)
    thread.start()
    if interrupt_after_s is not None:
        threading.Timer(interrupt_after_s, _thread.interrupt_main).start()
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            outcome = repr(model_mode.go_ahead_with_ghost(read_picture_bytes, record_episodes.read_window_key, ghost, show = counted_show))
    except BaseException as error:
        outcome = f"raised {type(error).__name__}"
    elapsed = time.monotonic() - started
    thread.join(timeout = 3)
    is_window_there = bool(user32.FindWindowW(None, record_episodes.WINDOW_NAME))
    print(f"== {title}")
    for note in notes:
        print("   " + note)
    print(f"   came back with {outcome} after {elapsed:.2f} s; the camera was read {len(reads)} times, {len(shown_at)} pictures were shown "
          f"({len(shown_at) / elapsed:.1f} a second), processor time {time.process_time() - processor:.2f} s; a picture window exists: {is_window_there}")
    told = [line for line in printed.getvalue().splitlines() if "could not be read" in line]
    if told:
        print(f"   the person was told {len(told)} time(s): {told[0]}")
    cv2.destroyAllWindows()


if sys.argv[1] == "enter":
    one_flow("Enter in the window after 2 s, two other keys before it", [(0.7, ord("x")), (1.2, ord(" ")), (2.0, 13)], good)
elif sys.argv[1] == "escape":
    one_flow("Esc in the window after 1.5 s", [(1.5, 27)], good)
elif sys.argv[1] == "unreadable":
    one_flow("the third and fourth pictures cannot be read, then Enter after 2 s", [(2.0, 13)], lambda index: b"not a picture" if index in (2, 3) else good(index))
else:
    one_flow("no picture can ever be read; Enter and Esc are tried, then it is ended from outside after 3 s, as Ctrl+C would end it", [(1.0, 13), (2.0, 27)],
             lambda index: b"not a picture", interrupt_after_s = 3.0)
