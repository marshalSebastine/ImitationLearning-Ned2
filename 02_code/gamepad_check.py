"""
Find out which axis and button numbers the gamepad sends, to check the mapping in config (GAMEPAD_PARAMETERS).

Nothing connects to the robot. The script asks for one control at a time (push a stick, press a button), notes which
axis moved most or which button went down, and at the end prints the mapping it found next to the one in config.
Run from 02_code:
    ..\\.venv\\Scripts\\python.exe gamepad_check.py
"""

import time
from typing import Dict, List, Tuple

from config import GAMEPAD_PARAMETERS
from gamepad import GamepadReader

STICK_PROMPTS = [
    ("move_x", "LEFT stick fully to the RIGHT", +1),
    ("move_y", "LEFT stick fully UP", +1),
    ("twist", "RIGHT stick fully to the RIGHT", +1),
    ("lean", "RIGHT stick fully UP", +1),
    ("down", "L2 (left trigger) fully down", +1),
    ("up", "R2 (right trigger) fully down", +1),
]
BUTTON_PROMPTS = [
    ("stop", "CROSS (X, bottom)"),
    ("discard", "CIRCLE (right)"),
    ("mistake", "SQUARE (left)"),
    ("lost", "TRIANGLE (top)"),
    ("hand_left", "L1 (left shoulder)"),
    ("hand_right", "R1 (right shoulder)"),
    ("quit", "OPTIONS"),
]


def raw_axes(reader: GamepadReader) -> List[float]:

    reader.read()
    return [float(reader.pad.get_axis(index)) for index in range(reader.pad.get_numaxes())]


def raw_buttons(reader: GamepadReader) -> List[int]:

    reader.read()
    return [index for index in range(reader.pad.get_numbuttons()) if reader.pad.get_button(index)]


def find_axis(
        reader: GamepadReader,
        prompt: str
) -> Tuple[int, float]:
    """The axis that moved most from rest while the person holds the stick as asked, and its raw value."""

    rest = raw_axes(reader)
    print(f"  {prompt} and hold it, touch nothing else (waiting up to a minute)...")
    deadline = time.monotonic() + 60.0
    best_axis, best_value, best_change = -1, 0.0, 0.0
    while time.monotonic() < deadline:
        now = raw_axes(reader)
        for index, (before, value) in enumerate(zip(rest, now)):
            if abs(value - before) > best_change:
                best_axis, best_value, best_change = index, value, abs(value - before)
        if best_change > 0.8:
            break
        time.sleep(0.05)
    print(f"    -> axis {best_axis} went to {best_value:+.2f}; let go of the stick")
    time.sleep(1.5)
    return best_axis, best_value


def find_button(
        reader: GamepadReader,
        prompt: str
) -> int:
    """The button number that goes down while the person presses as asked."""

    while raw_buttons(reader):
        time.sleep(0.05)
    print(f"  press {prompt}...")
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        pressed = raw_buttons(reader)
        if pressed:
            print(f"    -> button {pressed[0]}")
            while raw_buttons(reader):
                time.sleep(0.05)
            return pressed[0]
        time.sleep(0.03)
    print("    -> nothing pressed")
    return -1


def gamepad_check() -> Dict:

    reader = GamepadReader()
    print(f"gamepad: {reader.name}, {reader.pad.get_numaxes()} axes, {reader.pad.get_numbuttons()} buttons")
    found = {"axes": {}, "inverted_axes": [], "buttons": {}}
    try:
        print("Sticks (let each stick go back to the middle between prompts):")
        for name, prompt, wanted_sign in STICK_PROMPTS:
            axis, value = find_axis(reader, prompt)
            found["axes"][name] = axis
            if value * wanted_sign < 0:
                found["inverted_axes"].append(name)
        print("Buttons:")
        for name, prompt in BUTTON_PROMPTS:
            found["buttons"][name] = find_button(reader, prompt)
    finally:
        reader.close()

    checked_axes = {name: GAMEPAD_PARAMETERS["axes"][name] for name in found["axes"]}
    checked_inverted = sorted(name for name in GAMEPAD_PARAMETERS["inverted_axes"] if name in found["axes"])
    print("\nfound:  ", found)
    print("config: ", {"axes": checked_axes, "inverted_axes": checked_inverted, "buttons": GAMEPAD_PARAMETERS["buttons"]})
    is_same = found["axes"] == checked_axes and sorted(found["inverted_axes"]) == checked_inverted and found["buttons"] == GAMEPAD_PARAMETERS["buttons"]
    print("MAPPING MATCHES CONFIG" if is_same else "MAPPING DIFFERS: copy the 'found' values into GAMEPAD_PARAMETERS in config.py")
    return found


if __name__ == "__main__":
    gamepad_check()
