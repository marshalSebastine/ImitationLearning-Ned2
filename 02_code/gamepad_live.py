"""
Show the gamepad's raw axis values and pressed button numbers live, one line each time something changes, to see
whether the pad sends anything.

Nothing connects to the robot. Ctrl-C ends it. Run from 02_code:
    ..\.venv\Scripts\python.exe gamepad_live.py
"""

import time

from gamepad import GamepadReader


def show_gamepad_live(poll_seconds: float = 0.1) -> None:
    """
    Print the raw value of every axis, the numbers of the buttons held down and the hat, whenever one changes.

    Parameters
    ----------
    poll_seconds : float
        Time between two readings.
    """

    reader = GamepadReader()
    print(f"gamepad: {reader.name}, {reader.pad.get_numaxes()} axes, {reader.pad.get_numbuttons()} buttons, "
          f"{reader.pad.get_numhats()} hats. Move a stick or press a button; Ctrl-C ends.")
    line_before = ""
    try:
        while True:
            reader.read()
            axes = " ".join(f"{index}:{reader.pad.get_axis(index):+.1f}" for index in range(reader.pad.get_numaxes()))
            buttons = [index for index in range(reader.pad.get_numbuttons()) if reader.pad.get_button(index)]
            hats = [reader.pad.get_hat(index) for index in range(reader.pad.get_numhats())]
            line = f"axes {axes} | buttons {buttons} | hat {hats}"
            if line != line_before:
                print(line, flush = True)
                line_before = line
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print()
    finally:
        reader.close()


if __name__ == "__main__":
    show_gamepad_live()
