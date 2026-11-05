"""
Show which axis and button numbers the gamepad sends, to check the mapping in config (GAMEPAD_PARAMETERS).

Nothing connects to the robot. Move each stick and press each button in turn; the line shows the pad's raw axis
values and the numbers of the buttons held, then how the recording tool reads them with the current mapping.
Ctrl-C ends it. Run from 02_code:
    ..\\.venv\\Scripts\\python.exe gamepad_check.py
"""

import time

from gamepad import GamepadReader


def gamepad_check() -> None:

    reader = GamepadReader()
    print(f"gamepad: {reader.name}, {reader.pad.get_numaxes()} axes, {reader.pad.get_numbuttons()} buttons. Ctrl-C to end.")
    try:
        while True:
            named = reader.read()
            axes = " ".join(f"a{index}={reader.pad.get_axis(index):+.2f}" for index in range(reader.pad.get_numaxes()))
            buttons = [index for index in range(reader.pad.get_numbuttons()) if reader.pad.get_button(index)]
            sticks = " ".join(f"{name} {named[name]:+.2f}" for name in ("move_x", "move_y", "twist", "lean"))
            print(f"\r{axes}  buttons {buttons}  ->  {sticks}  {sorted(named['buttons'])}          ", end = "", flush = True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.close()


if __name__ == "__main__":
    gamepad_check()
