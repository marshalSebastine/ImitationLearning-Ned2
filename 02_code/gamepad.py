"""
Read a gamepad for the recording tool's gamepad mode, through pygame (SDL).

The named controls (move_x, move_y, twist, lean; stop, discard, mistake, lost, hand_left, hand_right, quit) are mapped
to the pad's axis and button numbers in config (GAMEPAD_PARAMETERS); gamepad_check.py prints the numbers of whatever
is pressed, so the mapping can be checked for the pad in use. The pad is read even while the picture window has the
keyboard focus.
"""

import os
from typing import Dict, Optional, Set

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame

from config import GAMEPAD_PARAMETERS


class GamepadReader:

    def __init__(self, gamepad_parameters: Optional[dict] = None):
        """
        Open the first gamepad found.

        Parameters
        ----------
        gamepad_parameters : dict, optional
            Overrides for GAMEPAD_PARAMETERS (axis and button numbers, dead zone, inverted axes).
        """

        self.gamepad_parameters = dict(GAMEPAD_PARAMETERS)
        if gamepad_parameters:
            self.gamepad_parameters.update(gamepad_parameters)
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no gamepad found: plug it in (or pair it) and start again")
        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()
        self.name = self.pad.get_name()


    def read(self) -> Dict:
        """
        The pad now: the four stick values in -1 to 1 (zero inside the dead zone; up and right positive for the
        sticks, so "move_y" up means towards the picture's top) and the set of named buttons held down.
        """

        pygame.event.pump()
        settings = self.gamepad_parameters
        values = {}
        for name, axis in settings["axes"].items():
            is_trigger = name in settings.get("trigger_axes", [])
            value = float(self.pad.get_axis(axis)) if axis < self.pad.get_numaxes() else (-1.0 if is_trigger else 0.0)
            if is_trigger:
                value = (value + 1.0) / 2.0
            if name in settings["inverted_axes"]:
                value = -value
            values[name] = 0.0 if abs(value) < settings["dead_zone"] else value
        held: Set[str] = {name for name, button in settings["buttons"].items()
                          if button < self.pad.get_numbuttons() and self.pad.get_button(button)}
        values["buttons"] = held
        return values


    def close(self) -> None:

        pygame.joystick.quit()
        pygame.quit()
