"""
Scratch helper, not project code: put key presses into THIS process's own console, as a keyboard would, through the
Windows call WriteConsoleInputW. Used inside a hidden console of its own, so nothing reaches any other window.
"""

import ctypes
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)


class KeyEventRecord(ctypes.Structure):

    _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD), ("wVirtualKeyCode", wintypes.WORD),
                ("wVirtualScanCode", wintypes.WORD), ("UnicodeChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]


class InputRecord(ctypes.Structure):

    _fields_ = [("EventType", wintypes.WORD), ("KeyEvent", KeyEventRecord)]


assert ctypes.sizeof(KeyEventRecord) == 16 and ctypes.sizeof(InputRecord) == 20

kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel32.WriteConsoleInputW.restype = wintypes.BOOL
kernel32.WriteConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(InputRecord), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

# name: (virtual key, scan code, character, control key state); 0x0100 marks an "enhanced" key such as an arrow
KEYS = {"enter": (0x0D, 0x1C, "\r", 0), "escape": (0x1B, 0x01, "\x1b", 0), "a": (0x41, 0x1E, "a", 0), "b": (0x42, 0x30, "b", 0),
        "up": (0x26, 0x48, "\x00", 0x0100), "f5": (0x74, 0x3F, "\x00", 0)}


def press(name: str, is_released: bool = True) -> None:
    """One key down (and up again) in this process's console."""

    virtual_key, scan_code, character, control_state = KEYS[name]
    handle = kernel32.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(f"this process has no console to type into (error {ctypes.get_last_error()})")
    states = (True, False) if is_released else (True,)
    records = (InputRecord * len(states))()
    for index, is_down in enumerate(states):
        records[index].EventType = 1
        records[index].KeyEvent.bKeyDown = is_down
        records[index].KeyEvent.wRepeatCount = 1
        records[index].KeyEvent.wVirtualKeyCode = virtual_key
        records[index].KeyEvent.wVirtualScanCode = scan_code
        records[index].KeyEvent.UnicodeChar = character
        records[index].KeyEvent.dwControlKeyState = control_state
    written = wintypes.DWORD(0)
    is_written = kernel32.WriteConsoleInputW(handle, records, len(states), ctypes.byref(written))
    kernel32.CloseHandle(handle)
    if not is_written or written.value != len(states):
        raise OSError(f"the key {name} could not be typed (error {ctypes.get_last_error()})")
