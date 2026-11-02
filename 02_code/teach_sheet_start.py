"""
Teach only where a sheet's line begins, so the recording tool can start the puck there.

The same procedure as teach_positions.py, for one spot: one person holds the arm at the elbow, the motors go off,
the rod's tip is touched to the paper where the line BEGINS (where the puck's centre will be), Enter is pressed, and
the position is printed. Pass the two numbers to the recording tool as --start-x and --start-y. The result is also
written to 03_data/raw/workspace/<date>_positions.json. Run from this folder in a terminal, because it waits for Enter:
    ..\\.venv\\Scripts\\python.exe teach_sheet_start.py
"""

from teach_positions import teach_positions

SHEET_START_PARAMETERS = {
    "spots": [
        ("line_start", "the rod's tip touching the paper exactly where this sheet's black line BEGINS")
    ]
}


if __name__ == "__main__":
    positions = teach_positions(SHEET_START_PARAMETERS)
    x, y = positions["line_start"]["pose"][:2]
    print(f"\nStart the recording tool with:  --start-x {x:.4f} --start-y {y:.4f}")
