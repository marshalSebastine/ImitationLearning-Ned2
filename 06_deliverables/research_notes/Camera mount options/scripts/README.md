# Analysis scripts behind the notes in this folder

Copied on 26.10.26 from the assistant's scratch folder. Where a note says
C:\Users\e7dc\AppData\Local\Temp\claude\D--AI-Project\...\scratchpad\<file>, read this folder instead.
Run with the system Python 3.13 (numpy, scipy, opencv, PIL, pymupdf, matplotlib, reportlab) from inside this folder, e.g.
    py -3.13 cam_options.py --fov 52 64.4
Order for a full re-run: fit_fk.py, cam_calibrate.py 52 104 64.4, cam_sheets.py, cam_options.py, cam_report.py,
cam_clearance.py, cam_e1_variants.py, cam_appendix.py; the review_*.py scripts check option E1 and F with the camera body.
Some scripts write their outputs next to themselves; outputs from 26.10.26 are included.
