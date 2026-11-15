"""
Verifier: why the open-loop replay of the WHOLE log of run 5 through model_mode.model_step ends with a refused step
(an artefact of the open loop: the logged answers belong to the path the robot really took). On paper only.

Run from this folder:  python v09b_run5_refusal.py > v09b_run5_refusal.out.txt
"""

import contextlib
import io
import os

import pandas

with contextlib.redirect_stdout(io.StringIO()):
    import v09_rule_replay as replay_module

pandas.set_option("display.width", 250)
steps = pandas.read_csv(os.path.join(replay_module.RUNS, replay_module.RUN_FOLDERS[4], "steps.csv"))
table, ending, final_tip, final_joint_4 = replay_module.replay(steps, is_guarded = False)
print("run 5, open-loop replay of the whole log through model_step with the predicted length: ended", ending)
print(table.iloc[14:][["step", "asked_cm", "rule_cm", "twist_deg", "tip_x_cm", "tip_y_cm", "joint_4_deg", "heading_deg", "kind"]].round(2).to_string(index = False))
