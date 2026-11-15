---
name: feedback-limits-near-hardware
description: Emranul (11.11.26) wants software joint limits only about 2 to 3 degrees inside the robot's hardware limits, for recording and for the model mode
metadata:
  type: feedback
---

Emranul, 11.11.26: "don't limit anything next time we are collecting data, maybe 2-3 degrees to the actual hardware
limit". The loops need the forearm's full swing (demonstrations reached joint 4 = 119.35 against the arm model's 119.75).

**Why:** a tighter software limit (the model mode had 115, the keyboard turns 90) cuts off part of the demonstrated
manoeuvre on the loop sheets.
**How to apply:** set joint limits about 2 to 3 degrees inside the hardware's (`ned2_fk.LIM_DEG`): done on 11.11.26 for
joint 4 (`RECORDING_PARAMETERS` min/max_turn_deg and `MODEL_MODE_PARAMETERS["joint_4_most_deg"]` = 117, not committed).
Safety limits that exist for another reason (the wrist limit from the old bracket, the table edge) are not widened
without a test first ([[feedback-test-before-hardware-changes]]).
