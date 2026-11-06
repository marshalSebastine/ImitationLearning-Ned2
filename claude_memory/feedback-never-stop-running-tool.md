---
name: feedback-never-stop-running-tool
description: Never stop a running robot tool or terminal command without first checking it is idle; stopping one mid-episode lost a recorded episode
metadata:
  type: feedback
---

Never stop (Ctrl-C, close the tab, kill the process) a recording tool or any command Emranul is using without first reading the terminal and making sure it is idle, for example waiting at the "Put the puck at the start" prompt. In early November Claude stopped a running tool mid-episode and the episode was lost; Emranul was upset.

**Why:** an episode is minutes of careful work at the robot, and a stopped tool also sends the robot home or leaves it mid-move.

**How to apply:** read the terminal tab first; if an episode is running, ask Emranul or wait until it is idle. Start new tools in a new tab instead of stopping someone's tab. Related: [[project-recording-tool]].
