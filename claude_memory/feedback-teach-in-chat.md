---
name: feedback-teach-in-chat
description: Emranul wants to be taught patiently in the chat itself, one lesson at a time, with no Markdown files made for the lessons
metadata:
  type: feedback
---

Emranul (10.11.26): "be patient with me while teaching me everything, don't generate any md file, just teach me here",
including how the data was cleaned and how the model architecture works.

**Why:** he calls himself a beginner and learns best from plain explanations he can ask about at once.
**How to apply:** teach in the chat, one small lesson at a time with everyday examples and a short check question, wait
for him before the next; never write lesson notes or explainer .md files unless he asks. See [[feedback-step-by-step]].

Lesson plan agreed on 10.11.26 (taught in the chat; done: 1). 1 big picture; 2 the data (sent once, to repeat); 3 how the
data was cleaned; 4 what the model answers, including which labels it trains on (exact step of 5 numbers plus stop, in
the picture frame; move names and table coordinates only for scoring and the chunk/backup models); 5 the architecture
(ResNet-18, 3 x 4 grid, head, outputs); 6 training; 7 testing and seeds; 8 augmentation; 9 how the code talks to the
robot (pyniryo, TCP port 40001, ROS, Ethernet 169.254.200.200 or hotspot 10.10.10.10, robot_link.py); 10 tools (VPN,
SSH, GPU pool, terminal); 11 next steps (final model, robot test). Also explain step within 45 vs path within 45.
Extra small topics to weave in (Emranul asked on 10.11.26 for everything he must know): what a model file (.pt
checkpoint) holds and that it is not code; training against inference (using the model) and the ms per picture;
overfitting against generalising, and why near-copy pictures make episode splits cheat (leakage); a baseline ("always
forward") and why every score needs one; loss, learning rate, batch, epoch in one picture; normalisation (ImageNet mean
and spread); class imbalance and the stop threshold (0.5, two in a row), precision and recall; offline scores against
closed-loop robot success, compounding errors in behaviour cloning and why DAgger corrections help; distribution shift
(new light, the taped camera moving); a recipe against a trained model (model 1, 2, ... exam model); the virtual
environment (.venv) and packages; git basics (branch, commit, push, why commits carry the students' names); the robot
safety rules (FreeMotion before touching, one program connected, motors and calibration).
