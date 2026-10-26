---
name: feedback-test-before-hardware-changes
description: "Never advise a physical change to the robot on a theory; test first with pictures or numbers, say what is known versus guessed, and never let Emranul spend hours on an unverified idea"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dd705f28-62f1-4a95-afe4-b6bd1767c0dc
  modified: 2026-10-25T17:48:41.252Z
---

On 25.10.26 Emranul lost three to four hours re-mounting the Niryo camera (tape on the wrist fork, then on the rod) because I diagnosed the "camera loses the puck" problem as a mounting-place problem. The real fix was the magnet block on the rod, which lifts the wrist; the camera's original screwed bracket was fine. Emranul said: "you should have been really careful, please assess everything from now on, I don't want to compromise on anything."

**Why:** Emranul is a beginner who follows my instructions literally and physically; a wrong theory costs them hours at the robot and trust.

**How to apply:** before suggesting any physical change (mounts, tools, sheet moves), run the cheapest test that would show whether the change is needed (a picture, a scripted push with pictures, a number), and prefer software or settings changes to hardware ones. State explicitly what is measured and what is a guess. When a test contradicts a theory, say so at once and drop it. Keep the original hardware configuration recoverable (screws, brackets kept). See [[ai-project-workspace-setup]] and [[feedback-step-by-step]].

Second instance, same day: my note "camera on the forearm, tilt changes 55 degrees, puck lost after 5 cm" was computed from the arm geometry and never checked against the pictures, which showed a constant view; the camera is on the wrist link. Measure from pictures on disk before writing any hardware fact ([[ai-project-workspace-setup]]).

Third instance (26.10): a collar camera mount I had drawn from a lens-point model was refuted when reviewers checked the camera's body against Niryo's meshes (the forearm fork). Emranul gets one 3D print only. Lesson: model the full body against the real meshes and run an adversarial review before any part is designed; never send a drawing of a part before that.
