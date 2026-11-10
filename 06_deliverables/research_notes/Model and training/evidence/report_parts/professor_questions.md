## Questions for the professor

None of these is on `05_meetings/requests_to_professor.md` yet; each changes what we build or how we test. Numbers in
brackets point to the questions already on that list.

1. **The robot's joints in the controller.** The model sees only the camera picture. To carry out its output (move the
   tip forward and left, twist the forearm, lean the rod), the controller must use the robot's own joint angles, as
   every move command does. May it? May it also hold the wrist and hand where they are (which keeps the camera's view
   like the training views)? Does the model's own previous output count as an input? (Asked in effect by the data
   audit; our first set-up needs a yes.)
2. **Correction rounds.** After the first model drives, we would stop it where it goes wrong, guide the arm by hand
   with the motors off and let the tool replay the correction, exactly like a demonstration. Does this count as our
   own demonstrations? (The 31.10 decisions note listed this for meeting 2; no answer is recorded.)
3. **More tracks.** Our measurement: spreading the same number of pictures over more different sheets helps more than
   more episodes of the same sheets, and the literature finds the same. When do the extra training tracks promised on
   26.10 arrive? May we print our own tracks of the same size and line thickness, for training and for testing (15)?
4. **What unseen tracks may contain.** Sharper corners, crossings near 90 degrees, line ends far from the paper's
   edge, close parallel lines, other dash rhythms? (Our six sheets show none of these; every line end in them lies
   2.35 to 8.03 cm from a paper edge, which a model may use as its stop cue.)
5. **The exam run.** How is "finished" judged (puck centre within what distance of the line's end)? When does the
   clock start and stop? How many attempts per track, and is there a time limit? Which table and which light (daylight
   appears only in sheets 02 and 04 of our data) (19)? How is the start set (18)?
6. **Where the model runs** (10): on this laptop, offline? Is the GPU pool usable for training (our VPN was not yet
   connected on 3.11)?
