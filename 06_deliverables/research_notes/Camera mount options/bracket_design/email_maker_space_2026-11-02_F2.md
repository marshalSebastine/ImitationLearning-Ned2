# E-mail to the Maker Space: corrective reprint F2 (draft, 2.11.26)

Attach: `TeamM_Project6_ned2_camera_bracket_F2.3mf` (and the `.step` as a fallback). Nothing else.

**Subject:** Team M (Project 6): corrective reprint of our camera bracket

---

Hi Julian,

thank you very much for printing our camera bracket last week. It fits the robot and the camera well. On the arm we
found one problem that was our mistake in the design: the post touches the robot's forearm when the wrist tilts up.
The attached version fixes it with a slightly leaning post, same material and settings as last time, about 78 x 34 x 90 mm.

Would it be possible to print it again? Whenever it fits your schedule is fine, and if it is not possible I fully
understand. Sorry for the extra work.

Kind regards
Emranul Hoque
Team M, AI Project (Project 6)

---

Deutsche Fassung:

Hallo Julian,

vielen Dank fürs Drucken unseres Kamerahalters letzte Woche. Er passt gut an den Roboter und an die Kamera. Am Arm hat
sich ein Problem gezeigt, das an unserem Entwurf lag: der Steg berührt den Unterarm des Roboters, wenn das Handgelenk
nach oben kippt. Die angehängte Version behebt das mit einem leicht geneigten Steg, gleiches Material und gleiche
Einstellungen wie beim letzten Mal, etwa 78 x 34 x 90 mm.

Wäre es möglich, das Teil noch einmal zu drucken? Wann immer es bei euch passt, und falls es nicht geht, habe ich
volles Verständnis. Entschuldigt die Extraarbeit.

Viele Grüße
Emranul Hoque
Team M, AI Project (Projekt 6)

## Sent

Sent by Emranul on 2.11.26 around midday, English version, 3MF and STEP attached. Reply: (none yet).

## Follow-up (2.11.26, afternoon): camera plate angle changed, sent as "current_version_F3"

Attach: the two files in `current_version_F3/` (the same part as F2 with the camera plate 30 degrees back instead of 22;
named F3 for Julian so the two e-mails cannot be confused).

**Subject:** Re: Team M (Project 6): corrective reprint of our camera bracket

Hi Julian,

please ignore my previous e-mail. Attached is the current version of the part, named current_version_F3; it replaces
the file from this morning. It is the same part, only the small plate at the top that holds the camera is angled a
little differently. Same size, material and settings as before.

Thank you again, and apologies for the back-and-forth.

Kind regards
Emranul Hoque

### Sent

Sent by Emranul on 2.11.26 in the afternoon. Reply from Julian on 3.11 (morning): the model has a gap at the top (plate not joined to the post) and cannot print; corrected file wanted by about 10:00 for a print by about 12:00.

## 3.11.26, morning: corrected file after Julian's screenshot

Cause: the wedge between the leaning post and the camera plate was drawn with three corners from the post's front face;
with the plate at 52 degrees it touched the plate only at one corner, so the slicer saw two bodies. Fix in
`bracket_f_design.py`: the wedge reaches 3 mm into the post and 1.5 mm into the plate (volume overlap), checked as one
solid for F, F2 and F3. Files in `current_version_F3/` regenerated 09:03.

**Subject:** Re: Team M (Project 6): corrective reprint of our camera bracket

Hi Julian,

thank you for catching that, and sorry. The top plate was not joined to the post in the file. Attached is the corrected
file, now one solid body; everything else is unchanged.

Kind regards
Emranul Hoque

### Sent

(not yet)
