# Questions for the Maker Space (DC 1.04) before the camera mount is sent

Written 26.10.26, reworded the same day. Send by e-mail or ask in person; write the answers under each question with the
date. The design is not finalised until these are answered.

## English

Subject: AI Project, Team M (Project 6, Niryo robot): a few questions before we send our camera-mount file

Dear Maker Space team,

we are two students from the AI Project of Prof. Nierhoff (Project 6, Team M, the Niryo robot). The project allows us to have
a modified camera mount printed, and we understand that we have one print for it. Because we would like to get it right the
first time and not waste your material or your time, we would be very grateful if you could help us with a few questions
before we send anything:

1. Which material do you usually print with (PLA or PETG), and with which nozzle and layer height? We would like to design
   the part to suit your printer.
2. Would it be acceptable for one file (3MF or STL) to contain two small bodies, for example the mount and a spare part?
   If that is not possible, we will of course send a single part.
3. If, despite our care, a bore turns out about a millimetre off, would a small corrective reprint be possible, or should we
   plan for strictly one print? We fully understand if it is one.
4. Should the design include pockets for ordinary M3 nuts, or do you by any chance provide screws or heat-set inserts?
5. Roughly how long does it take from sending the file to the part being ready?
6. Should we make sure the file is in millimetres ourselves, or do you check the units on your side?
7. If it is not too much trouble, could we borrow a pair of calipers for about half an hour to measure the robot's wrist and
   the camera before we design the part?

Thank you very much for your time and for the support you give to the project. We are happy to come by in person if that is
easier.

Kind regards,
Emranul Hoque and Marshal Sebastine, Team M

## Deutsch

Betreff: AI Project, Team M (Projekt 6, Niryo-Roboter): ein paar Fragen vor dem Versand unserer Kamerahalterung

Sehr geehrtes Maker-Space-Team,

wir sind zwei Studierende aus dem AI Project von Prof. Nierhoff (Projekt 6, Team M, Niryo-Roboter). Im Projekt dürfen wir eine
geänderte Kamerahalterung drucken lassen, und wir verstehen es so, dass uns dafür ein Druck zur Verfügung steht. Da wir es
gern gleich beim ersten Mal richtig machen möchten und weder Ihr Material noch Ihre Zeit verschwenden wollen, wären wir sehr
dankbar, wenn Sie uns vorab bei ein paar Fragen helfen könnten:

1. Mit welchem Material drucken Sie üblicherweise (PLA oder PETG), und mit welcher Düse und Schichthöhe? Wir würden das
   Teil gern passend zu Ihrem Drucker auslegen.
2. Wäre es in Ordnung, wenn eine Datei (3MF oder STL) zwei kleine Körper enthält, zum Beispiel die Halterung und ein
   Ersatzteil? Falls nicht, schicken wir selbstverständlich nur ein Teil.
3. Falls trotz aller Sorgfalt eine Bohrung um etwa einen Millimeter daneben liegen sollte: wäre ein kleiner Korrekturdruck
   möglich, oder sollten wir fest mit nur einem Druck planen? Wir hätten dafür vollstes Verständnis.
4. Sollen wir im Teil Aufnahmen für normale M3-Muttern vorsehen, oder stellen Sie eventuell Schrauben oder
   Gewindeeinsätze bereit?
5. Wie lange dauert es ungefähr von der Datei bis zum fertigen Teil?
6. Sollen wir selbst sicherstellen, dass die Datei in Millimetern angelegt ist, oder prüfen Sie die Einheiten bei Ihnen?
7. Falls es keine Umstände macht: dürften wir uns für etwa eine halbe Stunde einen Messschieber ausleihen, um Handgelenk
   und Kamera des Roboters vor dem Entwurf zu vermessen?

Vielen Dank für Ihre Zeit und für die Unterstützung des Projekts. Wir kommen auch gern persönlich vorbei, wenn Ihnen das
lieber ist.

Mit freundlichen Grüßen
Emranul Hoque und Marshal Sebastine, Team M

## Answers

Received by e-mail on 27.10.26 (filed by the Robot and hardware session on Emranul's instruction). They write that Prof.
Nierhoff has already told them about the project and that they are happy to support it. Design decisions are ours; they help
with manufacturing questions.

1. **Material and parameters.** Normally PLA or PETG; ABS also available. Standard: 0.4 mm nozzle, 0.2 mm layer height
   (0.2, 0.6 and 0.8 mm nozzles may be available; 0.4 is enough for us). Standard Bambu Studio profiles, adjusted to the part.
2. **Several bodies in one file.** Yes, in one STL or 3MF, as long as the bodies do not overlap.
3. **Corrective reprint.** The design should be right before the first print, but a corrective reprint is generally not a
   problem. Avoid massive or oversized designs to keep material use reasonable.
4. **Screws, nuts, inserts.** Our choice. A limited number of heat-set inserts may exist in the lab but is not guaranteed.
   Specific screws, nuts or inserts: coordinate beforehand or provide them ourselves.
5. **Turnaround.** Printing normally starts 1 to 2 working days after the file arrives, depending on workload. Print time
   depends on the part; they send an estimate by e-mail once the file is prepared, and an e-mail when the part is ready.
6. **Units.** Export in millimetres at the correct scale. An STL stores no units, so the scale must be checked at export.
7. **Calipers.** Can be borrowed while the lab is staffed, usually 8:00 to 13:00; outside those hours by prior arrangement.
   Return them to the lab after use.

What this changes in the plan: see `06_deliverables/research_notes/Camera mount options/checklist_camera_mount.md`, steps 1,
2, 3, 7 and 9 (updated 27.10.26).
