## Robot time per trial (calculated from measured numbers)

Measured (code-reading audit, 17,863 replayed steps): a step takes 1.22 s median (the move 1.03 s, waiting for a fresh
picture 0.11 s, deciding 0.06 s). Steps per demonstrated episode, median by sheet: 01 31.5, 02 66, 03 106, 04 168,
05 103.5, 06 82 (all start kinds mixed). A model adds its own time per picture: 15 to 26 ms for ResNet-18 or
MobileNetV3 on 160 x 120 pictures (measured, 4 threads, laptop busy), under 3 percent of a step.

| Sheet | Steps (median) | Driving time at 1.22 s per step | With about 1 minute to lay the puck and start |
|---|---|---|---|
| 01 straight | 31.5 | 38 s | about 1.6 min |
| 02 arc | 66 | 81 s | about 2.4 min |
| 03 two curves | 106 | 129 s | about 3.2 min |
| 04 meander | 168 | 205 s | about 4.4 min |
| 05 loop | 103.5 | 126 s | about 3.1 min |
| 06 dashed loop | 82 | 100 s | about 2.7 min |

So 10 trials on each of the six sheets take about 3 hours of driving and set-up (calculated: 10 x the last column
summed, 174 minutes), plus resets after failures and breaks: plan 4 hours. 20 trials per sheet take about 6 to 8
hours. The guess of 1 minute per set-up is not measured.
