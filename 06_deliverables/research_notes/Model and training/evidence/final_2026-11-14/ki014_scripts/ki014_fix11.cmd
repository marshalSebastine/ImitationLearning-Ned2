@echo off
echo Fix of the eleventh queue: the free-memory limit 1800 MiB is too strict while the other training runs (1685 MiB free); this sets it to 1400 and restarts the queue before any job has started.
echo Type the password ONCE when "password:" appears and press Enter.
ssh e.hoque@oth-aw.de@ki-014.oth-aw.de "sh -s" < "C:\Users\e7dc\AppData\Local\Temp\claude\1808d3456aab7b9a\997601d4-4a2e-4eb2-91d1-890ff53278fd\scratchpad\final\ki014_fix11.sh"
echo CONNECTION ENDED, error level %ERRORLEVEL%.
