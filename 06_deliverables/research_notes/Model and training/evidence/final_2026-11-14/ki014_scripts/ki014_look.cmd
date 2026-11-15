@echo off
echo Look only at ki-014: the card, who uses it, and how the tenth queue is doing. Starts and changes nothing.
echo Type the password ONCE when "password:" appears and press Enter.
ssh e.hoque@oth-aw.de@ki-014.oth-aw.de "sh -s" < "C:\Users\e7dc\AppData\Local\Temp\claude\1808d3456aab7b9a\997601d4-4a2e-4eb2-91d1-890ff53278fd\scratchpad\final\ki014_look.sh"
echo CONNECTION ENDED, error level %ERRORLEVEL%.
