@echo off
echo Fetch only, after the fifth login: gets every finished result from ki-014 and keeps fetching until the fifth queue has finished. Starts nothing new.
echo Type the password ONCE when "password:" appears, press Enter, and then type nothing more in this window.
ssh -o ServerAliveInterval=60 e.hoque@oth-aw.de@ki-014.oth-aw.de "cd ai_project && n=final5_stream && cp 02_code_5/$n.sh 02_code_5/final6_stream.sh && (pkill -u $(id -un) -f 'sh 02_code_5/'$n'.sh'; sleep 2; tail -n 3 logs/final3_queue.log logs/final5_queue.log 1>&2; sh 02_code_5/final6_stream.sh)" > "C:\Users\e7dc\AppData\Local\Temp\claude\1808d3456aab7b9a\997601d4-4a2e-4eb2-91d1-890ff53278fd\scratchpad\final\final6_results.tar"
echo CONNECTION ENDED, error level %ERRORLEVEL%.
