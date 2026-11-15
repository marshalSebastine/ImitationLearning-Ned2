@echo off
echo Seventh login: uploads the new training code into its own folder on ki-014, queues the combined recipes (robot models 15 and 16 first) and fetches ALL results so far.
echo Type the password ONCE when "password:" appears, press Enter, and then type nothing more in this window.
set W=D:\AI_Project\.claude\worktrees\training-analysis\02_code
set S=C:\Users\e7dc\AppData\Local\Temp\claude\1808d3456aab7b9a\997601d4-4a2e-4eb2-91d1-890ff53278fd\scratchpad\final
for %%f in (config.py train_policy_emranul.py training_targets.py training_probes.py training_metrics.py) do copy /Y "%W%\%%f" "%S%\fifth\02_code_5\" > nul
tar -cf - -C "%S%\fifth" 02_code_5 | ssh -o ServerAliveInterval=60 e.hoque@oth-aw.de@ki-014.oth-aw.de "cd ai_project && tar -xf - && sed -i 's/\r$//' 02_code_5/*.sh && echo upload done 1>&2 && (cd 02_code_5 && sh final7_start.sh) 1>&2 && sleep 5 && sh 02_code_5/final7_stream.sh" > "%S%\final7_results.tar"
echo CONNECTION ENDED, error level %ERRORLEVEL%. If "upload done" and "jobs in the seventh queue" were shown above, the queue on ki-014 keeps running by itself.
