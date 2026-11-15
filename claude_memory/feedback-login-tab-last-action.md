---
name: feedback-login-tab-last-action
description: ki-014 login tab must be opened as the last action of a turn; the password prompt closes after 2 minutes and Emranul only sees it once the turn ends
metadata:
  node_type: memory
  type: feedback
  originSessionId: 997601d4-4a2e-4eb2-91d1-890ff53278fd
  modified: 2026-11-14T21:40:26.759Z
---

Open the ki-014 login tab (the ssh command that asks for Emranul's OTH password) as the LAST action of a turn, then end
the turn at once with the instruction to type the password. Do not keep working or wait on the terminal in the same turn.

**Why:** the server closes the login after about 2 minutes without a password. On 14.11.26 this happened twice (the
mid-way training and the final-model queue): Claude opened the tab and kept working or waited, Emranul did not see the
prompt in time, and he had to ask for it again ("timeout again pls").

**How to apply:** prepare everything first (scripts written, VPN checked with `Get-NetAdapter` and
`Test-NetConnection ki-014.oth-aw.de -Port 22`), open the tab, say "type the password now, once, then nothing more in
that tab", end the turn. Also give the launch command as a one-line block so he can start it himself. Never type or
store the password ([[project-gpu-pool-access]]). One login can upload, start a `nohup` queue and stream results back
(scripts of 14.11.26: `final.cmd`, `final_queue.sh`, `final_stream.sh`; see [[project-final-model-search]]).
