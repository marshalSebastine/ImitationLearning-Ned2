echo "--- card"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- programs on the card (pid, memory, user, running for, command)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do echo "$p $m $(ps -o user= -p $p 2>/dev/null) $(ps -o etime= -p $p 2>/dev/null) $(ps -o args= -p $p 2>/dev/null | cut -c1-90)"; done
echo "--- tenth queue log"; cat ~/ai_project/logs/final10_queue.log
echo "--- out of memory in the tenth queue's job logs"; grep -l "out of memory" $(ls -t ~/ai_project/logs/final_resnet18_* | head -6) 2>/dev/null
echo "--- newest job logs, last line"; for f in $(ls -t ~/ai_project/logs/final_resnet18_* | head -3); do echo "$(basename $f): $(tail -c 300 $f | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-160)"; done
echo "LOOK DONE"
