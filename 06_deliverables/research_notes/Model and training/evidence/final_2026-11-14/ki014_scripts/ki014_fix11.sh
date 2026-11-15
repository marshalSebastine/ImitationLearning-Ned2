cd ~/ai_project/02_code_5 || exit 1
echo "--- before"; cat ../logs/final11_queue.log
if grep -q "job started" ../logs/final11_queue.log; then echo "a job has started already: the queue is left as it is"; exit 0; fi
pkill -u "$(id -un)" -f "sh final11_queue.sh"
sleep 2
sed -i 's/-ge 1800/-ge 1400/' final11_queue.sh
echo "limit now: $(grep -o 'ge 1[0-9]*' final11_queue.sh | head -1)"
(nohup sh final11_queue.sh > ../logs/final11_queue.log 2>&1 < /dev/null &)
sleep 20
echo "--- after"; cat ../logs/final11_queue.log
echo "card: $(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
