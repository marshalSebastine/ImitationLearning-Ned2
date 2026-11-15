echo "=== $(hostname): $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do echo "   in use: $m by $(ps -o user= -p $p 2>/dev/null) for $(ps -o etime= -p $p 2>/dev/null)"; done
[ -d ~/ai_project ] && echo "   our project folder: there" || echo "   our project folder: not there"
