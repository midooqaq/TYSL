cd /data/AI/cyj/TYSL

LOG_FILE="./test_da_x16.log"
PID_FILE="./test_da_x16.pid"

setsid python test_da_iterative.py \
  --opt options/test_da_x16.json \
  --iterations 1 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "x16 Test started in background"
echo "PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo ""
echo "To check status: tail -f $LOG_FILE"
echo "To stop test: kill \$(cat $PID_FILE)"


