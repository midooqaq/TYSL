cd /data/AI/cyj/TYSL

LOG_FILE="./test_da_x8.log"
PID_FILE="./test_da_x8.pid"

setsid torchrun --standalone --nproc_per_node=4 --rdzv_endpoint=127.0.0.1:29501 \
  test_da_iterative_patch_0.25.py \
  --opt options/test_da_x8_37.json \
  --dist \
  --iterations 1 \
  --patch_batch 64 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "Test started in background"
echo "PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo ""
echo "To check status: tail -f $LOG_FILE"
echo "To stop test: kill \$(cat $PID_FILE)"


