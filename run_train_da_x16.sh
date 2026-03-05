#!/bin/bash
# 后台运行 x16 训练脚本，即使退出终端也不会停止

# 设置工作目录
cd /data/AI/cyj/TYSL

# 设置日志文件
LOG_FILE="./train_da13_x16.log"
PID_FILE="./train_da13_x16.pid"

# 使用 setsid 创建新的会话，完全脱离终端
# 这样可以避免 SIGHUP 信号的影响
setsid torchrun --nproc_per_node=4 --master_port=29510 \
  main_train_da.py \
  --opt ./options/train_da13_patch_muon_x16.json \
  --dist \
  > "$LOG_FILE" 2>&1 &

# 保存进程ID
echo $! > "$PID_FILE"

echo "x16 Training started in background"
echo "PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo ""
echo "To check status: tail -f $LOG_FILE"
echo "To stop training: kill \$(cat $PID_FILE)"


