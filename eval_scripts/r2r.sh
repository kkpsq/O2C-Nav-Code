#!/bin/bash

export GLOG_minloglevel=0
export MAGNUM_LOG=verbose
export EGL_PLATFORM=surfaceless
unset DISPLAY

# ==========================================
# Language Action Model (LAM) 环境变量
# ==========================================
: "${LA_API_KEY:?Set LA_API_KEY before running evaluation}"
export LA_BASE_URL="${LA_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LA_MODEL_NAME="${LA_MODEL_NAME:-qwen3-vl-235b-a22b-instruct}"



export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "✅ 评估所需的环境变量已成功加载！"

# Generate timestamp for exp_name and log file
TIMESTAMP=$(date +"%m%d-%H%M%S")

# 指定要使用的两张物理显卡（假设你要用 6 和 7，根据你的实际机器情况修改）
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# 1. 加上了反斜杠 \ 保证换行时参数解析不断层
# 2. 将 TORCH_GPU_IDS 和 SIMULATOR_GPU_IDS 设置为 [0,1]

# flag="--exp_name 0518-013443 \
#       --run-type eval \
#       --exp-config vlnce_baselines/config/r2r.yaml \
#       --nprocesses 6 \
#       --resume \
#       NUM_ENVIRONMENTS 1 \
#       TRAINER_NAME o2cnav \
#       TORCH_GPU_IDS [0] \
#       SIMULATOR_GPU_IDS [0]"

flag="--exp_name ${TIMESTAMP} \
      --run-type eval \
      --exp-config vlnce_baselines/config/r2r.yaml \
      --nprocesses 6 \
      NUM_ENVIRONMENTS 1 \
      TRAINER_NAME o2cnav \
      TORCH_GPU_IDS [0] \
      SIMULATOR_GPU_IDS [0]"

echo "Starting experiment: ${TIMESTAMP}-api"
echo "Logging to: logs/${TIMESTAMP}.log"

setsid python run_mp.py $flag > logs/${TIMESTAMP}.log 2>&1 &

echo "Process started in background. PID: $!"
echo "To check logs: tail -f logs/${TIMESTAMP}.log"
