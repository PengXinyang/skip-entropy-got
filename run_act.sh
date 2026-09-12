#!/bin/bash
set -euo pipefail
source ~/.zshrc
proxyon

# ========== 1. 创建 venv ==========
if [ ! -d "venv" ]; then
    echo "[$(date)] 创建 venv ..."
    python -m venv venv
else
    echo "[$(date)] venv 已存在，跳过创建"
fi

# ========== 2. 激活 venv ==========
source venv/bin/activate

# ========== 3. 安装依赖 ==========
echo "[$(date)] 安装 requirements.txt ..."
pip install --upgrade pip
pip install -r requirements.txt
proxyoff

# ========== 4. 运行第一个实验 ==========
echo "[$(date)] 启动 parallel_batch_static_skip_experiment.py ..."
python examples/parallel_batch_static_skip_experiment.py \
  --model-name deepseek-32b-vllm \
  --tasks doc_merge \
  --num-shards 4 \
  --max-workers 4 \
  --skip-ratio 0.2 \
  --skip-orders low high \
  --entropy-field normalized_avg_entropy_bits \
  --resume-from examples/parallel_batch_static_skip_results \
  --copy-resumed

echo "[$(date)] 第一个实验结束，启动 judge ..."

# ========== 5. 运行 judge ==========
python examples/judge_full_got_experiment.py \
  --model-name deepseek-32b-vllm \
  --judge-model-name deepseek-32b-vllm \
  --tasks doc_merge \
  --input-run-root examples/parallel_batch_static_skip_results \
  --parallel-workers 4 \
  --skip-ratio 0.2 \
  --entropy-field normalized_avg_entropy_bits

echo "[$(date)] 全部完成"