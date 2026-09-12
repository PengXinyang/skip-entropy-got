# Linux 服务器部署与本地 DeepSeek 32B 测试说明

本文档用于在 Linux 服务器上部署 `skip-entropy-got`，并使用本地
`DeepSeek-R1-Distill-Qwen-32B` 运行 `static_skip_experiment.py`。

当前静态 skip 实验入口：

```bash
examples/static_skip_experiment.py
```

该脚本目前测试的是 `sorting_032` 排序任务。

## 1. 准备服务器

先检查 GPU 和 CUDA 驱动：

```bash
nvidia-smi
```

建议使用：

```text
Python 3.10
CUDA GPU
conda / miniconda
较新的 NVIDIA Driver
```

DeepSeek 32B 即使使用 4bit 量化，也建议使用大显存 GPU 或多卡。第一次测试不要直接把
`max_new_tokens` 设太大。

## 2. 准备代码

如果代码在 Git 仓库中：

```bash
git clone <你的项目仓库地址> skip-entropy-got
cd skip-entropy-got
```

如果你直接上传项目目录，也需要保证目录结构完整：

```text
skip-entropy-got/
  examples/
  graph_of_thoughts/
  requirements.txt
```

## 3. 创建 conda 环境

```bash
conda create -n skip-entropy-got python=3.10 -y
conda activate skip-entropy-got
python -m pip install -U pip
```

## 4. 安装 PyTorch

不要直接依赖 `requirements.txt` 里的 `torch` 自动选择版本。服务器上应该根据 CUDA
版本安装对应的 PyTorch。

例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装后检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
PY
```

如果 `cuda available` 是 `False`，先不要继续跑 32B，本地模型会非常慢或直接失败。

## 5. 安装项目依赖

```bash
pip install -r requirements.txt
```

再确认本地 32B 关键依赖：

```bash
python - <<'PY'
import transformers, accelerate, bitsandbytes, safetensors, huggingface_hub
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes ok")
PY
```

如果 `bitsandbytes` 报 CUDA 相关错误，通常是 CUDA/PyTorch/bitsandbytes 版本不匹配，
需要重新按服务器 CUDA 版本安装 PyTorch。

## 6. 下载 DeepSeek 32B 模型

建议把模型下载到服务器本地磁盘，例如 `/data/models`。

```bash
pip install -U huggingface_hub
huggingface-cli login
mkdir -p /data/models
```

下载：

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --local-dir /data/models/DeepSeek-R1-Distill-Qwen-32B \
  --local-dir-use-symlinks False
```

如果下载时间长，建议使用 `tmux`：

```bash
tmux new -s download_deepseek
```

模型下载完成后目录应类似：

```text
/data/models/DeepSeek-R1-Distill-Qwen-32B/
  config.json
  tokenizer.json
  model-00001-of-xxxxx.safetensors
  ...
```

## 7. 配置本地模型

编辑：

```bash
vim graph_of_thoughts/language_models/config.json
```

加入模型配置。注意 JSON 最外层是一个对象，如果前面已有配置项，记得在上一项后面加逗号。

```json
"deepseek-32b-hf": {
  "model_id": "/data/models/DeepSeek-R1-Distill-Qwen-32B",
  "cache_dir": "/data/hf_cache",
  "prompt_token_cost": 0.0,
  "response_token_cost": 0.0,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "do_sample": true,
  "max_new_tokens": 512,
  "torch_dtype": "bfloat16",
  "device_map": "auto",
  "load_in_4bit": true,
  "force_single_response_calls": true
}
```

关键字段说明：

```text
model_id: 本地模型目录，也可以写 HuggingFace repo id。
cache_dir: HuggingFace 缓存目录。
max_new_tokens: 每次生成的最大 token 数。第一次建议 256 或 512。
load_in_4bit: 使用 4bit 量化加载，降低显存占用。
device_map: auto 表示让 accelerate 自动分配到 GPU。
force_single_response_calls: 统一为一个 thought/response 一次调用。
```

如果要计算完整词表熵，保持：

```json
"top_p": 1.0,
"top_k": 0
```

不要设置为：

```json
"top_p": 0.95,
"top_k": 50
```

否则生成分布可能被截断，得到的就不是完整词表熵。

## 8. 最小加载测试

先只测试模型能否加载，不跑完整 GoT：

```bash
python - <<'PY'
from graph_of_thoughts.language_models.factory import build_language_model

lm = build_language_model(
    "graph_of_thoughts/language_models/config.json",
    "deepseek-32b-hf",
    cache=False,
)
print(type(lm))
print(lm.model_id)
print("api_calls:", lm.api_calls)
PY
```

如果这里显存不足：

```text
1. 确认 load_in_4bit=true
2. 降低 max_new_tokens
3. 使用更多 GPU 或更大显存 GPU
4. 临时换 7B/14B 模型跑通流程
```

## 9. 最小生成与熵测试

```bash
python - <<'PY'
from graph_of_thoughts.language_models.factory import build_language_model

lm = build_language_model(
    "graph_of_thoughts/language_models/config.json",
    "deepseek-32b-hf",
    cache=False,
)
responses = lm.query("Sort this list: [3, 1, 2]", num_responses=1)
texts = lm.get_response_texts(responses)
metadata = lm.consume_last_response_metadata(1)[0]

print("text:", texts[0][:500])
print("entropy_estimate:", metadata.get("entropy_estimate"))
print("avg_entropy_bits:", metadata.get("avg_entropy_bits"))
print("vocab_size:", metadata.get("vocab_size"))
print("api_calls:", lm.api_calls)
PY
```

如果看到：

```text
entropy_estimate: full_vocab
```

说明使用的是本地完整词表熵，不是 API top-k 熵。

## 10. 运行静态 thought-level skip 实验

```bash
python examples/static_skip_experiment.py \
  --data-id 0 \
  --model-name deepseek-32b-hf \
  --skip-ratio 0.2 \
  --entropy-field avg_entropy_bits
```

第一次建议：

```text
skip_ratio: 0.1 或 0.2
max_new_tokens: 256 或 512
```

跑通后再逐步增大。

## 11. 查看输出结果

结果默认在：

```text
examples/static_skip_results/
```

每次运行会生成一个时间戳目录，里面包含：

```text
full_graph.json
compressed_graph.json
summary.json
candidate_ranking.json
candidate_ranking.csv
```

重点看：

```bash
cat examples/static_skip_results/<本次运行目录>/summary.json
```

主要指标：

```text
full.solved: 完整 GoT 是否解对。
compressed.solved: skip 后是否解对。
total_token_reduction: token 总量减少比例。
api_call_reduction: 调用次数减少比例。
latency_reduction: 总耗时减少比例。
```

查看可跳过节点排名：

```bash
head -n 20 examples/static_skip_results/<本次运行目录>/candidate_ranking.csv
```

`candidate_ranking.csv` 只记录可跳过节点，包括：

```text
generate / aggregate / improve 的 thought 节点
validate_and_improve 的 refine 节点
```

其中：

```text
rank: 熵从低到高的排名。
selected_for_skip: 本次是否被选中跳过。
node_label: 思维图里的节点位置。
ranking_score: 当前用于排序的熵值。
operation_index: GoT operation 位置。
thought_index: operation 内 thought 位置。
```

## 12. 验证完整词表熵

```bash
grep -n "entropy_estimate" examples/static_skip_results/<本次运行目录>/full_graph.json | head
```

应看到：

```json
"entropy_estimate": "full_vocab"
```

也可以查看候选排名：

```bash
grep -n "vocab_size" examples/static_skip_results/<本次运行目录>/full_graph.json | head
```

## 13. 常见问题

### 13.1 显存不足

优先尝试：

```text
max_new_tokens: 256
load_in_4bit: true
device_map: auto
```

如果仍然不行，先用较小模型验证代码流程。

### 13.2 bitsandbytes 报错

通常是 CUDA 版本、PyTorch 版本、bitsandbytes 版本不匹配。先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

然后重新安装匹配 CUDA 的 PyTorch。

### 13.3 模型下载失败

可以设置镜像或提前在其他机器下载后上传。上传后把 `model_id` 指向本地目录即可。

### 13.4 完整词表熵很慢

这是正常的。完整词表熵需要在每个生成 token 上保留并处理完整 vocabulary logits。
先用小数据、小 `max_new_tokens` 跑通，再扩大实验规模。
