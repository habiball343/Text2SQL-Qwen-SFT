#!/bin/bash
# AutoDL 环境准备（2026-11-23 实测可用）。镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8，显卡 RTX 4090 24GB。
# 非交互 ssh 里 conda 不在 PATH 上，先补上。
export PATH=/root/miniconda3/bin:$PATH
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# 1. 训练环境：装在镜像自带的 base 环境
#    - 锁住 torch 2.8.0：LlamaFactory 依赖 torchaudio，不锁的话 pip 会装最新的 torchaudio 并把 torch 一起升级
#    - fla-core 0.5.0：Qwen3.5 的线性注意力层用它的 Triton 内核；0.6.0 需要更新的 Triton，和 torch 2.8 自带的 3.4 不兼容
#    - 不装 flash-linear-attention 0.6：它要求 transformers>=5.6.2，和 LlamaFactory 0.9.5 的 transformers<=5.6.0 冲突
pip install --timeout 120 --retries 10 "llamafactory[metrics]==0.9.5" "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
  "fla-core==0.5.0" bitsandbytes openai modelscope

# 2. 推理环境：vLLM 装在单独的虚拟环境（它要自己的 torch 版本）。实测装上 vllm 0.34.0 / torch 2.14.0+cu130
pip install uv
uv venv /root/vllm-env --python 3.12
. /root/vllm-env/bin/activate && UV_HTTP_TIMEOUT=300 uv pip install vllm --index-url $PIP_INDEX_URL
deactivate

# 3. 模型：ModelScope 下载，约 19GB
modelscope download --model Qwen/Qwen3.5-9B --local_dir /root/autodl-tmp/models/Qwen3.5-9B
