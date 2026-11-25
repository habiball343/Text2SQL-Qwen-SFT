#!/bin/bash
# 目录约定：/root/autodl-tmp/sqlft/proj 放本项目的 代码/数据/配置，/root/autodl-tmp/sqlft/db 放两个数仓。
# --max-num-seqs 32：Qwen3.5 每条并发序列占一个 Mamba 状态块，默认 256 超出 24GB 显卡能分配的块数，vLLM 会启动失败。
# --limit-mm-per-prompt：纯文本任务，不给图片和视频预留显存。
# 用法：bash eval_model.sh <模型目录> <结果名>   在 vLLM 上部署模型并跑 run_eval.py，跑完关掉服务
MODEL=$1; NAME=$2; PORT=8000
W=/root/autodl-tmp/sqlft
export PATH=/root/miniconda3/bin:$PATH
/root/vllm-env/bin/vllm serve "$MODEL" --served-model-name "$NAME" --port $PORT --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --max-num-seqs 32 --limit-mm-per-prompt '{"image":0,"video":0}' > $W/vllm_$NAME.log 2>&1 &
VP=$!
for i in $(seq 1 180); do
  curl -s localhost:$PORT/v1/models > /dev/null && break
  if ! kill -0 $VP 2>/dev/null; then echo "VLLM_DIED"; tail -40 $W/vllm_$NAME.log; exit 1; fi
  sleep 5
done
echo "VLLM_READY $(date +%T)"
cd $W/proj/代码 && SQL_DB_DIR=$W/db python run_eval.py --base-url http://localhost:$PORT/v1 --model "$NAME" --name "$NAME" --no-thinking --workers 16 ${LIMIT:+--limit $LIMIT}
echo "EVAL_EXIT $?"
kill $VP; wait $VP 2>/dev/null
echo "EVAL_DONE_$NAME $(date +%T)"
