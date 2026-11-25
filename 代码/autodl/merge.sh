export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/sqlft
date +%T
llamafactory-cli export 配置/merge_lora.yaml（远端用绝对路径版）
echo MERGE_EXIT $?
ls -la /root/autodl-tmp/models/qwen35-9b-sql-sft-merged
du -sh /root/autodl-tmp/models/qwen35-9b-sql-sft-merged
date +%T
