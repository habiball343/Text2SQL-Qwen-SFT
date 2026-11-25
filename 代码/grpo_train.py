"""在 SFT 合并后的模型上做 GRPO，奖励为 SQL 执行结果是否正确（规则见 sql_check.reward）。

说明：本脚本基于 TRL 的 GRPOTrainer 编写，没有在 GPU 上实际跑过。首次使用先用 --max-steps 5 冒烟，
确认显存、速度和奖励曲线正常再正式训练。TRL 版本差异可能导致参数名不同，以所装版本的文档为准。

用法：
  SQL_DB_DIR=../../Attribution-Agent/数据 python grpo_train.py --model 合并后的SFT模型目录 --output saves/grpo
"""
import argparse
import json
from pathlib import Path

from sql_check import default_db_dir, reward as sql_reward

HERE = Path(__file__).resolve().parent
DB_DIR = default_db_dir()
DB = {"ecommerce": str(DB_DIR / "ecommerce.db"), "fmcg": str(DB_DIR / "fmcg.db")}


def reward_fn(completions, db, gold_type, gold_result_json, **kwargs):
    """TRL 会把数据集中的其余列按名称传进来。completions 为对话格式时取最后一条消息内容。"""
    scores = []
    for comp, d, gt, gr in zip(completions, db, gold_type, gold_result_json):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        item = dict(db=d, gold_type=gt, gold_result=json.loads(gr))
        scores.append(sql_reward(item, text, DB[d]))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="saves/grpo")
    ap.add_argument("--data", default=str(HERE.parent / "数据" / "grpo_train.jsonl"))
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--use-vllm", action="store_true", help="用 vLLM 加速采样，需要额外显存")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda x: {"prompt": [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["prompt"]}]},
                remove_columns=["system"])
    ds = ds.shuffle(seed=42)

    cfg = GRPOConfig(
        output_dir=args.output,
        learning_rate=5e-6,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=1600,
        max_completion_length=512,
        temperature=0.9,
        beta=0.02,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        report_to="none",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_fn, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
