"""在测试集上评测一个模型（任何 OpenAI 兼容接口：vLLM 部署的微调模型、原始模型或云端 API 模型）。

用法示例：
  # vLLM 部署的模型
  python run_eval.py --base-url http://localhost:8000/v1 --model sql-sft --name sft --no-thinking
  # 云端 API 模型（以兼容 OpenAI 协议的服务为例）
  python run_eval.py --base-url https://你的服务地址/v1 --api-key $KEY --model 模型名 --name api

结果写到 ../结果/<name>/：predictions_*.jsonl 和 metrics.json，并在终端打印汇总表。
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sql_check import default_db_dir, score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "数据"
DB_DIR = default_db_dir()
DB = {"ecommerce": DB_DIR / "ecommerce.db", "fmcg": DB_DIR / "fmcg.db"}


def call(client, args, item):
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None
    if args.extra_body:  # 例如 DeepSeek 关闭思考：--extra-body '{"thinking": {"type": "disabled"}}'
        extra = {**(extra or {}), **json.loads(args.extra_body)}
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=args.max_tokens,
                messages=[{"role": "system", "content": item["system"]}, {"role": "user", "content": item["prompt"]}],
                extra_body=extra)
            return resp.choices[0].message.content, time.time() - t0
        except Exception as e:  # 网络或限流，退避重试
            if attempt == 3:
                return f"__ERROR__ {e}", 0.0
            time.sleep(2 ** attempt)


def summarize(rows):
    sql_rows = [r for r in rows if r["gold_type"] == "sql"]
    clar = [r for r in rows if r["gold_type"] == "clarify"]
    ref = [r for r in rows if r["gold_type"] == "refuse"]
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None
    return dict(
        n=len(rows),
        execution_accuracy=pct(sum(r["correct"] for r in sql_rows), len(sql_rows)),
        first_try_executable=pct(sum(bool(r["executable"]) for r in sql_rows), len(sql_rows)),
        false_refusal=pct(sum(r["pred_type"] in ("clarify", "refuse") for r in sql_rows), len(sql_rows)),
        clarify_accuracy=pct(sum(r["correct"] for r in clar), len(clar)),
        refuse_accuracy=pct(sum(r["correct"] for r in ref), len(ref)),
        safety_violations=sum(r["safety"] in ("write", "restricted_table", "multi_statement") for r in rows),
        parse_failures=sum(not r["parsed"] for r in rows),
        n_sql=len(sql_rows), n_clarify=len(clar), n_refuse=len(ref),
        avg_latency_s=round(sum(r["latency"] for r in rows) / max(1, len(rows)), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="结果目录名，如 api / base / sft / grpo")
    ap.add_argument("--splits", default="test_in_domain,test_cross_domain")
    ap.add_argument("--limit", type=int, default=0, help="每个测试集只跑前 N 条，用于冒烟测试")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-thinking", action="store_true", help="关闭 Qwen 思考模式（vLLM 部署时使用）")
    ap.add_argument("--extra-body", default="", help="附加到请求体的 JSON，例如 DeepSeek 关闭思考：'{\"thinking\": {\"type\": \"disabled\"}}'")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    out_dir = HERE.parent / "结果" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split in args.splits.split(","):
        items = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
        if args.limit:
            items = items[: args.limit]
        with ThreadPoolExecutor(args.workers) as ex:
            outs = list(ex.map(lambda it: call(client, args, it), items))
        rows = []
        with open(out_dir / f"predictions_{split}.jsonl", "w", encoding="utf-8") as fh:
            for it, (text, lat) in zip(items, outs):
                r = score(it, text, DB[it["db"]])
                r.update(id=it["id"], kind=it["kind"], template_id=it["template_id"], question=it["question"],
                         output=text, latency=lat)
                rows.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics[split] = summarize(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n模型：{args.model}（{args.name}）")
    keys = ["n", "execution_accuracy", "first_try_executable", "false_refusal", "clarify_accuracy", "refuse_accuracy",
            "safety_violations", "parse_failures", "avg_latency_s"]
    print("指标".ljust(22) + "".join(s.ljust(20) for s in metrics))
    for k in keys:
        print(k.ljust(22) + "".join(str(metrics[s][k]).ljust(20) for s in metrics))


if __name__ == "__main__":
    main()
