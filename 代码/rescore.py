"""用当前的 sql_check 重新给已有预测打分（不重新调用模型）。用法：python rescore.py 结果目录名 ..."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run_eval
from sql_check import score
for name in sys.argv[1:]:
    out = run_eval.HERE.parent / "结果" / name
    metrics = {}
    for split in ("test_in_domain", "test_cross_domain"):
        gold = {json.loads(l)["id"]: json.loads(l) for l in open(run_eval.DATA / f"{split}.jsonl", encoding="utf-8")}
        old = [json.loads(l) for l in open(out / f"predictions_{split}.jsonl", encoding="utf-8")]
        rows = []
        for o in old:
            it = gold[o["id"]]
            r = score(it, o["output"], run_eval.DB[it["db"]])
            r.update(id=o["id"], kind=o["kind"], template_id=o["template_id"], question=o["question"], output=o["output"], latency=o["latency"])
            rows.append(r)
        with open(out / f"predictions_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics[split] = run_eval.summarize(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, {s: (m["execution_accuracy"], m["first_try_executable"], m["false_refusal"], m["clarify_accuracy"], m["safety_violations"], m["parse_failures"]) for s, m in metrics.items()})
