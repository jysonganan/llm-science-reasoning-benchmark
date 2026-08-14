"""Entry point: run Evaluation 1 (original prompt/schema) against every configured LLM provider."""
from __future__ import annotations
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR.parent / ".env")

from model_runners import RUNNERS
from evaluation import evaluate_provider


def main():
    results = []
    for name, runner, env_key in RUNNERS:
        if not os.getenv(env_key):
            print(f"SKIP {name}: {env_key} not set")
            continue
        print(f"Running {name} ...")
        r = evaluate_provider(name, runner)
        results.append(r)
        print(f"  model={r['model']} schema={r['schema_valid']} score={r['grade']['score']}/7 latency={r['latency_seconds']}s")

    with open(THIS_DIR / "eval1_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    rows = []
    for r in results:
        a = r["answer"] or {}
        rows.append({
            "provider": r["provider"], "model": r["model"], "schema_valid": r["schema_valid"],
            "score": r["grade"]["score"], "passed": r["grade"]["passed"],
            "best_current_method": a.get("best_current_method"),
            "batch_remaining": a.get("batch_information_remaining"),
            "vae_motivated": a.get("hierarchical_vae_scientifically_motivated"),
            "vae_validated": a.get("hierarchical_vae_empirically_validated"),
            "effect_if_confirmed": a.get("effect_if_predicted_results_confirmed"),
            "next_experiment": a.get("next_experiment"),
            "latency_seconds": r["latency_seconds"], "error": r["error"]
        })
    pd.DataFrame(rows).to_csv(THIS_DIR / "eval1_results.csv", index=False)
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
