"""Grading, output validation, and per-provider evaluation harness."""
from __future__ import annotations
import json
import time
from typing import Any, Callable

from schema import Eval1Output

GOLD = {
    "best_current_method": "AdaBN + Gradient Reversal",
    "batch_information_remaining": True,
    "hierarchical_vae_scientifically_motivated": True,
    "hierarchical_vae_empirically_validated": False,
    "effect_if_predicted_results_confirmed": "strengthen",
    "next_experiment": "compare_hierarchical_vae_vs_best_baseline_on_held_out_batches",
}
REQUIRED_EVIDENCE = {
    "perturbation_accuracy",
    "batch_accuracy_vs_chance",
    "cross_batch_cosine",
    "predicted_vae_metrics_are_unobserved",
}


def grade_eval1(answer: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "best_current_method": answer.get("best_current_method") == GOLD["best_current_method"],
        "batch_information_remaining": answer.get("batch_information_remaining") is True,
        "hierarchical_vae_scientifically_motivated": answer.get("hierarchical_vae_scientifically_motivated") is True,
        "hierarchical_vae_empirically_validated": answer.get("hierarchical_vae_empirically_validated") is False,
        "effect_if_predicted_results_confirmed": answer.get("effect_if_predicted_results_confirmed") == "strengthen",
        "essential_evidence": REQUIRED_EVIDENCE.issubset(set(answer.get("essential_evidence", []))),
        "next_experiment": answer.get("next_experiment") == GOLD["next_experiment"],
    }
    score = sum(checks.values())
    return {"score": score, "max_score": len(checks), "passed": score == len(checks), "checks": checks}


def validate_output(raw: Any):
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        parsed = Eval1Output.model_validate(raw)
        return parsed.model_dump(), None
    except Exception as e:
        return None, str(e)


def evaluate_provider(name: str, runner: Callable[[], dict[str, Any]]):
    start = time.time()
    try:
        result = runner()
        answer, schema_error = validate_output(result["raw_text"])
        grade = grade_eval1(answer) if answer else {"score": 0, "max_score": 7, "passed": False, "checks": {}}
        return {
            "runner": name, "provider": result["provider"], "model": result["model"],
            "latency_seconds": round(time.time() - start, 3),
            "schema_valid": answer is not None, "schema_error": schema_error,
            "answer": answer, "raw_text": result["raw_text"], "grade": grade, "error": None
        }
    except Exception as e:
        return {
            "runner": name, "provider": name, "model": None, "latency_seconds": round(time.time() - start, 3),
            "schema_valid": False, "schema_error": None, "answer": None, "raw_text": None,
            "grade": {"score": 0, "max_score": 7, "passed": False, "checks": {}},
            "error": repr(e)
        }
