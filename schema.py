"""Structured output schema for Evaluation 1."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class Eval1Output(BaseModel):
    best_current_method: Literal["Baseline", "Gradient Reversal", "AdaBN", "AdaBN + Gradient Reversal"]
    batch_information_remaining: bool
    hierarchical_vae_scientifically_motivated: bool
    hierarchical_vae_empirically_validated: bool
    effect_if_predicted_results_confirmed: Literal["strengthen", "weaken", "change", "no_effect"]
    essential_evidence: list[Literal[
        "perturbation_accuracy",
        "batch_accuracy_vs_chance",
        "cross_batch_cosine",
        "predicted_vae_metrics_are_unobserved",
    ]]
    next_experiment: Literal[
        "compare_hierarchical_vae_vs_best_baseline_on_held_out_batches",
        "accept_predicted_vae_metrics_without_testing",
        "run_only_training_batch_evaluation",
        "ignore_batch_metrics_and_optimize_cosine_only",
    ]
    reasoning: str


SCHEMA = Eval1Output.model_json_schema()
