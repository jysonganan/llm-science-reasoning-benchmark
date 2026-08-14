"""Structured output schema for Evaluation 2."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class Eval2Output(BaseModel):
    best_current_method: Literal["Baseline", "Gradient Reversal", "AdaBN", "AdaBN + Gradient Reversal"]
    batch_information_remaining: bool
    hierarchical_vae_supported: bool
    effect_on_conclusion: Literal["strengthen", "weaken"]
    reasoning: str


SCHEMA = Eval2Output.model_json_schema()
SCHEMA["additionalProperties"] = False