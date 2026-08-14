"""System and task prompts for Evaluation 1: batch-invariant morphology embeddings (v1 phrasing)."""
from __future__ import annotations

SYSTEM_PROMPT = """You are a scientific reviewer evaluating evidence from a computational-biology experiment. Use only the information supplied in the task. Do not browse, use external facts, or assume missing experimental results. Distinguish a scientifically motivated hypothesis from an empirically validated result. Return only the requested structured output."""

EVAL1_PROMPT = r"""
You are evaluating morphology embeddings for microscopy images.

A benchmark compares four embedding methods.

The objective is to preserve biological information while removing batch effects.

The benchmark reports the following metrics:

- Perturbation accuracy (higher is better)
- Batch prediction accuracy (lower is better)
- Cross-batch cosine similarity for the same perturbation (higher is better)

A researcher proposes replacing AdaBN with a hierarchical VAE that explicitly models disentangled latent.

They argue this will better separate biological variation from technical variation.

Using only the results below, determine whether the proposal is scientifically justified.


Benchmark table

| Method                    | Perturbation Accuracy | Batch Accuracy | Cross-batch Cosine |
| ------------------------- | --------------------: | -------------: | -----------------: |
| Baseline                  |                    72 |             95 |               0.63 |
| Gradient Reversal         |                    75 |             38 |               0.76 |
| AdaBN                     |                    82 |             70 |               0.90 |
| AdaBN + Gradient Reversal |                    84 |             33 |               0.91 |

Hierarchical VAE results

| Metric                | Expected |
| --------------------- | -------: |
| Perturbation Accuracy |       86 |
| Batch Accuracy        |       18 |
| Cross-batch Cosine    |     0.93 |


Answer the following:

1. Which current method in the benchmark table is best?
2. Does the evidence suggest batch information still exists?
3. Is the hierarchical VAE scientifically motivated?
4. Would it strengthen, weaken, or change the paper's conclusion?
5. Explain why using only the provided evidence.

"""
