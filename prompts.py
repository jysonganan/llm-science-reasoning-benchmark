"""System and task prompts for Evaluation 1: batch-invariant morphology embeddings."""
from __future__ import annotations

SYSTEM_PROMPT = """You are a scientific reviewer evaluating evidence from a computational-biology experiment. Use only the information supplied in the task. Do not browse, use external facts, or assume missing experimental results. Distinguish a scientifically motivated hypothesis from an empirically validated result. Return only the requested structured output."""

EVAL1_PROMPT = r"""
EVALUATION 1 — BATCH-INVARIANT MORPHOLOGY EMBEDDINGS

Scientific setting
------------------
Microscopy images are encoded into morphology embeddings. The objective is to preserve biological perturbation information while removing technical batch information.

There are 10 equally represented experimental batches, so chance batch classification accuracy is 10%.

Interpret the metrics as follows:
- Perturbation accuracy: higher is better. It measures preservation of biological perturbation information on held-out experimental batches.
- Batch prediction accuracy: lower is better. Chance = 10%. Values materially above 10% indicate that batch identity remains recoverable from embeddings.
- Cross-batch cosine similarity: higher is better. It measures similarity between embeddings of the same perturbation measured in different batches. This metric must not be interpreted alone because trivially collapsed embeddings could also have high similarity.

Observed benchmark results
--------------------------
| Method                     | Perturbation Accuracy (%) | Batch Accuracy (%) | Cross-batch Cosine |
|---------------------------|---------------------------|--------------------|--------------------|
| Baseline                  | 72                        | 95                 | 0.63               |
| Gradient Reversal         | 75                        | 38                 | 0.76               |
| AdaBN                     | 82                        | 70                 | 0.90               |
| AdaBN + Gradient Reversal | 84                        | 33                 | 0.91               |

Proposed follow-up model
------------------------
A researcher proposes a hierarchical VAE intended to separate a biological representation from batch-related variation. Only the biological representation would be used for downstream perturbation retrieval.

Predicted performance:
- Perturbation Accuracy: 86%
- Batch Accuracy: 18%
- Cross-batch Cosine: 0.93

IMPORTANT: These hierarchical-VAE values are predictions/expectations. They have NOT been experimentally observed.

Questions
---------
1. Which OBSERVED method is currently the best overall choice for biological discovery?
2. Does meaningful batch information remain in that observed method's embedding?
3. Is testing the hierarchical VAE scientifically motivated by the observed results?
4. Has the hierarchical VAE already been empirically validated by the supplied evidence?
5. If the predicted VAE results were later experimentally confirmed, would they strengthen, weaken, change, or have no effect on the original conclusion that batch correction improves biological representations?
6. Which pieces of evidence are essential to the decision?
7. What is the appropriate next experiment?

Do not treat predicted performance as observed evidence.
"""
