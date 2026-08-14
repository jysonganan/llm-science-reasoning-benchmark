# llm-science-reasoning-benchmark

Multi-model benchmark that tests whether LLMs can correctly reason about a scientific
evidence-vs-prediction scenario (batch-invariant morphology embeddings) and return a
structured, gradable answer.

The repo has two independent versions of the eval — different prompt wording and a
different output schema — kept in separate folders so they can be run and scored on
their own.

## Layout

- `eval_v1/` — original prompt/schema (7-field structured answer: best method, batch
  info, VAE motivation vs. validation, effect if predictions were confirmed, essential
  evidence, next experiment)
- `eval_v2/` — shorter prompt, 4-field structured answer (best method, batch info, VAE
  support, effect on conclusion)

Each version folder is self-contained:

- `prompts.py` — system prompt and the eval task prompt
- `schema.py` — pydantic output model and its JSON schema
- `model_runners.py` — one runner per provider (OpenAI, Anthropic, Gemini, xAI, DeepSeek, Together)
- `evaluation.py` — output validation, grading against gold answers, per-provider harness
- `main.py` — runs every provider with a configured API key and writes results into that folder

Both versions read API keys from a single `.env` at the repo root.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in the keys for the providers you want to run
```

## Run

Run each version separately:

```bash
python eval_v1/main.py
python eval_v2/main.py
```

Results are written next to each `main.py`: `eval_v1/eval1_results.jsonl` /
`eval_v1/eval1_results.csv`, and `eval_v2/eval2_results.jsonl` / `eval_v2/eval2_results.csv`.
Only providers whose API key is set in `.env` are run; others are skipped.
