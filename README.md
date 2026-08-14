# llm-science-reasoning-benchmark

Multi-model benchmark that tests whether LLMs can correctly reason about a scientific
evidence-vs-prediction scenario (batch-invariant morphology embeddings) and return a
structured, gradable answer.

## Layout

- `prompts.py` — system prompt and the eval task prompt
- `schema.py` — `Eval1Output` pydantic model and its JSON schema
- `model_runners.py` — one runner per provider (OpenAI, Anthropic, Gemini, xAI, DeepSeek, Together)
- `evaluation.py` — output validation, grading against gold answers, per-provider harness
- `main.py` — runs every provider with a configured API key and writes results

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in the keys for the providers you want to run
```

## Run

```bash
python main.py
```

Results are written to `eval1_results.jsonl` (full detail) and `eval1_results.csv` (summary table).
Only providers whose API key is set in `.env` are run; others are skipped.
