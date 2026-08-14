"""Provider-specific runners that call each LLM and return its raw response."""
from __future__ import annotations
import json
import os

from prompts import SYSTEM_PROMPT, EVAL1_PROMPT
from schema import SCHEMA


def run_openai(model="gpt-5.6-sol"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=EVAL1_PROMPT,
        reasoning={"effort": "high"},
        max_output_tokens=1200,
        text={"format": {"type": "json_schema", "name": "eval1_output", "schema": SCHEMA, "strict": True}},
    )
    return {"provider": "openai", "model": model, "raw_text": r.output_text, "usage": getattr(r, "usage", None)}


def run_anthropic(model="claude-sonnet-5"):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    m = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": EVAL1_PROMPT}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    raw_text = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
    return {"provider": "anthropic", "model": model, "raw_text": raw_text, "usage": getattr(m, "usage", None)}


def run_gemini(model="gemini-3.1-pro-preview"):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=model,
        contents=EVAL1_PROMPT,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            temperature=0,
            max_output_tokens=1200,
        ),
    )
    return {"provider": "google", "model": model, "raw_text": r.text, "usage": getattr(r, "usage_metadata", None)}


def run_xai(model="grok-4.6"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    r = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=EVAL1_PROMPT,
        max_output_tokens=1200,
        text={"format": {"type": "json_schema", "name": "eval1_output", "schema": SCHEMA, "strict": True}},
    )
    return {"provider": "xai", "model": model, "raw_text": r.output_text, "usage": getattr(r, "usage", None)}


def run_deepseek(model="deepseek-v4-pro"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    system = SYSTEM_PROMPT + "\nReturn JSON matching this exact schema:\n" + json.dumps(SCHEMA)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": EVAL1_PROMPT}],
        thinking={"type": "enabled"},
        reasoning_effort="high",
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return {"provider": "deepseek", "model": model, "raw_text": r.choices[0].message.content, "usage": getattr(r, "usage", None)}


def run_together(model: str, provider_name: str):
    from together import Together
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    system = SYSTEM_PROMPT + "\nReturn only JSON matching this schema:\n" + json.dumps(SCHEMA)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": EVAL1_PROMPT}],
        max_tokens=1200,
        temperature=0,
        reasoning={"enabled": False},
        response_format={"type": "json_schema", "json_schema": {"name": "eval1_output", "schema": SCHEMA}},
    )
    return {"provider": provider_name, "model": model, "raw_text": r.choices[0].message.content, "usage": getattr(r, "usage", None)}


def run_llama():
    return run_together("meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta_via_together")


def run_qwen():
    return run_together("Qwen/Qwen3.5-9B", "qwen_via_together")


RUNNERS = [
    ("openai", run_openai, "OPENAI_API_KEY"),
    ("anthropic", run_anthropic, "ANTHROPIC_API_KEY"),
    ("google", run_gemini, "GEMINI_API_KEY"),
    ("xai", run_xai, "XAI_API_KEY"),
    ("deepseek", run_deepseek, "DEEPSEEK_API_KEY"),
    ("meta_via_together", run_llama, "TOGETHER_API_KEY"),
    ("qwen_via_together", run_qwen, "TOGETHER_API_KEY"),
]
