import os
import json
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("NIM_API_KEY")
from config.constants import BASE_URL, NER_MODEL

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
    )

SYSTEM_PROMPT = """You are an expert information extraction engine. Your sole task is to analyze the provided text and extract all named entities and abstract concepts from it.

Return your output as a valid JSON object with exactly this structure:
{
  "entities": [],
  "concepts": []
}

Definitions:
- "entities": Specific, named real-world objects - people, organizations, locations, products, events, dates, technologies, etc.
- "concepts": Abstract or generalized ideas, themes, theories, phenomena, or domain knowledge referenced in the text.

Rules:
1. Output ONLY the raw JSON object. No preamble, explanation, markdown, or code fences.
2. Each item must be a short noun phrase (1 to 4 words max).
3. No duplicates. Normalize case (title case for entities, lowercase for concepts).
4. If a category has no items, return an empty array [].
5. Do not infer or hallucinate - extract only what is explicitly present or strongly implied in the text.
"""


def _parse_json_object(raw_content: str) -> dict[str, Any]:
  cleaned = raw_content.strip()
  if not cleaned:
    return {"entities": [], "concepts": []}

  try:
    return json.loads(cleaned)
  except json.JSONDecodeError:
    pass

  match = re.search(r"\{[\s\S]*\}", cleaned)
  if not match:
    return {"entities": [], "concepts": []}

  try:
    return json.loads(match.group(0))
  except json.JSONDecodeError:
    return {"entities": [], "concepts": []}


def _normalize_items(items: Any, *, lowercase: bool) -> list[str]:
  if not isinstance(items, list):
    return []

  seen: set[str] = set()
  normalized: list[str] = []

  for item in items:
    if not isinstance(item, str):
      continue

    value = " ".join(item.split()).strip()
    if not value:
      continue

    if lowercase:
      value = value.lower()

    dedupe_key = value.casefold()
    if dedupe_key in seen:
      continue

    seen.add(dedupe_key)
    normalized.append(value)

  return normalized


def extract_entities_concepts(text: str) -> dict[str, list[str]]:
  if not text or not text.strip():
    return {"entities": [], "concepts": []}

  response = client.chat.completions.create(
      model=NER_MODEL,
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": text}
      ],
      temperature=0.1,
      max_tokens=2048
  )

  content = response.choices[0].message.content or ""
  parsed = _parse_json_object(content)

  entities = _normalize_items(parsed.get("entities", []), lowercase=False)
  concepts = _normalize_items(parsed.get("concepts", []), lowercase=True)

  return {
    "entities": entities,
    "concepts": concepts,
  }


#Need to implement concurrency and rate limiter