import os

from dotenv import load_dotenv
from openai import OpenAI

from graph.schema import KnowledgeGraph

load_dotenv()

api_key = os.environ.get("ROUTER_API_KEY")
from config.constants import BASE_URL, NER_MODEL

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
    )

SYSTEM_PROMPT = """You are a high-precision Knowledge Graph Extraction Engine. Your goal is to extract a comprehensive ontology from the provided text to populate a Kùzu graph database.
Instructions:
    1) Entity Extraction: Identify every distinct entity. Assign a label that matches its semantic category and a description that captures its essence in the text.
    2) Relationship Mapping: Identify all connections between entities.
    3) Weighting: Assign a weight (0.0 to 1.0) based on how central the relationship is to the core narrative of the text.
    Deduplication: Ensure the same entity is not extracted twice under different names. Use the most formal name as the primary name.
Strict Rule: Output MUST strictly follow the provided JSON schema. Do not include any conversational filler.
"""

kuzu_schema = KnowledgeGraph.model_json_schema()

tool = {
  "name": "extract_knowledge_graph",
  "description": "Extracts a structured knowledge graph from unstructured text, identifying entities and",
  "input_schema": kuzu_schema,
}


def extract_entities_concepts(text: str) -> list[dict]:
  if not text or not text.strip():
    return []

  response = client.chat.completions.create(
      model=NER_MODEL,
      tools=[{"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}],
      tool_choice={"type": "function", "function": {"name": "extract_knowledge_graph"}},
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": text}
      ],
      temperature=0.1,
      max_tokens=2048
  )

  try:
      tool_calls = response.choices[0].message.tool_calls
      if tool_calls:
          import json
          args = json.loads(tool_calls[0].function.arguments)
          return args.get("entities", [])
      else:
          content = response.choices[0].message.content
          if content:
              import json
              args = json.loads(content)
              return args.get("entities", [])
  except Exception as e:
      print(f"Error parsing completion: {e}")

  return []