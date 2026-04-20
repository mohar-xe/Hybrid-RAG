import os
import json
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from graph.schema import KnowledgeGraph

load_dotenv()

from config.constants import BASE_URL, NER_MODEL

api_key = os.environ.get("ROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)

SYSTEM_PROMPT = """You are an expert knowledge graph extraction system.
Based on the text provided, identify the relationships between the provided list of entities.

CRITICAL INSTRUCTION: 
The 'source' and 'target' fields MUST exactly match one of the names in the provided Valid Entities list. 
Do not invent new entities. Output MUST strictly follow the provided JSON schema.
"""

kuzu_schema = KnowledgeGraph.model_json_schema()

tool = {
    "name": "extract_relations",
    "description": "Extracts a structured list of relationships from unstructured text based on given entities",
    "input_schema": kuzu_schema,
}


def extract_relations(text: str, entities: List[dict]) -> List[dict]:
    if not text or not text.strip():
        return []

    valid_entity_names = [e["name"] for e in entities]
    
    prompt = f"""
    Valid Entities:
    {valid_entity_names}

    Text to analyze:
    {text}
    """

    response = client.chat.completions.create(
        model=NER_MODEL,
        tools=[{"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}],
        tool_choice={"type": "function", "function": {"name": "extract_relations"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2048
    )

    relations = []
    try:
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            args = json.loads(tool_calls[0].function.arguments)
            raw_relations = args.get("relationships", [])
        else:
            content = response.choices[0].message.content
            if content:
                args = json.loads(content)
                raw_relations = args.get("relationships", [])
            else:
                raw_relations = []
                
        for rel in raw_relations:
            source = rel.get("source")
            target = rel.get("target")
            if source in valid_entity_names and target in valid_entity_names:
                relations.append(rel)
            else:
                print(f"Skipping invalid relation: {source} -> {target} (Entity not in node list)")
                
    except Exception as e:
        print(f"Error parsing completion: {e}")

    return relations