import os
from typing import List
from openai import OpenAI
from config.constants import NER_MODEL
from graph.schema import RelationList

client = OpenAI(api_key=os.environ.get("NIM_API_KEY"))

def extract_relations(text: str, entities: List[dict]) -> List[dict]:
    valid_entity_names = [e["name"] for e in entities]
    
    prompt = f"""
    You are an expert knowledge graph extraction system.
    Based on the text provided, identify the relationships between the provided list of entities.
    
    CRITICAL INSTRUCTION: 
    The 'source' and 'target' fields MUST exactly match one of the names in the provided Valid Entities list. 
    Do not invent new entities.

    Valid Entities:
    {valid_entity_names}

    Text to analyze:
    {text}
    """

    response = client.beta.chat.completions.parse(
        model=NER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=RelationList,
        temperature=0.1
    )
    
    relations = []
    for rel in response.choices[0].message.parsed.relations:
        if rel.source in valid_entity_names and rel.target in valid_entity_names:
            relations.append(rel.model_dump())
        else:
            print(f"Skipping invalid relation: {rel.source} -> {rel.target} (Entity not in node list)")
            
    return relations