from typing import List
from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(..., description="The unique identifier or name of the entity.")
    label: str = Field(..., description="The category of the entity (e.g., Person, Tech, Concept).")
    description: str = Field(..., description="A short property defining what this entity is.")

class Relationship(BaseModel):
    source: str = Field(..., description="The name of the starting entity.")
    target: str = Field(..., description="The name of the ending entity.")
    relation_type: str = Field(..., description="The type of connection (e.g., MENTIONS, WORKS_AT).")
    weight: float = Field(..., description="Strength of relationship between 0.0 and 1.0 based on context relevance.")

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]