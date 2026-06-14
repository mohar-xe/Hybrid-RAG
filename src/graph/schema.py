"""Pydantic models for graph entities and relationships."""

from typing import Literal

from pydantic import BaseModel, field_validator

VALID_RELATION_TYPES = {
    "implements", "trained_on", "evaluates", "part_of", "introduces",
    "extends", "depends_on", "contrasts_with", "applied_to", "measured_by",
    "founded_by", "developed_by", "defined_as", "consists_of", "is_type_of",
    "based_on", "used_for", "created_by", "located_in", "predecessor_of",
}


class Node(BaseModel):
    title: str
    type: Literal["entity", "concept"]


class Relation(BaseModel):
    type: str
    weight: float

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v not in VALID_RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {v}")
        return v

    @field_validator("weight")
    @classmethod
    def check_weight(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return round(v, 2)


class Triplet(BaseModel):
    source: Node
    relation: Relation
    target: Node
