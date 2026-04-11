from typing import List

from pydantic import BaseModel, Field


class Relation(BaseModel):
	source: str = Field(description="Source entity name")
	target: str = Field(description="Target entity name")
	relationship: str = Field(description="Predicate describing the relationship")


class RelationList(BaseModel):
	relations: List[Relation]
