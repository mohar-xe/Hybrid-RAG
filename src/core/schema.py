from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
import uuid

class MemoryUnit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source_type: str
    timestamp: datetime
    entities: List[str] = []
    concepts: List[str] = []
    emotional_weight: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    stability: float = 1.0
    decay_score: float = 1.0
    embedding_ids: List[str] = []
    metadata: dict = {}