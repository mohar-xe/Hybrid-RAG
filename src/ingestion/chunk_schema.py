from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    text: str
    embeddings: list[float]
    source_type: str
    source_id: str
    chunk_index: int
    keyword: list[str]
    doc_id: str | None = None
