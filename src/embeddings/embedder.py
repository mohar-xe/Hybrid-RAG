from typing import Any, Mapping
import numpy as np
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[Mapping[str, Any]], batch_size: int = 32) -> list[dict[str, Any]]:

    text_chunks = [chunk["text"] for chunk in chunks]

    try:
        enriched_chunks: list[dict[str, Any]] = []
        total_chunks = len(text_chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=True, normalize_embeddings=True)
            chunk_batch = chunks[i:i + batch_size]

            for chunk, embedding in zip(chunk_batch, batch_embeddings):
                chunk_with_embedding = dict(chunk)
                chunk_with_embedding["embeddings"] = np.array(embedding).astype("float32").tolist()
                enriched_chunks.append(chunk_with_embedding)
        
        LOGGER.info(f"Successfully embedded {total_chunks} chunks in batches of {batch_size}.")
        return enriched_chunks
    except Exception as exc:
        LOGGER.exception(f"Error embedding chunks: {exc}")
        raise RuntimeError(f"Failed to embed chunks: {exc}") from exc