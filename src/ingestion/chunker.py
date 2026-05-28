from typing import Literal

import numpy as np

from constants.logger import setup_logger
from ingestion.normalize import split_into_paragraphs
from constants.exceptions import ModelError, ConfigurationError
from embeddings.embedder import embedder
from ingestion.chunk_schema import Chunk

LOGGER = setup_logger(__name__)

def clustering(text: str, threshold: float = 0.7) -> list[list[str]]:
        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            return []

        LOGGER.info("Embedding paragraphs for clustering.")    
        embeddings = np.array(embedder(paragraphs))

        LOGGER.info("Normalizing vectors.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

        clusters = []
        current_segment = [paragraphs[0]]

        LOGGER.info("Comparing neighbours.")
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                current_segment.append(paragraphs[idx + 1])
            else:
                clusters.append(current_segment)
                current_segment = [paragraphs[idx + 1]]

        clusters.append(current_segment)
        LOGGER.info(f"Cluster created with {len(clusters)} segmnts.")
        return clusters

def split_large_chunks(paragraphs: list[str], chunk_length: int = 300, overlap: int = 2) -> list[str]:
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    except Exception as e:
        LOGGER.error(f"Unable to fetch model: en_core_web_sm")
        raise ModelError(f"Unable to fetch model: {e}")

    text = " ".join(paragraphs)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    sent_entities: list[set[str]] = []
    for sent in doc.sents:
        ents = {ent.text.lower() for ent in sent.ents}
        sent_entities.append(ents)

    chunks = []
    start = 0

    while start < len(sentences):
        word_count = 0
        end = start

        while end < len(sentences) and word_count < chunk_length:
            word_count += len(sentences[end].split())
            end += 1

        if end >= len(sentences):
            chunks.append(" ".join(sentences[start:]))
            break

        window_entities: set[str] = set()
        for i in range(start, end):
            window_entities |= sent_entities[i]

        cut = end
        effective_overlap = overlap

        if not window_entities:
            effective_overlap = overlap + 2
        else:
            for i in range(end, min(end + 5, len(sentences))):
                if not sent_entities[i] & window_entities:
                    cut = i
                    break

        chunks.append(" ".join(sentences[start:cut]))
        start = max(cut - effective_overlap, start + 1)

    LOGGER.info(f"Split oversized cluster into {len(chunks)} chunks.")
    return chunks


def chunk_text(text: str, threshold: float = 0.7, chunk_length: int = 300) -> list[str]:
    clusters = clustering(text, threshold)
    chunks = []

    for cluster in clusters:
        word_count = sum(len(p.split()) for p in cluster)
        if word_count > chunk_length:
            chunks.extend(split_large_chunks(cluster, chunk_length))
        else:
            chunks.append(" ".join(cluster))

    LOGGER.info(f"Final chunk count: {len(chunks)}")
    return chunks


def extract_keyphrases(text: str) -> list[str]:
    if not text:
        LOGGER.warning("Empty text passed for keyphrases extraction.")
        return []
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedup_lim=0.1,
            top=10
        )
    except:
        LOGGER.error(f"Dependency not installed. 'uv add -r requirements'")
        raise ConfigurationError(f"Dependency not installed. 'uv add -r requirements'")
    
    keywords_score = kw_extractor.extract_keywords(text)
    keywords = [kw for kw, score in keywords_score]
    return keywords

def chunk_enrich(chunks: list[str], source: Literal['PDF', 'Reel', 'Youtube'], id: str) -> list[Chunk]:

    import uuid

    from embeddings.embedder import embedder
    embeddings = embedder(chunks)

    finalized_chunks = []

    for idx, chunk in enumerate(chunks):
        enriched = Chunk(
            chunk_id = uuid.uuid1(),
            text = chunk,
            embeddings = embeddings[idx],
            source_type = source,
            source_id = id,
            chunk_index=idx,
            keyword = extract_keyphrases(chunk))

        finalized_chunks.append(enriched)
    return finalized_chunks