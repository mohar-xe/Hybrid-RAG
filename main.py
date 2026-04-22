import os
import logging
from os import path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

from ingestion.chunker import chunk_text
from embeddings.embedder import embed_chunks
from graph.entity_extraction import extract_entities_concepts
from graph.relation_extraction import extract_relations
from retrieval.kuzu import KuzuGraphRetriever
from retrieval.pgvector import store_vector, retrieve_relevant_chunks
from context.builder import build_context
from llm.generator import generate_with_citations

def ingest(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise ValueError("Provided path is not a file")
    
    if not path.endswith('.pdf'):
        raise ValueError("Only PDF files are supported")
    
    chunks = chunk_text(path)
    return chunks

def add_pgvector(chunks: list[dict]) -> list[dict]:
    enriched_chunks = embed_chunks(chunks)
    store_vector(enriched_chunks)
    return enriched_chunks

def _process_chunk_graph(content: str):
    entities = extract_entities_concepts(content)
    relations = extract_relations(content, entities)
    return entities, relations

def add_graph(chunks: list[dict]) -> None:
    contents = [chunk['content'] for chunk in chunks]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_chunk_graph, contents))
        
    list_entities = [res[0] for res in results]
    list_relations = [res[1] for res in results]
        
    graph_db = KuzuGraphRetriever()
    
    entity_label_map = {}
    

    for chunk_entities in list_entities:
        for entity in chunk_entities:
            name = entity.get("name")
            label = entity.get("label", "Entity")
            desc = entity.get("description", "")
            
            if not name:
                continue
                
            entity_label_map[name] = label
            
            properties = {
                "name": name,
                "description": desc
            }
            graph_db.upsert_node(label=label, primary_key="name", properties=properties)
            
    for chunk_relations in list_relations:
        for rel in chunk_relations:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("relation_type", "RELATED_TO")
            weight = rel.get("weight", 1.0)
            
            if not source or not target:
                continue
                
            src_label = entity_label_map.get(source)
            dst_label = entity_label_map.get(target)
            
            if src_label and dst_label:
                graph_db.upsert_relationship(
                    src_label=src_label,
                    src_pk="name",
                    src_val=source,
                    dst_label=dst_label,
                    dst_pk="name",
                    dst_val=target,
                    rel_label=rel_type,
                    properties={"weight": weight}
                )

def query(pdf_path: str, question: str) -> dict:

    LOGGER.info(f"Ingesting {pdf_path}...")
    chunks = ingest(pdf_path)
    
    LOGGER.info("Adding chunks to vector store...")
    add_pgvector(chunks)
    
    LOGGER.info("Building knowledge graph...")
    add_graph(chunks)

    LOGGER.info(f"Retrieving context for: {question}")
    retrieved_chunks = retrieve(question, top_k=5)

    LOGGER.info("Generating answer...")
    result = generate_with_citations(question, retrieved_chunks)
    
    return result