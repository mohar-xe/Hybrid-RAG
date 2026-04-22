def build_context(retrieved_chunks: list[dict], query: str) -> str:
    context_parts = []
    context_parts.append("Retrieved Context\n")
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        content = chunk.get('content', '')
        source = chunk.get('source', 'unknown')
        score = chunk.get('score', 0)
        
        context_parts.append(f"[{i}] (from {source}, score: {score:.2f})")
        context_parts.append(content)
        context_parts.append("")
    
    context_parts.append("End Context\n")
    context_parts.append(f"Question: {query}\n")
    
    return "\n".join(context_parts)