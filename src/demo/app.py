"""Gradio demo for Hybrid-RAG — shows retrieval internals for technical demos."""

import gradio as gr
from config.settings import get_settings
from config.init_db import init_db
from embeddings.embedder import embedder
from reasoning.router import classify_query
from reasoning.query_interpreter import interpret_query
from retrieval.pgvector import (
    hard_filter_docs,
    doc_level_soft_rank,
    document_routed_search,
)
from retrieval.kuzu_store import get_entity_context, structural_expansion
from retrieval.reranker import rerank
from context.builder import build_context
from llm.generator import generate
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()
init_db()

# Custom CSS for a professional tool aesthetic
CUSTOM_CSS = """
/* Theme tokens - restrained professional palette */
:root {
    --bg-surface: #0f1419;
    --bg-elevated: #1a2028;
    --bg-interactive: #242d38;
    --text-primary: #e7eaed;
    --text-secondary: #8b98a5;
    --text-muted: #5c6b7a;
    --accent-primary: #1d9bf0;
    --accent-success: #00ba7c;
    --accent-warning: #f59e0b;
    --border-default: #2f3942;
    --border-subtle: #232b35;
}

/* Base styling */
.gradio-container { max-width: 1400px !important; }
body { background: var(--bg-surface) !important; }

/* Typography */
h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
.prose { line-height: 1.6; max-width: 75ch; }

/* Input styling */
.input-box textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.input-box textarea:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(29, 155, 240, 0.15) !important;
    outline: none !important;
}
.input-box textarea::placeholder { color: var(--text-muted) !important; }

/* Button styling */
.primary-btn {
    background: var(--accent-primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
}
.primary-btn:hover { background: #1a8cd8 !important; }
.primary-btn:active { transform: scale(0.98) !important; }

/* Tab styling */
.tabs { border-bottom: 1px solid var(--border-subtle) !important; margin-bottom: 16px !important; }
.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.tab-nav button:hover { color: var(--text-primary) !important; }
.tab-nav button.selected {
    color: var(--accent-primary) !important;
    border-bottom-color: var(--accent-primary) !important;
}

/* Data table styling */
.dataframe {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}
.dataframe th {
    background: var(--bg-interactive) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid var(--border-default) !important;
}
.dataframe td {
    color: var(--text-primary) !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}
.dataframe tr:hover td { background: var(--bg-interactive) !important; }

/* Output containers */
.output-container {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

/* Checkbox styling */
.checkbox-wrap label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* Slider styling */
.slider-wrap input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

/* Badge/status indicators */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
}
.status-active { background: rgba(29, 155, 240, 0.15); color: var(--accent-primary); }
.status-inactive { background: var(--bg-interactive); color: var(--text-muted); }
.status-api { background: rgba(0, 186, 124, 0.15); color: var(--accent-success); }
.status-local { background: rgba(245, 158, 11, 0.15); color: var(--accent-warning); }

/* Metrics row */
.metric-row {
    display: flex;
    gap: 24px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.metric-item { display: flex; flex-direction: column; gap: 2px; }
.metric-label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: var(--text-primary); font-size: 20px; font-weight: 600; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
"""


def query_rag(
    question: str, use_reranker: bool, use_graph: bool, use_siblings: bool, top_k: int
):
    """Run RAG pipeline with toggles and return detailed intermediates."""
    if not question.strip():
        return ("Enter a question to start.", "", "", [], [], [], "", "", "")

    complexity = classify_query(question)
    interpreted = interpret_query(question)
    semantic_query = interpreted.get("semantic_query", question)
    filters = interpreted.get("filters", {})
    query_emb = embedder([semantic_query])[0]

    # Stage 1: hard filter
    doc_ids = hard_filter_docs(
        doc_type=filters.get("doc_type"),
        is_latest=filters.get("is_latest"),
        date_after=filters.get("date_after"),
    )
    hard_filter_count = len(doc_ids)

    # Stage 2: doc-level soft ranking
    ranked_docs = doc_level_soft_rank(query_emb, semantic_query, doc_ids)
    if ranked_docs:
        doc_ids = [d[0] for d in ranked_docs]
        doc_rank_map = {d[0]: i + 1 for i, d in enumerate(ranked_docs)}
        doc_stage = [
            (i + 1, d[0][:8], f"{d[1]:.4f}") for i, d in enumerate(ranked_docs[:10])
        ]
    else:
        doc_rank_map = None
        doc_stage = []

    # Stage 3-4: chunk-level hybrid search + RRF fusion
    chunks = document_routed_search(
        query_emb, semantic_query, doc_ids, doc_rank_map=doc_rank_map
    )
    pre_rerank = [
        (i + 1, round(c.score, 4), c.source_id[:40], c.text[:80] + "...")
        for i, c in enumerate(chunks[:10])
    ]

    # Stage 5: rerank
    if use_reranker and chunks:
        chunks = rerank(semantic_query, chunks, top_k=top_k)

    # Stage 6: structural expansion
    sibling_chunks = []
    if use_siblings and chunks:
        seed_ids = [c.chunk_id for c in chunks]
        try:
            sibling_chunks = structural_expansion(seed_ids)
            if sibling_chunks:
                LOGGER.info("Structural expansion: %d siblings.", len(sibling_chunks))
        except Exception as e:
            LOGGER.warning(f"Structural expansion skipped ({e})")

    # Stage 7: graph facts
    graph_facts = ""
    graph_entities = []
    if use_graph:
        from graph.entity_extraction import extract_query_entities

        entities = extract_query_entities(question)
        if entities:
            try:
                graph_facts = get_entity_context(entities)
                graph_entities = entities[:10]
            except Exception as e:
                LOGGER.warning(f"Graph lookup skipped ({e})")
                graph_facts = ""

    context, citations = build_context(chunks, graph_facts, max_tokens=1500)
    answer = generate(question, context, stream=False)

    emb_backend = settings.embedding.backend
    rerank_backend = settings.reranker.backend
    gen_backend = settings.generator.backend

    routing = f"""<div class="metric-row">
<div class="metric-item"><span class="metric-label">Complexity</span><span class="metric-value">{complexity.value.upper()}</span></div>
<div class="metric-item"><span class="metric-label">Filters</span><span class="metric-value" style="font-size:13px;">{"&#183;".join(k for k, v in filters.items() if v) or "none"}</span></div>
<div class="metric-item"><span class="metric-label">Embed</span><span class="status-badge {"status-api" if emb_backend == "api" else "status-local"}">{emb_backend.upper()}</span></div>
<div class="metric-item"><span class="metric-label">Rerank</span><span class="status-badge {"status-api" if rerank_backend == "api" and use_reranker else "status-local" if rerank_backend != "api" and use_reranker else "status-inactive"}">{"ON" if use_reranker else "OFF"}</span></div>
<div class="metric-item"><span class="metric-label">Generate</span><span class="status-badge {"status-api" if gen_backend == "api" else "status-local"}">{gen_backend.upper()}</span></div>
<div class="metric-item"><span class="metric-label">Graph</span><span class="status-badge {"status-active" if use_graph else "status-inactive"}">{"ON" if use_graph else "OFF"}</span></div>
<div class="metric-item"><span class="metric-label">Top-K</span><span class="metric-value">{top_k}</span></div>
</div>"""

    retrieval = f"""<div style="color: var(--text-secondary); font-size: 13px;">
<span style="color: var(--text-primary); font-weight: 600;">{hard_filter_count}</span> docs (hard filter) → 
<span style="color: var(--text-primary); font-weight: 600;">{len(ranked_docs)}</span> docs (soft rank) → 
<span style="color: var(--text-primary); font-weight: 600;">{len(chunks)}</span> chunks → 
<span style="color: var(--text-primary); font-weight: 600;">{len(sibling_chunks)}</span> siblings
</div>"""

    final = [
        (c.source_id[:50], f"{c.score:.4f}", c.text[:120] + "...")
        for c in chunks[:top_k]
    ]
    citations_text = "\n".join(
        [f"[{c.ref_id}] {c.source_id} — {c.text_preview}..." for c in citations]
    )

    graph_disp = f"Entities: {', '.join(graph_entities) if graph_entities else 'None'}\n\n{graph_facts if graph_facts else 'No graph facts retrieved.'}"
    siblings_disp = (
        "\n".join([f"[{s[0][:8]}] {s[1][:100]}..." for s in sibling_chunks[:10]])
        if sibling_chunks
        else "No structural expansion applied."
    )

    return (
        answer,
        routing,
        retrieval,
        doc_stage,
        pre_rerank,
        final,
        citations_text,
        graph_disp,
        siblings_disp,
    )


def create_demo():
    with gr.Blocks(title="Hybrid-RAG Demo") as demo:
        # Header
        gr.HTML("""
<div style="padding: 24px 0 16px; border-bottom: 1px solid var(--border-subtle); margin-bottom: 24px;">
<h1 style="margin: 0 0 8px; font-size: 28px; color: var(--text-primary);">Hybrid-RAG</h1>
<p style="margin: 0; color: var(--text-secondary); font-size: 14px;">8-stage retrieval funnel — hard filter → doc soft rank → chunk hybrid → RRF fusion → rerank → graph expansion → facts → generate</p>
</div>
""")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask about your documents...",
                    lines=2,
                    elem_classes=["input-box"],
                )
                with gr.Row():
                    use_reranker = gr.Checkbox(
                        value=True,
                        label="Reranking",
                        info="Cross-encoder relevance scoring",
                    )
                    use_graph = gr.Checkbox(
                        value=True,
                        label="Knowledge Graph",
                        info="Entity context from KùzuDB",
                    )
                    use_siblings = gr.Checkbox(
                        value=True,
                        label="Structural Expansion",
                        info="Small-to-big graph traversal for sibling chunks",
                    )
                    top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K Results")

                ask_btn = gr.Button(
                    "Ask", variant="primary", size="lg", elem_classes=["primary-btn"]
                )

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_out = gr.Markdown(elem_classes=["prose"])
                with gr.Accordion("Citations", open=False):
                    citations_out = gr.Textbox(
                        lines=6, interactive=False, show_label=False
                    )

            with gr.TabItem("Routing"):
                routing_out = gr.HTML()

            with gr.TabItem("Retrieval"):
                retrieval_out = gr.HTML()
                with gr.Row():
                    doc_stage_out = gr.Dataframe(
                        headers=["Rank", "Doc ID", "Score"],
                        label="Stage 2: Doc-Level Soft Rank",
                        elem_classes=["dataframe"],
                    )
                    pre_rerank_out = gr.Dataframe(
                        headers=["Rank", "Score", "Source", "Preview"],
                        label="Stage 3-4: Chunk Candidates (pre-rerank)",
                        elem_classes=["dataframe"],
                    )

            with gr.TabItem("Final Chunks"):
                final_out = gr.Dataframe(
                    headers=["Source", "Score", "Text"],
                    label="Stage 5: Top-K After Rerank",
                    elem_classes=["dataframe"],
                )

            with gr.TabItem("Graph"):
                graph_out = gr.Textbox(
                    lines=10, interactive=False, label="Stage 7: Knowledge Graph Facts"
                )

            with gr.TabItem("Siblings"):
                siblings_out = gr.Textbox(
                    lines=10,
                    interactive=False,
                    label="Stage 6: Structural Expansion (Sibling Chunks)",
                )

        ask_btn.click(
            query_rag,
            [question, use_reranker, use_graph, use_siblings, top_k],
            [
                answer_out,
                routing_out,
                retrieval_out,
                doc_stage_out,
                pre_rerank_out,
                final_out,
                citations_out,
                graph_out,
                siblings_out,
            ],
        )

        gr.Examples(
            [
                ["What is this document about?"],
                ["How are these concepts related?"],
                ["Summarize the key findings"],
                ["latest contract terms"],
            ],
            question,
            label="Try these",
        )

    return demo


if __name__ == "__main__":
    create_demo().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
