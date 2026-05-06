# src/generation/prompt_templates.py
"""
Prompt Templates
================
All prompts live here — never hardcoded in generator or route files.
Changing a prompt is a one-line edit in this file, not a hunt across modules.

Design principles:
  1. Strict grounding — "ONLY using the context chunks below"
  2. Explicit no-answer protocol — avoids hallucination on knowledge gaps
  3. Consistent citation format — [Source N] for automated extraction
  4. Narrow role framing — "data discovery assistant" limits scope
"""

from __future__ import annotations
from typing import List, Optional


# ── System prompts ────────

RAG_SYSTEM = """\
You are a precise data discovery assistant for an engineering organization.

Rules:
  1. Answer ONLY from the context chunks provided below.
  2. Cite every factual claim inline using [Source N] matching the context headers.
  3. If the answer is not in the provided context, respond EXACTLY:
     "I don't have enough information in the available documents to answer this."
  4. Never fabricate, infer, or use outside knowledge.
  5. Reproduce code, SQL, or schema examples verbatim when present in context.

Tone: concise, technical, direct.\
"""

HYDE_SYSTEM = """\
You are a technical documentation author. Write a concise factual paragraph \
(2–3 sentences) that would appear in internal documentation answering this question. \
Documentation style. No preamble. No hedging. Write the content directly.\
"""


# ── Context builders ──────

def build_context(chunks: list) -> str:
    """
    Format chunks as a numbered, delimited context block.

    Example:
        [Source 1] users_schema.pdf  (relevance: 0.94)
        The users table stores user_id (UUID), email, created_at...

        ---

        [Source 2] etl_runbook.md  (relevance: 0.81)
        The ETL pipeline runs every 15 minutes via Databricks Workflows...
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        src   = c.display_source if hasattr(c, "display_source") else f"source_{i}"
        score = f"  (relevance: {c.score:.2f})" if getattr(c, "score", 0) > 0 else ""
        text  = c.text.strip() if hasattr(c, "text") else str(c)
        parts.append(f"[Source {i}] {src}{score}\n{text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    query:        str,
    chunks:       list,
    chat_history: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Assemble the full message list for the OpenAI chat endpoint.

    Args:
        query:        user question
        chunks:       RetrievedChunk list (after reranking)
        chat_history: prior turns for multi-turn support

    Returns:
        List of {"role": ..., "content": ...} dicts
    """
    context     = build_context(chunks)
    user_content = (
        f"Context documents:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )
    messages = [{"role": "system", "content": RAG_SYSTEM}]
    if chat_history:
        messages.extend(chat_history[-6:])   # keep last 3 exchanges
    messages.append({"role": "user", "content": user_content})
    return messages


def no_results_message() -> str:
    return (
        "I don't have enough information in the available documents to answer this question. "
        "Please rephrase your question, or verify that the relevant documents have been ingested."
    )
